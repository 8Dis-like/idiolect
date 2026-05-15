"""Unified Trainer class for pre-training and fine-tuning.

Handles the full training loop with:
- Mixed-precision (AMP with bf16/fp16)
- Gradient accumulation
- Learning rate scheduling (cosine with warmup)
- Checkpointing (save/resume)
- Weights & Biases logging
- Distributed training (DDP)
- Gradient clipping
"""

import os
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


@dataclass
class TrainerConfig:
    """Training configuration loaded from YAML."""

    # Optimization
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    # Precision
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 5000
    eval_every_steps: int = 1000

    # Logging
    wandb_project: str = "codeforge"
    wandb_run_name: str = "run"
    log_every_steps: int = 50

    # Reproducibility
    seed: int = 42


class Trainer:
    """Unified training loop for CodeForge models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        device: torch.device,
        is_distributed: bool = False,
        rank: int = 0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.is_distributed = is_distributed
        self.rank = rank
        self.is_main = rank == 0

        # Optimizer: AdamW with decoupled weight decay
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision)
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Wandb (optional — gracefully skips if not installed)
        self.wandb_run = None
        if self.is_main:
            try:
                import wandb  # type: ignore[import-not-found]
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
            except Exception as e:
                print(f"[Trainer] W&B init failed: {e}. Continuing without logging.")

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * step / self.config.warmup_steps
        # Cosine decay
        decay_ratio = (step - self.config.warmup_steps) / (
            self.config.max_steps - self.config.warmup_steps
        )
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_learning_rate + coeff * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def _update_lr(self, step: int) -> float:
        """Update optimizer learning rate."""
        lr = self._get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            # pyrefly: ignore [unexpected-keyword]
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision):
                _, loss = self.model(input_ids, targets=targets)

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 50:  # Cap eval at 50 batches for speed
                break

        self.model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if not self.is_main:
            return

        if path is None:
            path = os.path.join(self.config.checkpoint_dir, f"step_{self.global_step}.pt")

        # Get the raw model (unwrap DDP if needed)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model

        checkpoint = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": raw_model.config if hasattr(raw_model, "config") else None,
        }
        torch.save(checkpoint, path)
        print(f"[Trainer] Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            print(f"[Trainer] Best model saved: {best_path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"[Trainer] Resumed from step {self.global_step}")

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        train_iter = iter(self.train_loader)

        accum_loss = 0.0
        t_start = time.time()

        if self.is_main:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            task = progress.add_task("Training", total=self.config.max_steps)
            progress.start()

        while self.global_step < self.config.max_steps:
            # Update learning rate
            lr = self._update_lr(self.global_step)

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            for micro_step in range(self.config.gradient_accumulation):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward pass with mixed precision
                # pyrefly: ignore [unexpected-keyword]
                with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision):
                    _, loss = self.model(input_ids, targets=targets)
                    loss = loss / self.config.gradient_accumulation

                # Backward pass
                # pyrefly: ignore [missing-attribute]
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1

            # Logging
            if self.is_main and self.global_step % self.config.log_every_steps == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = (
                    self.config.batch_size
                    * self.config.gradient_accumulation
                    * self.config.log_every_steps
                    * input_ids.shape[1]
                    / elapsed
                )
                print(
                    f"Step {self.global_step} | "
                    f"Loss: {accum_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad Norm: {grad_norm:.2f} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )

                if self.wandb_run:
                    import wandb  # type: ignore[import-not-found]
                    wandb.log({
                        "train/loss": accum_loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": self.global_step,
                    })

                accum_loss = 0.0
                t_start = time.time()

            # Evaluation
            if self.global_step % self.config.eval_every_steps == 0:
                val_loss = self.evaluate()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if self.is_main:
                    print(f"Step {self.global_step} | Val Loss: {val_loss:.4f} {'(best!)' if is_best else ''}")
                    if self.wandb_run:
                        import wandb  # type: ignore[import-not-found]
                        wandb.log({"val/loss": val_loss, "val/step": self.global_step})
                    if is_best:
                        self.save_checkpoint(is_best=True)

            # Periodic checkpoint
            if self.is_main and self.global_step % self.config.save_every_steps == 0:
                self.save_checkpoint()

            if self.is_main:
                progress.update(task, completed=self.global_step)

        if self.is_main:
            progress.stop()
            self.save_checkpoint()
            print(f"[Trainer] Training complete at step {self.global_step}")
            if self.wandb_run:
                import wandb  # type: ignore[import-not-found]
                wandb.finish()
