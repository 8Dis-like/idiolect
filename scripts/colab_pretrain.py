# ============================================================
# CodeForge: Pre-training on Google Colab Pro
# ============================================================
#
# HOW TO USE:
# 1. Go to colab.research.google.com → New Notebook
# 2. Runtime → Change runtime type → T4 GPU (or A100 if available)
# 3. Paste each section into separate cells and run in order
# 4. Training auto-saves checkpoints to Google Drive every 1000 steps
# 5. If session disconnects, just re-run — it resumes from latest checkpoint
#
# Pre-requisites:
# - Trained tokenizer (from Kaggle step) uploaded to Google Drive
#   at: /MyDrive/codeforge/artifacts/tokenizer/tokenizer.json
#
# Estimated time: ~40 hours on T4, ~12 hours on A100
# Split across multiple Colab sessions as needed
# ============================================================

# %%
# === CELL 1: Setup & Mount Drive ===

import subprocess, sys, os

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Create project directory on Drive
DRIVE_DIR = "/content/drive/MyDrive/idiolect"
os.makedirs(f"{DRIVE_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/artifacts", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/logs", exist_ok=True)

# Install dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch>=2.2.0", "tokenizers>=0.19.0", "datasets>=2.18.0",
    "wandb>=0.16.0", "rich>=13.7.0", "safetensors>=0.4.0",
], check=True)

# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %%
# === CELL 2: Clone repo & import project ===

# Clone your repo (or upload files)
# Uncomment and replace with your actual repo URL:
# !git clone https://github.com/YOUR_USERNAME/codeforge.git /content/codeforge

# If you haven't pushed to GitHub yet, upload the codeforge/ directory manually
# For now, let's define everything inline:

os.makedirs("/content/codeforge", exist_ok=True)
os.chdir("/content/codeforge")

# If cloned from git:
# sys.path.insert(0, "/content/codeforge")
# from codeforge.model import CodeForgeConfig, CodeForgeModel

# %%
# === CELL 3: Model Definition (self-contained for Colab) ===
# This is a self-contained version so the notebook works standalone.
# Once your repo is on GitHub, replace this with: from codeforge.model import ...

import math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CodeForgeConfig:
    vocab_size: int = 32000
    max_seq_len: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = False
    rope: bool = True

    def __post_init__(self):
        self.head_dim = self.d_model // self.n_heads


def precompute_rope_frequencies(head_dim, max_seq_len, theta=10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos_freqs, sin_freqs):
    seq_len = x.size(2)
    cos = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(0)
    d = x.shape[-1]
    x_rotated = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * cos + x_rotated * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, bias=False, max_seq_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        cos_freqs, sin_freqs = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos_freqs, self.sin_freqs)
        k = apply_rope(k, self.cos_freqs, self.sin_freqs)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(attn_out)), None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.dropout, config.bias, config.max_seq_len)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, kv_cache=None):
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, None


class CodeForgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("2.weight"):  # down_proj
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.emb_dropout(self.tok_emb(input_ids))
        for block in self.blocks:
            x, _ = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


print(f"✅ Model defined. Params: {CodeForgeModel(CodeForgeConfig()).count_parameters():,}")

# %%
# === CELL 4: Data Pipeline (streaming from HuggingFace) ===

from torch.utils.data import IterableDataset, DataLoader
from tokenizers import Tokenizer
import copy

# Load your trained tokenizer from Drive
TOKENIZER_PATH = f"{DRIVE_DIR}/artifacts/tokenizer/tokenizer.json"

# If tokenizer not on Drive yet, upload it:
if not os.path.exists(TOKENIZER_PATH):
    print("⚠️  Tokenizer not found on Drive!")
    print(f"   Please upload tokenizer.json to: {TOKENIZER_PATH}")
    print("   (From your Kaggle output → artifacts/tokenizer/tokenizer.json)")
    # Fallback: use a simple byte-level tokenizer for testing
    from tokenizers import Tokenizer, models, pre_tokenizers
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    VOCAB_SIZE = 256  # byte-level fallback
    print("   Using byte-level fallback tokenizer for now.")
else:
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"✅ Tokenizer loaded: vocab_size={VOCAB_SIZE}")


class StreamingCodeDataset(IterableDataset):
    """Stream Python code from HuggingFace, tokenize on-the-fly, pack sequences."""
    def __init__(self, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.token_to_id("<|eos|>") or 2

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
        buffer = []
        for sample in ds:
            code = sample.get("content", "")
            if not code or len(code) < 50:
                continue
            tokens = self.tokenizer.encode(code).ids
            buffer.extend(tokens)
            buffer.append(self.eos_id)
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]
                ids = torch.tensor(chunk[:-1], dtype=torch.long)
                tgt = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": ids, "targets": tgt}

print("✅ Data pipeline ready (streaming)")

# %%
# === CELL 5: Training Loop ===

import time
import json
from pathlib import Path

# --- Config ---
CHECKPOINT_DIR = f"{DRIVE_DIR}/checkpoints"
LOG_FILE = f"{DRIVE_DIR}/logs/training_log.jsonl"

config = CodeForgeConfig(vocab_size=VOCAB_SIZE)
model = CodeForgeModel(config).cuda()
print(f"Model: {model.count_parameters():,} params on {torch.cuda.get_device_name(0)}")

# --- WandB Init ---
import wandb
wandb.init(
    project="idiolect-pretrain",
    name="a100-run",
    config=vars(config),
    resume="allow"
)

# Training hyperparams (tuned for Colab T4)
BATCH_SIZE = 16          # Per-step batch size
GRAD_ACCUM = 8           # Effective batch = 16 * 8 = 128
MAX_STEPS = 50_000       # Total training steps
WARMUP_STEPS = 1000
LR = 3e-4
MIN_LR = 3e-5
LOG_EVERY = 50
SAVE_EVERY = 250        # Save to Drive more frequently (every ~2-5 mins on A100)
EVAL_EVERY = 500

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

# DataLoader
dataset = StreamingCodeDataset(tokenizer, max_seq_len=config.max_seq_len)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, prefetch_factor=4)

# --- Resume from checkpoint ---
start_step = 0
best_loss = float("inf")

# Find latest checkpoint
ckpt_files = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))
if ckpt_files:
    latest_ckpt = str(ckpt_files[-1])
    print(f"🔄 Resuming from {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_step = ckpt["global_step"]
    best_loss = ckpt.get("best_loss", float("inf"))
    print(f"   Resumed at step {start_step}, best_loss={best_loss:.4f}")
else:
    print("🆕 Starting fresh training")

# --- LR Schedule ---
def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    decay = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * min(decay, 1.0)))

# --- Training ---
model.train()
data_iter = iter(loader)
accum_loss = 0.0
t_start = time.time()

print(f"\n🚀 Training: steps {start_step} → {MAX_STEPS}")
print(f"   Batch={BATCH_SIZE}, GradAccum={GRAD_ACCUM}, EffBatch={BATCH_SIZE * GRAD_ACCUM}")
print(f"   Checkpoints saved to: {CHECKPOINT_DIR}")
print()

for step in range(start_step, MAX_STEPS):
    # Update LR
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    for micro in range(GRAD_ACCUM):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].cuda(non_blocking=True)
        targets = batch["targets"].cuda(non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _, loss = model(input_ids, targets=targets)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        accum_loss += loss.item()

    # Clip gradients + step
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # Logging
    if (step + 1) % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        tps = BATCH_SIZE * GRAD_ACCUM * LOG_EVERY * config.max_seq_len / elapsed
        log = {
            "step": step + 1, "loss": round(accum_loss, 4),
            "lr": f"{lr:.2e}", "grad_norm": f"{grad_norm:.2f}",
            "tokens_per_sec": f"{tps:.0f}", "elapsed": f"{elapsed:.1f}s"
        }
        print(f"Step {step+1:>6} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
              f"Grad: {grad_norm:.2f} | {tps:.0f} tok/s")

        # Append to log file on Drive
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log) + "\n")
            
        # Log to WandB
        wandb.log({
            "loss": accum_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_per_sec": tps
        }, step=step+1)

        accum_loss = 0.0
        t_start = time.time()

    # Save checkpoint to Drive
    if (step + 1) % SAVE_EVERY == 0:
        ckpt_path = f"{CHECKPOINT_DIR}/step_{step+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": step + 1,
            "best_loss": best_loss,
            "config": vars(config),
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

        # Keep only last 3 checkpoints to save Drive space
        old_ckpts = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))[:-3]
        for old in old_ckpts:
            old.unlink()

print("\n🎉 Training complete!")
print(f"   Final checkpoint: {CHECKPOINT_DIR}/step_{MAX_STEPS}.pt")
print(f"   Copy to your local machine and push to GitHub")

wandb.finish()
