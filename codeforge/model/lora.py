"""LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

LoRA freezes the pre-trained model weights and injects trainable rank-decomposition
matrices into each layer, dramatically reducing the number of trainable parameters.

Paper: https://arxiv.org/abs/2106.09685

Instead of fine-tuning W (d_model x d_model), we learn:
    W' = W + (alpha/rank) * B @ A
where A is (d_model, rank) and B is (rank, d_model), both randomly initialized.
Only A and B are trained — W stays frozen.
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter.

    Wraps an existing nn.Linear and adds low-rank trainable matrices.
    The original linear weights are frozen; only LoRA weights are trained.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # LoRA matrices: A projects down (in -> rank), B projects up (rank -> out)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (frozen)
        base_out = self.base_linear(x)
        # LoRA forward (trainable)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base linear for efficient inference."""
        merged = nn.Linear(
            self.base_linear.in_features,
            self.base_linear.out_features,
            bias=self.base_linear.bias is not None,
        )
        merged.weight.data = (
            self.base_linear.weight.data
            + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        )
        if self.base_linear.bias is not None:
            merged.bias.data = self.base_linear.bias.data
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> nn.Module:
    """Apply LoRA adapters to specified modules in the model.

    Args:
        model: The pre-trained model to add LoRA to.
        rank: LoRA rank (lower = fewer params, higher = more capacity).
        alpha: LoRA scaling factor.
        dropout: Dropout on LoRA input.
        target_modules: List of module name suffixes to apply LoRA to.
            Defaults to attention Q, K, V, O projections.

    Returns:
        Model with LoRA adapters applied. Only LoRA params are trainable.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_count = 0
    lora_params = 0

    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Replace the linear layer with LoRA-wrapped version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_linear)

                lora_count += 1
                lora_params += sum(
                    p.numel() for p in lora_linear.parameters() if p.requires_grad
                )

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[LoRA] Applied to {lora_count} layers")
    print(f"[LoRA] Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    return model


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter weights (very small file)."""
    lora_state = {
        name: param.data
        for name, param in model.named_parameters()
        if "lora_" in name
    }
    torch.save(lora_state, path)
    size_mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / 1e6
    print(f"[LoRA] Saved {len(lora_state)} tensors ({size_mb:.1f} MB) to {path}")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights into model."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    print(f"[LoRA] Loaded adapter from {path} (missing={len(missing)}, unexpected={len(unexpected)})")
    return model
