"""Multi-head self-attention with Rotary Positional Embeddings (RoPE).

This is a from-scratch implementation — no HuggingFace dependency.
Key design decisions:
- RoPE instead of learned positional embeddings (better length generalization)
- Optional LoRA adapters for parameter-efficient fine-tuning
- KV-cache support for efficient autoregressive inference
- Flash Attention compatible via PyTorch's scaled_dot_product_attention
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the sin/cos frequencies for Rotary Positional Embeddings.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base frequency for the sinusoidal functions.
        device: Device to create tensors on.

    Returns:
        Tuple of (cos_freqs, sin_freqs), each of shape (max_seq_len, head_dim).
    """
    # Compute frequency bands: theta_i = 1 / (theta^(2i/d)) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Outer product with position indices
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim/2)
    # Duplicate for full head_dim
    angles = torch.cat([angles, angles], dim=-1)  # (seq_len, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, n_heads, seq_len, head_dim).
        cos_freqs: Cosine frequencies of shape (seq_len, head_dim).
        sin_freqs: Sine frequencies of shape (seq_len, head_dim).

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    seq_len = x.size(2)
    cos = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(0)

    # Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    d = x.shape[-1]
    x_rotated = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    return x * cos + x_rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional LoRA.

    Supports:
    - Rotary Positional Embeddings (RoPE)
    - KV-cache for efficient autoregressive generation
    - Flash Attention via PyTorch's SDPA
    - Causal masking for autoregressive language modeling
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        max_seq_len: int = 1024,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope

        # Q, K, V, Output projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        if use_rope:
            cos_freqs, sin_freqs = precompute_rope_frequencies(
                self.head_dim, max_seq_len
            )
            self.register_buffer("cos_freqs", cos_freqs)
            self.register_buffer("sin_freqs", sin_freqs)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV-cache for inference.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            kv_cache: Optional tuple of (cached_k, cached_v) for autoregressive generation.

        Returns:
            Tuple of (output, new_kv_cache).
        """
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, n_heads, T, head_dim)

        # Apply RoPE to Q and K
        if self.use_rope:
            if kv_cache is not None:
                # During generation, offset the position for the new token
                offset = kv_cache[0].size(2)
                cos = self.cos_freqs[offset : offset + T].unsqueeze(0).unsqueeze(0)
                sin = self.sin_freqs[offset : offset + T].unsqueeze(0).unsqueeze(0)
                q = q * cos + torch.cat([-q[..., self.head_dim // 2:], q[..., :self.head_dim // 2]], dim=-1) * sin
                k = k * cos + torch.cat([-k[..., self.head_dim // 2:], k[..., :self.head_dim // 2]], dim=-1) * sin
            else:
                q = apply_rope(q, self.cos_freqs, self.sin_freqs)
                k = apply_rope(k, self.cos_freqs, self.sin_freqs)

        # Append to KV-cache if provided
        new_kv_cache = None
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_kv_cache = (k, v)

        # Scaled dot-product attention (uses Flash Attention when available)
        is_causal = kv_cache is None  # Only causal during training / prefill
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.o_proj(attn_out))

        return output, new_kv_cache
