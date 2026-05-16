"""CodeForge GPT-2 Style Transformer — built from scratch.

This is NOT a HuggingFace wrapper. Every component is implemented manually:
- Multi-head self-attention with RoPE
- Feed-forward network with GELU activation
- Pre-LayerNorm (more stable training than post-norm)
- Tied embedding weights (input embedding = output projection)
- KV-cache for efficient autoregressive generation

Architecture follows GPT-2 with modern improvements from LLaMA/Mistral:
- RoPE instead of learned positional embeddings
- Pre-norm instead of post-norm
- No bias in linear layers
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from codeforge.model.attention import MultiHeadAttention
from codeforge.model.config import CodeForgeConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Backwards compatible implementation for PyTorch < 2.4.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    FFN(x) = Dropout(Linear(GELU(Linear(x))))

    Uses the standard GPT-2 FFN architecture:
    d_model -> d_ff -> d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture.

    Pre-norm (used in GPT-2, LLaMA, Mistral) is more stable than post-norm:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, config: CodeForgeConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
            use_rope=config.rope,
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention with residual connection
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        # Pre-norm FFN with residual connection
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class CodeForgeModel(nn.Module):
    """CodeForge: GPT-2 style causal language model for code generation.

    Built from scratch with modern architectural improvements.

    Usage:
        config = CodeForgeConfig()
        model = CodeForgeModel(config)

        # Training (causal LM)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits, loss = model(input_ids, targets=input_ids)

        # Generation
        generated = model.generate(prompt_ids, max_new_tokens=100)
    """

    def __init__(self, config: CodeForgeConfig):
        super().__init__()
        self.config = config

        # Token embedding (no positional embedding — we use RoPE)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = RMSNorm(config.d_model)

        # Output projection (tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection
        # This is a well-established technique that reduces parameters and improves performance
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled initialization to residual projections (GPT-2 style)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training or inference.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            targets: Target token IDs for loss computation (shifted by 1 internally).
            kv_caches: List of KV caches per layer for autoregressive generation.

        Returns:
            Tuple of (logits, loss). Loss is None if targets not provided.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        # Token embeddings (RoPE is applied inside attention, not here)
        x = self.emb_dropout(self.tok_emb(input_ids))  # (B, T, d_model)

        # Pass through transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_kv_caches.append(new_cache)

        # Final norm + project to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute cross-entropy loss if targets provided
        loss = None
        if targets is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-1,  # Padding token
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache, top-k, and top-p sampling.

        Args:
            input_ids: Prompt token IDs of shape (batch, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold.

        Returns:
            Generated token IDs of shape (batch, prompt_len + generated_len).
        """
        self.eval()
        B, T = input_ids.shape
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        generated = input_ids

        for step in range(max_new_tokens):
            # Use only the last token if we have KV cache, else use full sequence
            if kv_caches is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            logits, _ = self.forward(curr_input, kv_caches=kv_caches)

            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample from the filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Note: KV-cache update happens inside forward(), but for simplicity
            # in this implementation we re-process the full sequence each step.
            # A production implementation would properly manage the cache.
            # TODO: Wire up KV-cache properly for 10x inference speedup

        return generated

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
