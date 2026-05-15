"""Model configuration for CodeForge GPT-2 style Transformer."""

from dataclasses import dataclass


@dataclass
class CodeForgeConfig:
    """Configuration for the CodeForge Transformer model.

    Default values produce a ~200M parameter model similar to GPT-2 Medium,
    optimized for code generation with modern architectural improvements
    (RoPE, no bias, SwiGLU-optional).
    """

    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 1024

    # Architecture
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072  # 4 * d_model
    dropout: float = 0.1
    activation: str = "gelu"  # "gelu" or "swiglu"
    bias: bool = False  # Modern transformers skip bias for efficiency
    rope: bool = True  # Rotary Positional Embeddings (better than learned)

    # LoRA (only used during fine-tuning)
    lora_rank: int = 0  # 0 = disabled
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.d_model // self.n_heads

    @property
    def num_parameters(self) -> int:
        """Estimate total parameter count (excluding embeddings)."""
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.d_model * self.d_model * self.n_layers
        # FFN: up + down projections
        ffn_params = 2 * self.d_model * self.d_ff * self.n_layers
        # Layer norms
        norm_params = 2 * self.d_model * self.n_layers + self.d_model
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        return attn_params + ffn_params + norm_params + embed_params

    def __repr__(self) -> str:
        total = self.num_parameters
        if total > 1e9:
            size_str = f"{total / 1e9:.1f}B"
        else:
            size_str = f"{total / 1e6:.0f}M"
        return (
            f"CodeForgeConfig({size_str} params, "
            f"{self.n_layers}L/{self.n_heads}H/{self.d_model}D, "
            f"vocab={self.vocab_size}, seq={self.max_seq_len})"
        )


# Pre-defined model sizes for convenience
CONFIGS = {
    "small": CodeForgeConfig(n_layers=6, n_heads=6, d_model=384, d_ff=1536),   # ~30M
    "medium": CodeForgeConfig(n_layers=12, n_heads=12, d_model=768, d_ff=3072), # ~200M
    "large": CodeForgeConfig(n_layers=24, n_heads=16, d_model=1024, d_ff=4096), # ~400M
}
