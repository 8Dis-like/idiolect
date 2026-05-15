"""Custom BPE tokenizer training for code.

We train our own BPE tokenizer on Python code rather than using a general-purpose
one. Code has unique tokenization needs:
- Indentation is semantically meaningful (we preserve whitespace patterns)
- Operators and syntax tokens should be single tokens
- Common code patterns (def, return, import, etc.) should be single tokens
- Variable names should be split into meaningful sub-words

This uses HuggingFace's `tokenizers` library for the fast Rust-based BPE implementation,
but the training pipeline and special tokens are our own design.
"""

import os
import json
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


# Special tokens for code-specific use cases
SPECIAL_TOKENS = [
    "<|pad|>",         # Padding token
    "<|bos|>",         # Beginning of sequence
    "<|eos|>",         # End of sequence
    "<|unk|>",         # Unknown token
    "<|sep|>",         # Separator (between files/functions)
    "<|indent|>",      # Indentation marker
    "<|dedent|>",      # De-indentation marker
    "<|newline|>",     # Explicit newline
    "<|fim_prefix|>",  # Fill-in-the-middle prefix
    "<|fim_suffix|>",  # Fill-in-the-middle suffix
    "<|fim_middle|>",  # Fill-in-the-middle target
    "<|user_style|>",  # Marker for personal style fine-tuning
]


def train_tokenizer(
    data_dir: str | Path,
    vocab_size: int = 32000,
    output_dir: str | Path = "artifacts/tokenizer",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train a BPE tokenizer on Python code files.

    Args:
        data_dir: Directory containing .py files or .txt files with code.
        vocab_size: Target vocabulary size.
        output_dir: Directory to save the trained tokenizer.
        min_frequency: Minimum frequency for a token to be included.

    Returns:
        Trained Tokenizer instance.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Pre-tokenizer: split on whitespace and punctuation, but preserve indentation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: add BOS/EOS tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            ("<|bos|>", 1),
            ("<|eos|>", 2),
        ],
    )

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Collect training files
    files = []
    for ext in ["*.py", "*.txt"]:
        files.extend(str(f) for f in data_dir.rglob(ext))

    if not files:
        raise ValueError(f"No .py or .txt files found in {data_dir}")

    print(f"[Tokenizer] Training on {len(files)} files, target vocab_size={vocab_size}")

    # Train
    tokenizer.train(files, trainer=trainer)

    # Save
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Save config
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "data_dir": str(data_dir),
        "num_files": len(files),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Tokenizer] Saved to {output_dir} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str | Path = "artifacts/tokenizer") -> Tokenizer:
    """Load a trained tokenizer from disk."""
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return tokenizer


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(
        train: bool = typer.Option(False, help="Train a new tokenizer"),
        vocab_size: int = typer.Option(32000, help="Vocabulary size"),
        data_dir: str = typer.Option("data/raw", help="Directory with training files"),
        output_dir: str = typer.Option("artifacts/tokenizer", help="Output directory"),
        test_text: Optional[str] = typer.Option(None, help="Test string to tokenize"),
    ):
        if train:
            tokenizer = train_tokenizer(data_dir, vocab_size, output_dir)
        else:
            tokenizer = load_tokenizer(output_dir)

        if test_text:
            encoded = tokenizer.encode(test_text)
            print(f"Tokens: {encoded.tokens}")
            print(f"IDs: {encoded.ids}")
            print(f"Length: {len(encoded.ids)}")
        else:
            # Default test
            test = 'def hello_world():\n    print("Hello, World!")\n    return True'
            encoded = tokenizer.encode(test)
            print(f"Test: {test!r}")
            print(f"Tokens ({len(encoded.ids)}): {encoded.tokens[:20]}...")

    app()
