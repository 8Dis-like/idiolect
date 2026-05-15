"""PyTorch DataLoader for code pre-training and fine-tuning.

Handles:
- Streaming from HuggingFace datasets (The Stack v2) for pre-training
- Loading local Python files for personal style fine-tuning
- Packing multiple short sequences into one training sample for efficiency
- GitHub repo cloning and code extraction for fine-tuning data
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
# pyrefly: ignore [missing-import]
from tokenizers import Tokenizer


class CodePretrainDataset(IterableDataset):
    """Streaming dataset for pre-training on The Stack v2 Python subset.

    Uses HuggingFace datasets in streaming mode to avoid downloading the full 50GB.
    Tokenizes on-the-fly and packs sequences to max_seq_len.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        split: str = "train",
        subset: str = "small",  # "small" for testing, "full" for real training
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.subset = subset
        self.eos_id = tokenizer.token_to_id("<|eos|>")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # pyrefly: ignore [missing-import]
        from datasets import load_dataset

        # Stream The Stack v2 Python subset
        if self.subset == "small":
            # Use a smaller dataset for testing
            ds = load_dataset(
                "bigcode/starcoderdata",
                data_dir="python",
                split="train",
                streaming=True,
            )
        else:
            ds = load_dataset(
                "bigcode/the-stack-v2",
                data_dir="Python",
                split="train",
                streaming=True,
            )

        # Token buffer for packing sequences
        buffer: list[int] = []

        for sample in ds:
            code = sample.get("content", "")
            if not code or len(code) < 50:  # Skip very short files
                continue

            # Tokenize
            encoded = self.tokenizer.encode(code)
            tokens = encoded.ids

            # Add to buffer with EOS separator
            buffer.extend(tokens)
            buffer.append(self.eos_id)

            # Yield packed sequences
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "targets": targets}


class PersonalCodeDataset(Dataset):
    """Dataset for fine-tuning on personal code from GitHub repos or local directory.

    Clones GitHub repos, extracts Python files, tokenizes, and creates
    training samples for personal style adaptation.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        github_repos: Optional[list[str]] = None,
        local_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = tokenizer.token_to_id("<|eos|>")
        self.style_id = tokenizer.token_to_id("<|user_style|>")

        # Collect all Python code
        all_code = []

        if github_repos:
            for repo_url in github_repos:
                code_files = self._clone_and_extract(repo_url)
                all_code.extend(code_files)

        if local_dir:
            local_path = Path(local_dir)
            for py_file in local_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 50:  # Skip trivial files
                        all_code.append(content)
                except Exception:
                    continue

        print(f"[PersonalCodeDataset] Collected {len(all_code)} code files")

        # Tokenize and pack all code into chunks
        self.samples = self._tokenize_and_pack(all_code)
        print(f"[PersonalCodeDataset] Created {len(self.samples)} training samples")

    def _clone_and_extract(self, repo_url: str) -> list[str]:
        """Clone a GitHub repo and extract Python files."""
        code_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", repo_url, tmpdir],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                for py_file in Path(tmpdir).rglob("*.py"):
                    try:
                        content = py_file.read_text(encoding="utf-8", errors="ignore")
                        if len(content) > 50:
                            code_files.append(content)
                    except Exception:
                        continue
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"[Warning] Failed to clone {repo_url}: {e}")
        return code_files

    def _tokenize_and_pack(self, code_files: list[str]) -> list[dict[str, torch.Tensor]]:
        """Tokenize code files and pack into fixed-length training samples."""
        samples = []
        buffer: list[int] = []

        for code in code_files:
            # Add style marker token before each file
            if self.style_id is not None:
                buffer.append(self.style_id)

            encoded = self.tokenizer.encode(code)
            buffer.extend(encoded.ids)
            buffer.append(self.eos_id)

            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                samples.append({"input_ids": input_ids, "targets": targets})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def create_pretrain_dataloader(
    tokenizer: Tokenizer,
    max_seq_len: int = 1024,
    batch_size: int = 32,
    num_workers: int = 8,
    subset: str = "small",
) -> DataLoader:
    """Create DataLoader for pre-training."""
    dataset = CodePretrainDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        subset=subset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )


def create_finetune_dataloader(
    tokenizer: Tokenizer,
    max_seq_len: int = 1024,
    batch_size: int = 16,
    github_repos: Optional[list[str]] = None,
    local_dir: Optional[str] = None,
    train_split: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders for fine-tuning on personal code."""
    dataset = PersonalCodeDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        github_repos=github_repos,
        local_dir=local_dir,
    )

    # Split into train/val
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader
