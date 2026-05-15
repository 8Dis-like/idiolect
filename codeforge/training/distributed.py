"""Distributed training utilities for PyTorch DDP.

Handles:
- Multi-GPU setup with NCCL backend
- Process group initialization
- Gradient synchronization
- Mixed-precision training context
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[DDP] Initialized: world_size={world_size}")

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: torch.nn.Module,
    local_rank: int,
    find_unused_parameters: bool = False,
) -> DDP:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap.
        local_rank: Local GPU rank for this process.
        find_unused_parameters: Set True if model has unused params (e.g., LoRA).

    Returns:
        DDP-wrapped model.
    """
    model = model.to(local_rank)
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
