"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

DDP is more efficient than DataParallel:
- Each GPU runs its own process (vs single process with DataParallel)
- Better scaling and memory efficiency
- Faster communication via NCCL backend
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank, world_size):
    """
    Initialize DDP process group.

    Args:
        rank: Current process rank (GPU ID)
        world_size: Total number of processes (number of GPUs)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # NVIDIA GPUs
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor):
    """
    Reduce tensor across all processes (for metrics).

    Args:
        tensor: Tensor to reduce

    Returns:
        Reduced tensor (averaged across all processes)
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt
