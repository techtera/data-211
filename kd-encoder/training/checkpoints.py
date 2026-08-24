"""
Checkpoint saving and loading utilities.
"""

import torch
import os
from typing import Dict, Optional


def save_checkpoint(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    save_path: str,
    projection: Optional[torch.nn.Module] = None
):
    """
    Save training checkpoint.

    Args:
        student: Student encoder
        optimizer: Optimizer state
        scheduler: LR scheduler
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        projection: Projection heads (optional, saved separately)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.current_step if hasattr(scheduler, 'current_step') else 0,
        'loss': loss,
    }

    if projection is not None:
        checkpoint['projection_state_dict'] = projection.state_dict()

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    student: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
    projection: Optional[torch.nn.Module] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        student: Student encoder
        optimizer: Optimizer (optional, for resuming)
        scheduler: LR scheduler (optional, for resuming)
        projection: Projection heads (optional)
        device: Device to load onto

    Returns:
        Checkpoint metadata dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load student
    student.load_state_dict(checkpoint['student_state_dict'])

    # Load projection if provided
    if projection is not None and 'projection_state_dict' in checkpoint:
        projection.load_state_dict(checkpoint['projection_state_dict'])

    # Load optimizer if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler if provided
    if scheduler is not None and 'scheduler_step' in checkpoint:
        scheduler.current_step = checkpoint['scheduler_step']

    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")

    return checkpoint


def save_student_only(student: torch.nn.Module, save_path: str):
    """
    Save student encoder only (after training, no projection heads).

    Args:
        student: Student encoder
        save_path: Path to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(student.state_dict(), save_path)
    print(f"\n✓ Student encoder saved: {save_path}")
