"""
Learning rate scheduler: Cosine annealing with warmup.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


class WarmupCosineScheduler:
    """
    Cosine LR schedule with linear warmup.

    Warmup: Linear increase from 0 to base_lr over warmup_epochs
    Cosine: Cosine decay from base_lr to min_lr over remaining epochs
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        min_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            steps_per_epoch: Steps per epoch (len(dataloader))
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.min_lr = min_lr

        self.base_lr = optimizer.param_groups[0]['lr']
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch

        self.current_step = 0

    def step(self):
        """Update learning rate (call after each optimizer.step())."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr: float = 1e-6
) -> WarmupCosineScheduler:
    """
    Create warmup + cosine scheduler.

    Args:
        optimizer: Optimizer
        warmup_epochs: Warmup duration
        total_epochs: Total epochs
        steps_per_epoch: Steps per epoch
        min_lr: Minimum LR

    Returns:
        Scheduler
    """
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=min_lr
    )

    return scheduler
