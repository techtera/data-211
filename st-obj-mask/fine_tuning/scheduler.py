"""
Learning rate scheduler for Object Mask fine-tuning.

Linear warmup (5% of steps) followed by cosine decay to 0.
"""

import math

import torch.optim.lr_scheduler as lr_scheduler

from .config import WARMUP_FRACTION


def build_scheduler(optimizer, total_steps):
    """
    Build a LambdaLR scheduler with:

    - Steps 0 .. warmup_steps:
        Linear ramp from 0 to base_lr

    - Steps warmup_steps .. total_steps:
        Cosine decay from base_lr to 0

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    total_steps : int
        Total number of training iterations (epochs * batches).

    Returns
    -------
    scheduler : LambdaLR
    warmup_steps : int
    """

    warmup_steps = int(WARMUP_FRACTION * total_steps)

    def lr_lambda(current_step):

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("=" * 60)
    print("Building Scheduler")
    print("=" * 60)

    print(f"Type               : Cosine with Linear Warmup")
    print(f"Total Steps        : {total_steps}")
    print(f"Warmup Steps       : {warmup_steps}")
    print(f"Warmup Fraction    : {WARMUP_FRACTION}")

    return scheduler, warmup_steps
