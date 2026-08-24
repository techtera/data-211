"""
Optimizer setup: AdamW for student encoder + projection heads.
"""

import torch
from torch.optim import AdamW
from typing import Iterator


def create_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999)
) -> AdamW:
    """
    Create AdamW optimizer for distillation training.

    Args:
        parameters: Model parameters to optimize
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        betas: Adam beta parameters

    Returns:
        AdamW optimizer
    """
    optimizer = AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )

    return optimizer
