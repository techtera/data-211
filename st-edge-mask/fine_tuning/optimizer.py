"""
Optimizer for Edge Mask Fine-Tuning.
"""

import torch.optim as optim

from .config import LEARNING_RATE, WEIGHT_DECAY


# ============================================================
# Build Optimizer
# ============================================================

def build_optimizer(model):
    """
    Build the AdamW optimizer.

    Only trainable parameters are passed to the optimizer.
    The encoder (aggregator) is frozen and excluded.
    """

    print("=" * 60)
    print("Building Optimizer")
    print("=" * 60)

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    num_parameters = sum(
        p.numel()
        for p in trainable_parameters
    )

    print(f"Optimizer          : AdamW")
    print(f"Learning Rate      : {LEARNING_RATE}")
    print(f"Weight Decay       : {WEIGHT_DECAY}")
    print(f"Trainable Params   : {num_parameters:,}")
    print(f"Parameter Groups   : {len(optimizer.param_groups)}")

    return optimizer
