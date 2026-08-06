"""
Loss functions for semantic segmentation.
"""

import torch.nn as nn


# ============================================================
# Cross Entropy Loss
# ============================================================

def build_loss():
    """
    Build the segmentation loss function.

    Returns
    -------
    nn.Module
        CrossEntropyLoss for semantic segmentation.
    """

    print("=" * 60)
    print("Building Loss Function")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()

    print("✓ CrossEntropyLoss Initialized")

    return criterion