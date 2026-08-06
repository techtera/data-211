"""
Loss functions for semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        # logits : (B, C, H, W)
        # targets: (B, H, W)

        num_classes = logits.shape[1]

        probs = F.softmax(logits, dim=1)

        targets = F.one_hot(
            targets,
            num_classes=num_classes,
        )

        targets = targets.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(
            probs * targets,
            dims,
        )

        cardinality = torch.sum(
            probs + targets,
            dims,
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            cardinality + self.smooth
        )

        loss = 1.0 - dice.mean()

        return loss


# ============================================================
# Combined Loss
# ============================================================

class SegmentationLoss(nn.Module):
    """
    CrossEntropy + Dice Loss.
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=1.0,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets,
    ):

        ce_loss = self.ce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss


# ============================================================
# Build Loss
# ============================================================

def build_loss():

    print("=" * 60)
    print("Building Loss Function")
    print("=" * 60)

    criterion = SegmentationLoss()

    print("✓ CrossEntropy + Dice Loss Initialized")

    return criterion