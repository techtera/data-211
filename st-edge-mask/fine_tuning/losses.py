"""
Loss functions for Edge Mask fine-tuning.

Combines Weighted BCE + Dice Loss with deep supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    BCE_WEIGHT,
    DICE_WEIGHT,
    POS_WEIGHT_CLAMP,
    DS1_WEIGHT,
    DS2_WEIGHT,
    FINAL_WEIGHT,
    DEVICE,
)


# ============================================================
# Edge Loss (WeightedBCE + Dice)
# ============================================================

class EdgeLoss(nn.Module):
    """
    Combined loss for edge detection:

    loss = bce_weight * WeightedBCE + dice_weight * DiceLoss

    - pos_weight is computed dynamically per batch
      (neg_pixels / pos_pixels), clamped to avoid
      extreme values.

    - Dice uses sigmoid(logits), not raw logits.

    - Inputs are raw logits (no sigmoid applied).
    """

    def __init__(self):
        super().__init__()

        self.bce_weight = BCE_WEIGHT
        self.dice_weight = DICE_WEIGHT
        self.pos_weight_min = POS_WEIGHT_CLAMP[0]
        self.pos_weight_max = POS_WEIGHT_CLAMP[1]

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: [B, 1, H, W] or [B, S, 1, H, W]
            target: [B, 1, H, W] or [B, S, 1, H, W]
        """
        # --------------------------------------------------------
        # Dynamic pos_weight (per batch)
        # --------------------------------------------------------

        pos = target.sum()
        neg = target.numel() - pos

        pos_weight = (neg / (pos + 1e-6)).clamp(
            self.pos_weight_min,
            self.pos_weight_max,
        )

        # --------------------------------------------------------
        # Weighted BCE
        # --------------------------------------------------------

        # Expand pos_weight to match pred_logits shape
        if pred_logits.ndim == 5:
            # [B, S, 1, H, W]
            pos_weight_expanded = pos_weight.view(1, 1, 1, 1, 1).expand_as(pred_logits)
        else:
            # [B, 1, H, W]
            pos_weight_expanded = pos_weight.view(1, 1, 1, 1).expand_as(pred_logits)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits,
            target,
            pos_weight=pos_weight_expanded,
        )

        # --------------------------------------------------------
        # Dice Loss
        # --------------------------------------------------------

        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()

        dice = 1.0 - (
            (2.0 * intersection + 1e-6)
            / (pred.sum() + target.sum() + 1e-6)
        )

        # --------------------------------------------------------
        # Combined
        # --------------------------------------------------------

        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================
# Total Loss with Deep Supervision
# ============================================================

def compute_total_loss(final_logits, ds1_logits, ds2_logits, target, loss_fn):
    """
    Compute total training loss with deep supervision weights:

        L = FINAL_WEIGHT * loss(final)
          + DS2_WEIGHT   * loss(ds2)
          + DS1_WEIGHT   * loss(ds1)

    DS1 and DS2 provide auxiliary gradient signal to
    intermediate decoder nodes without over-constraining them.
    """

    loss_final = loss_fn(final_logits, target)
    loss_ds1 = loss_fn(ds1_logits, target)
    loss_ds2 = loss_fn(ds2_logits, target)

    total = (
        FINAL_WEIGHT * loss_final
        + DS2_WEIGHT * loss_ds2
        + DS1_WEIGHT * loss_ds1
    )

    return total


# ============================================================
# Wrapper for Training (Deep Supervision)
# ============================================================

class DeepSupervisionEdgeLoss(nn.Module):
    """
    Wrapper that combines EdgeLoss with deep supervision.
    Accepts 4 arguments: final_logits, ds1_logits, ds2_logits, target.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = EdgeLoss()

    def forward(self, final_logits, ds1_logits, ds2_logits, target):
        """
        Args:
            final_logits: [B, 1, H, W] or [B, S, 1, H, W]
            ds1_logits: [B, 1, H, W] or [B, S, 1, H, W]
            ds2_logits: [B, 1, H, W] or [B, S, 1, H, W]
            target: [B, 1, H, W] or [B, S, 1, H, W]

        Returns:
            Combined loss scalar
        """
        return compute_total_loss(final_logits, ds1_logits, ds2_logits, target, self.loss_fn)


# ============================================================
# Build Loss
# ============================================================

def build_loss():
    """
    Build the EdgeLoss criterion with deep supervision wrapper.
    """

    print("=" * 60)
    print("Building Loss Function")
    print("=" * 60)

    criterion = DeepSupervisionEdgeLoss().to(DEVICE)

    print(f"BCE Weight         : {BCE_WEIGHT}")
    print(f"Dice Weight        : {DICE_WEIGHT}")
    print(f"Pos Weight Clamp   : {POS_WEIGHT_CLAMP}")
    print(f"DS Weights         : final={FINAL_WEIGHT}, ds2={DS2_WEIGHT}, ds1={DS1_WEIGHT}")
    print(f"Device             : {DEVICE}")

    return criterion
