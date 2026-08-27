import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight_clamp=(5, 25)):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_min = pos_weight_clamp[0]
        self.pos_weight_max = pos_weight_clamp[1]

    def forward(self, pred_logits, target):
        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(self.pos_weight_min, self.pos_weight_max)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target,
            pos_weight=pos_weight.view(1, 1, 1, 1).expand_as(pred_logits),
        )

        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()
        dice = 1.0 - (2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

        return self.bce_weight * bce + self.dice_weight * dice


def compute_total_loss(final_logits, ds1_logits, ds2_logits, target, loss_fn):
    loss_final = loss_fn(final_logits, target)
    loss_ds1 = loss_fn(ds1_logits, target)
    loss_ds2 = loss_fn(ds2_logits, target)
    return 1.0 * loss_final + 0.2 * loss_ds2 + 0.1 * loss_ds1
