"""
Evaluation metrics for Edge Mask fine-tuning.

Metrics:
    - BF1 (Boundary F1): precision/recall within distance tolerance
    - ODS (Optimal Dataset Scale): best F1 across thresholds
    - Dice Score: 2*intersection / (pred + gt)
    - Confusion Matrix: TP, FP, FN, TN counts
"""

import torch
import torch.nn.functional as F
import numpy as np

from .config import DEVICE


# ============================================================
# Confusion Matrix
# ============================================================

def confusion_matrix(pred_binary, target_binary):
    """
    Compute TP, FP, FN, TN from binary predictions and targets.

    Parameters
    ----------
    pred_binary : Tensor [N, 1, H, W] (0 or 1)
    target_binary : Tensor [N, 1, H, W] (0 or 1)

    Returns
    -------
    dict with tp, fp, fn, tn counts
    """

    pred = pred_binary.bool()
    target = target_binary.bool()

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ============================================================
# Dice Score
# ============================================================

def dice_score(pred_binary, target_binary, smooth=1e-6):
    """
    Compute Dice coefficient.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)

    Parameters
    ----------
    pred_binary : Tensor [N, 1, H, W] (0 or 1)
    target_binary : Tensor [N, 1, H, W] (0 or 1)

    Returns
    -------
    float
    """

    intersection = (pred_binary * target_binary).sum().item()
    pred_sum = pred_binary.sum().item()
    target_sum = target_binary.sum().item()

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice


# ============================================================
# BF1 (Boundary F1)
# ============================================================

def _dilate_mask(mask, radius):
    """
    Dilate a binary mask by `radius` pixels using max pooling.

    Parameters
    ----------
    mask : Tensor [N, 1, H, W] (0 or 1, float)
    radius : int

    Returns
    -------
    Tensor [N, 1, H, W] dilated mask
    """

    kernel_size = 2 * radius + 1
    padding = radius

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return dilated


def boundary_f1(pred_binary, target_binary, tolerance=2):
    """
    Compute Boundary F1 score.

    A predicted edge pixel is a true positive if it falls within
    `tolerance` pixels of a ground truth edge pixel (and vice versa).

    Parameters
    ----------
    pred_binary : Tensor [N, 1, H, W] (0 or 1, float)
    target_binary : Tensor [N, 1, H, W] (0 or 1, float)
    tolerance : int (dilation radius in pixels)

    Returns
    -------
    dict with precision, recall, f1
    """

    target_dilated = _dilate_mask(target_binary, tolerance)
    pred_dilated = _dilate_mask(pred_binary, tolerance)

    # Precision: predicted edges that fall within tolerance of GT
    tp_precision = (pred_binary * target_dilated).sum().item()
    pred_count = pred_binary.sum().item()

    precision = tp_precision / max(pred_count, 1e-6)

    # Recall: GT edges that fall within tolerance of prediction
    tp_recall = (target_binary * pred_dilated).sum().item()
    target_count = target_binary.sum().item()

    recall = tp_recall / max(target_count, 1e-6)

    # F1
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# ODS (Optimal Dataset Scale)
# ============================================================

def optimal_dataset_scale(pred_probs, target_binary, thresholds=None):
    """
    Compute ODS: the best F1 score across multiple thresholds.

    For each threshold, binarize predictions and compute Dice F1.
    Return the threshold that gives the highest F1.

    Parameters
    ----------
    pred_probs : Tensor [N, 1, H, W] (sigmoid probabilities)
    target_binary : Tensor [N, 1, H, W] (0 or 1)
    thresholds : list of float, or None for default range

    Returns
    -------
    dict with best_threshold, best_f1, all_scores
    """

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05).tolist()

    best_f1 = 0.0
    best_threshold = 0.5
    all_scores = []

    for t in thresholds:

        pred_bin = (pred_probs > t).float()
        f1 = dice_score(pred_bin, target_binary)

        all_scores.append({"threshold": t, "f1": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "all_scores": all_scores,
    }


# ============================================================
# Full Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, threshold=0.5, bf1_tolerance=2):
    """
    Run full evaluation on a dataloader.

    Returns
    -------
    dict with:
        - dice: Dice score at given threshold
        - bf1: Boundary F1 dict (precision, recall, f1)
        - ods: ODS dict (best_threshold, best_f1)
        - confusion: dict (tp, fp, fn, tn)
    """

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    model.eval()

    all_probs = []
    all_targets = []

    for images, masks in dataloader:

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        # model.eval() returns sigmoid(logits)
        probs = model(images)

        # Reshape: [B, S, 1, H, W] -> [B*S, 1, H, W]
        B, S = probs.shape[:2]
        probs = probs.view(B * S, 1, probs.shape[3], probs.shape[4])
        masks = masks.view(B * S, 1, masks.shape[3], masks.shape[4])

        all_probs.append(probs)
        all_targets.append(masks)

    # --------------------------------------------------------
    # Concatenate all batches
    # --------------------------------------------------------

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # --------------------------------------------------------
    # Binarize predictions at threshold
    # --------------------------------------------------------

    pred_binary = (all_probs > threshold).float()
    target_binary = all_targets.float()

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = dice_score(pred_binary, target_binary)

    print(f"Dice Score         : {dice:.4f}")

    # --------------------------------------------------------
    # BF1
    # --------------------------------------------------------

    bf1 = boundary_f1(pred_binary, target_binary, tolerance=bf1_tolerance)

    print(f"BF1 Precision      : {bf1['precision']:.4f}")
    print(f"BF1 Recall         : {bf1['recall']:.4f}")
    print(f"BF1 F1             : {bf1['f1']:.4f}")

    # --------------------------------------------------------
    # ODS
    # --------------------------------------------------------

    ods = optimal_dataset_scale(all_probs, target_binary)

    print(f"ODS Best Threshold : {ods['best_threshold']:.2f}")
    print(f"ODS Best F1        : {ods['best_f1']:.4f}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(pred_binary, target_binary)

    print(f"Confusion Matrix   : TP={cm['tp']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}  TN={cm['tn']:,}")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "dice": dice,
        "bf1": bf1,
        "ods": ods,
        "confusion": cm,
    }

    return results
