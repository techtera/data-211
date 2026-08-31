"""
Evaluation metrics for edge mask prediction.
"""

import torch


def compute_edge_metrics(pred_logits, target, threshold=0.5):
    """
    Compute edge detection metrics: Precision, Recall, F1, IoU.

    Args:
        pred_logits: [B, 1, H, W] or [B, S, 1, H, W] model output logits
        target: [B, 1, H, W] or [B, S, 1, H, W] ground truth binary masks (0 or 1)
        threshold: Binary threshold for predictions

    Returns:
        dict with all metrics
    """
    with torch.no_grad():
        # Apply sigmoid and threshold
        pred = (torch.sigmoid(pred_logits) > threshold).float()

        # Flatten for metric computation
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp + 1e-6)

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn + 1e-6)

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # IoU: TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-6)

        # Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

        metrics = {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item(),
            'iou': iou.item(),
            'pixel_accuracy': accuracy.item(),
        }

    return metrics


def compute_dice_score(pred_logits, target, threshold=0.5):
    """
    Compute Dice Score for edge detection.

    Args:
        pred_logits: [B, 1, H, W] model output logits
        target: [B, 1, H, W] ground truth

    Returns:
        Dice score (float)
    """
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > threshold).float()

        intersection = (pred * target).sum()
        dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)

    return dice.item()


def compute_ods_metrics(pred_logits, target, thresholds=None):
    """
    Compute ODS (Optimal Dataset Scale) metrics.
    Finds the best threshold that maximizes F1 score.

    Args:
        pred_logits: [B, 1, H, W] model output logits
        target: [B, 1, H, W] ground truth
        thresholds: List of thresholds to evaluate

    Returns:
        dict with ODS metrics including best threshold and best F1
    """
    if thresholds is None:
        # Use range from 0.1 to 0.95 in steps of 0.05
        thresholds = [i * 0.05 for i in range(2, 20)]  # 0.10, 0.15, ..., 0.95

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = None

    for thresh in thresholds:
        metrics = compute_edge_metrics(pred_logits, target, threshold=thresh)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = thresh
            best_metrics = metrics

    best_metrics['ods_threshold'] = best_threshold
    best_metrics['ods_f1'] = best_f1

    return best_metrics


def compute_complete_edge_metrics(pred_logits, target):
    """
    Compute all edge detection metrics including BF1, ODS, and Dice.

    Args:
        pred_logits: [B, 1, H, W] model output logits
        target: [B, 1, H, W] ground truth

    Returns:
        dict with all metrics
    """
    # Standard metrics at 0.5 threshold (for BF1)
    bf1_metrics = compute_edge_metrics(pred_logits, target, threshold=0.5)

    # Dice score
    dice = compute_dice_score(pred_logits, target, threshold=0.5)

    # ODS metrics (optimal threshold)
    ods_metrics = compute_ods_metrics(pred_logits, target)

    # Combine all metrics
    all_metrics = {
        'dice_score': dice,
        'bf1_precision': bf1_metrics['precision'],
        'bf1_recall': bf1_metrics['recall'],
        'bf1_f1': bf1_metrics['f1_score'],
        'ods_threshold': ods_metrics['ods_threshold'],
        'ods_f1': ods_metrics['ods_f1'],
        'ods_precision': ods_metrics['precision'],
        'ods_recall': ods_metrics['recall'],
    }

    return all_metrics
