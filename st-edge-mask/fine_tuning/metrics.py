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


def compute_edge_metrics_multi_threshold(pred_logits, target, thresholds=[0.3, 0.5, 0.7]):
    """
    Compute metrics at multiple thresholds and return the best F1.

    Args:
        pred_logits: [B, 1, H, W] model output logits
        target: [B, 1, H, W] ground truth
        thresholds: List of thresholds to evaluate

    Returns:
        dict with metrics at best threshold
    """
    best_metrics = None
    best_f1 = 0.0

    for thresh in thresholds:
        metrics = compute_edge_metrics(pred_logits, target, threshold=thresh)
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_metrics = metrics
            best_metrics['best_threshold'] = thresh

    return best_metrics
