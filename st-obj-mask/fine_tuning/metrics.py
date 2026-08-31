"""
Evaluation metrics for object mask segmentation.
"""

import torch
import torch.nn.functional as F


def compute_iou(pred, target, num_classes=2, ignore_index=None):
    """
    Compute IoU (Intersection over Union) per class.

    Args:
        pred: [B, C, H, W] logits or [B, H, W] class predictions
        target: [B, H, W] ground truth labels
        num_classes: Number of classes
        ignore_index: Class to ignore (optional)

    Returns:
        iou_per_class: [num_classes] IoU for each class
        miou: Mean IoU across classes
    """
    if pred.ndim == 4:
        # Logits: [B, C, H, W] -> [B, H, W]
        pred = pred.argmax(dim=1)

    iou_list = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou = torch.tensor(1.0 if intersection == 0 else 0.0)
        else:
            iou = intersection / union

        iou_list.append(iou)

    iou_per_class = torch.stack(iou_list)
    miou = iou_per_class.mean()

    return iou_per_class, miou


def compute_dice(pred, target, num_classes=2, ignore_index=None):
    """
    Compute Dice Score per class.

    Args:
        pred: [B, C, H, W] logits or [B, H, W] class predictions
        target: [B, H, W] ground truth labels
        num_classes: Number of classes
        ignore_index: Class to ignore (optional)

    Returns:
        dice_per_class: [num_classes] Dice for each class
        mean_dice: Mean Dice across classes
    """
    if pred.ndim == 4:
        # Logits: [B, C, H, W] -> [B, H, W]
        pred = pred.argmax(dim=1)

    dice_list = []

    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue

        pred_mask = (pred == cls).float()
        target_mask = (target == cls).float()

        intersection = (pred_mask * target_mask).sum()

        dice = (2.0 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
        dice_list.append(dice)

    dice_per_class = torch.stack(dice_list)
    mean_dice = dice_per_class.mean()

    return dice_per_class, mean_dice


def compute_pixel_accuracy(pred, target):
    """
    Compute overall pixel accuracy.

    Args:
        pred: [B, C, H, W] logits or [B, H, W] class predictions
        target: [B, H, W] ground truth labels

    Returns:
        accuracy: Pixel accuracy (0-1)
    """
    if pred.ndim == 4:
        pred = pred.argmax(dim=1)

    correct = (pred == target).sum().float()
    total = target.numel()

    accuracy = correct / total

    return accuracy


def compute_segmentation_metrics(pred_logits, target, num_classes=2):
    """
    Compute all segmentation metrics.

    Args:
        pred_logits: [B, C, H, W] model output logits
        target: [B, H, W] ground truth labels
        num_classes: Number of classes

    Returns:
        dict with all metrics
    """
    with torch.no_grad():
        iou_per_class, miou = compute_iou(pred_logits, target, num_classes)
        dice_per_class, mean_dice = compute_dice(pred_logits, target, num_classes)
        pixel_acc = compute_pixel_accuracy(pred_logits, target)

        metrics = {
            'miou': miou.item(),
            'mean_dice': mean_dice.item(),
            'pixel_accuracy': pixel_acc.item(),
            'iou_background': iou_per_class[0].item(),
            'iou_object': iou_per_class[1].item(),
            'dice_background': dice_per_class[0].item(),
            'dice_object': dice_per_class[1].item(),
        }

    return metrics
