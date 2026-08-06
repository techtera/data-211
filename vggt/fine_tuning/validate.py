"""
Validation utilities for SegFormer fine-tuning.
"""

import torch

from .config import (
    DEVICE,
    NUM_CLASSES,
)

from .metrics import (
    confusion_matrix,
    pixel_accuracy,
    dice_score,
    mean_iou,
    precision,
    recall,
    f1_score,
)


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
):
    """
    Evaluate the model on the validation dataset.

    Returns
    -------
    dict
        Validation metrics.
    """

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    model.eval()

    running_loss = 0.0

    total_samples = 0

    global_confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64,
    )

    with torch.no_grad():

        for images, masks in dataloader:

            # ------------------------------------------------
            # Move to device
            # ------------------------------------------------

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            batch_size = images.size(0)

            total_samples += batch_size

            # ------------------------------------------------
            # VGGT expects (B,S,C,H,W)
            # ------------------------------------------------

            images = images.unsqueeze(1)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            predictions = model(images)

            logits = predictions["mask_logits"]

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                masks,
            )

            running_loss += (
                loss.item() * batch_size
            )

            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            global_confusion_matrix += confusion_matrix(
                logits,
                masks,
                NUM_CLASSES,
            )

    # ========================================================
    # Dataset-Level Metrics
    # ========================================================

    avg_loss = running_loss / total_samples

    pixel_acc = (
        torch.diag(global_confusion_matrix).sum().float()
        / (
            global_confusion_matrix.sum().float()
            + 1e-6
        )
    ).item()

    precision_score = (
        precision(global_confusion_matrix)
        .mean()
        .item()
    )

    recall_score = (
        recall(global_confusion_matrix)
        .mean()
        .item()
    )

    f1 = (
        f1_score(global_confusion_matrix)
        .mean()
        .item()
    )

    # --------------------------------------------------------
    # Dice & IoU
    # --------------------------------------------------------

    #
    # These require logits + targets.
    #
    # Since the confusion matrix has already been accumulated,
    # compute them directly from the confusion matrix.
    #

    tp = torch.diag(global_confusion_matrix).float()

    fp = global_confusion_matrix.sum(0).float() - tp

    fn = global_confusion_matrix.sum(1).float() - tp

    smooth = 1e-6

    dice = (
        (2 * tp + smooth)
        /
        (2 * tp + fp + fn + smooth)
    ).mean().item()

    iou = (
        (tp + smooth)
        /
        (tp + fp + fn + smooth)
    ).mean().item()

    # ========================================================
    # Results
    # ========================================================

    results = {

        "loss":
            avg_loss,

        "pixel_accuracy":
            pixel_acc,

        "dice":
            dice,

        "mean_iou":
            iou,

        "precision":
            precision_score,

        "recall":
            recall_score,

        "f1":
            f1,

        "confusion_matrix":
            global_confusion_matrix,
    }

    # ========================================================
    # Print Summary
    # ========================================================

    print(f"Validation Loss      : {results['loss']:.4f}")
    print(f"Pixel Accuracy       : {results['pixel_accuracy']:.4f}")
    print(f"Dice Score           : {results['dice']:.4f}")
    print(f"Mean IoU             : {results['mean_iou']:.4f}")
    print(f"Precision            : {results['precision']:.4f}")
    print(f"Recall               : {results['recall']:.4f}")
    print(f"F1 Score             : {results['f1']:.4f}")

    print("\nConfusion Matrix")
    print(global_confusion_matrix)

    return results