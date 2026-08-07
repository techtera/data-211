"""
Evaluation metrics for semantic segmentation.
"""

import torch


# ============================================================
# Confusion Matrix
# ============================================================

def confusion_matrix(
    logits,
    targets,
    num_classes,
):
    """
    Compute the confusion matrix.

    Parameters
    ----------
    logits : Tensor
        Shape (B, C, H, W)

    targets : Tensor
        Shape (B, H, W)

    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    Tensor
        Confusion matrix of shape
        (num_classes, num_classes)

        Rows    -> Ground Truth
        Columns -> Predictions
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    valid = (
        (targets >= 0)
        &
        (targets < num_classes)
    )

    predictions = predictions[valid]
    targets = targets[valid]

    indices = (
        targets * num_classes
        + predictions
    )

    cm = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    )

    cm = cm.reshape(
        num_classes,
        num_classes,
    )

    return cm


# ============================================================
# Pixel Accuracy
# ============================================================

def pixel_accuracy(
    logits,
    targets,
):
    """
    Compute pixel accuracy.

    Returns
    -------
    float
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        predictions == targets
    ).float()

    accuracy = correct.mean()

    return accuracy.item()


# ============================================================
# Dice Score
# ============================================================

def dice_score(
    logits,
    targets,
    num_classes,
    smooth=1e-6,
):
    """
    Compute mean Dice score across classes.

    Returns
    -------
    float
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    dice_scores = []

    for cls in range(num_classes):

        pred = (
            predictions == cls
        ).float()

        target = (
            targets == cls
        ).float()

        intersection = (
            pred * target
        ).sum()

        union = (
            pred.sum()
            + target.sum()
        )

        dice = (
            2.0 * intersection
            + smooth
        ) / (
            union
            + smooth
        )

        dice_scores.append(dice)

    dice_scores = torch.stack(
        dice_scores
    )

    return dice_scores.mean().item()


# ============================================================
# Mean IoU
# ============================================================

def mean_iou(
    logits,
    targets,
    num_classes,
    smooth=1e-6,
):
    """
    Compute mean Intersection-over-Union.

    Returns
    -------
    float
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    iou_scores = []

    for cls in range(num_classes):

        pred = (
            predictions == cls
        )

        target = (
            targets == cls
        )

        intersection = (
            pred & target
        ).sum().float()

        union = (
            pred | target
        ).sum().float()

        iou = (
            intersection
            + smooth
        ) / (
            union
            + smooth
        )

        iou_scores.append(iou)

    iou_scores = torch.stack(
        iou_scores
    )

    return iou_scores.mean().item()


# ============================================================
# Precision
# ============================================================

def precision(
    cm,
):
    """
    Compute per-class precision.

    Parameters
    ----------
    cm : Tensor
        Confusion matrix.

    Returns
    -------
    Tensor
    """

    tp = torch.diag(cm)

    fp = cm.sum(dim=0) - tp

    precision = tp.float() / (
        tp + fp + 1e-6
    )

    return precision


# ============================================================
# Recall
# ============================================================

def recall(
    cm,
):
    """
    Compute per-class recall.

    Parameters
    ----------
    cm : Tensor
        Confusion matrix.

    Returns
    -------
    Tensor
    """

    tp = torch.diag(cm)

    fn = cm.sum(dim=1) - tp

    recall = tp.float() / (
        tp + fn + 1e-6
    )

    return recall


# ============================================================
# F1 Score
# ============================================================

def f1_score(
    cm,
):
    """
    Compute per-class F1 score.

    Parameters
    ----------
    cm : Tensor
        Confusion matrix.

    Returns
    -------
    Tensor
    """

    p = precision(cm)

    r = recall(cm)

    f1 = (
        2
        * p
        * r
    ) / (
        p
        + r
        + 1e-6
    )

    return f1


# ============================================================
# Evaluate All Metrics
# ============================================================

def evaluate_metrics(
    logits,
    targets,
    num_classes,
):
    """
    Compute all segmentation metrics.

    Returns
    -------
    dict
        Dictionary containing:

        - pixel_accuracy
        - dice
        - mean_iou
        - precision
        - recall
        - f1
        - confusion_matrix
    """

    cm = confusion_matrix(
        logits,
        targets,
        num_classes,
    )

    metrics = {

        "pixel_accuracy":
            pixel_accuracy(
                logits,
                targets,
            ),

        "dice":
            dice_score(
                logits,
                targets,
                num_classes,
            ),

        "mean_iou":
            mean_iou(
                logits,
                targets,
                num_classes,
            ),

        "precision":
            precision(cm).mean().item(),

        "recall":
            recall(cm).mean().item(),

        "f1":
            f1_score(cm).mean().item(),

        "confusion_matrix":
            cm,
    }

    return metrics