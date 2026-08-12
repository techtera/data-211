"""
Validation utilities for Edge Mask fine-tuning.
"""

import torch
from torch.amp import autocast

from .config import DEVICE
from .losses import compute_total_loss


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, dataloader, criterion):
    """
    Evaluate the model on the validation set.

    Returns
    -------
    dict with:
        - loss: average validation loss
        - edge_ratio: mean predicted edge pixel ratio
                      (for detecting all-zero collapse)
    """

    print("\n" + "-" * 60)
    print("Validation")
    print("-" * 60)

    total_loss = 0.0
    total_samples = 0
    total_edge_ratio = 0.0

    for images, masks in dataloader:

        # --------------------------------------------------------
        # Move to device
        # --------------------------------------------------------

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        batch_size = images.size(0)
        total_samples += batch_size

        # --------------------------------------------------------
        # Forward (need logits for loss, so use train mode briefly)
        # --------------------------------------------------------

        with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):

            model.train()
            model.feature_extractor.aggregator.eval()

            logits, ds1_logits, ds2_logits = model(images)

            loss = compute_total_loss(
                logits, ds1_logits, ds2_logits,
                masks, criterion,
            )

            model.eval()

        # --------------------------------------------------------
        # Edge Ratio (collapse detection)
        # --------------------------------------------------------

        preds = torch.sigmoid(logits)
        edge_ratio = (preds > 0.5).float().mean().item()

        total_loss += loss.item() * batch_size
        total_edge_ratio += edge_ratio * batch_size

    # ============================================================
    # Results
    # ============================================================

    avg_loss = total_loss / max(total_samples, 1)
    avg_edge_ratio = total_edge_ratio / max(total_samples, 1)

    results = {
        "loss": avg_loss,
        "edge_ratio": avg_edge_ratio,
    }

    print(f"Validation Loss    : {avg_loss:.4f}")
    print(f"Edge Ratio         : {avg_edge_ratio:.4f}")

    return results
