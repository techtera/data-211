"""
Standalone single-batch overfitting test.

This script verifies that the model can memorize a single batch.
"""

import torch
import torch.nn.functional as F

from fine_tuning.config import DEVICE, NUM_CLASSES

from fine_tuning.dataloader import build_dataloaders
from fine_tuning.model_builder import build_model
from fine_tuning.losses import build_loss
from fine_tuning.optimizer import build_optimizer
from fine_tuning.metrics import evaluate_metrics


# ============================================================
# Configuration
# ============================================================

NUM_ITERATIONS = 500

PRINT_EVERY = 10

TARGET_LOSS = 0.01


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("Single Batch Overfit Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Build Components
    # --------------------------------------------------------

    model = build_model()

    train_loader, _ = build_dataloaders()

    criterion = build_loss()

    optimizer = build_optimizer(model)

    # --------------------------------------------------------
    # Get One Batch
    # --------------------------------------------------------

    images, masks = next(iter(train_loader))

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    images = images.unsqueeze(1)

    print(f"\nBatch Size : {images.shape[0]}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    loss_history = []

    initial_loss = None

    best_loss = float("inf")

    final_metrics = None

    for iteration in range(1, NUM_ITERATIONS + 1):

        optimizer.zero_grad()

        predictions = model(images)

        logits = predictions["mask_logits"]

        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = criterion(
            logits,
            masks,
        )

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()

        optimizer.step()

        loss_history.append(loss.item())

        best_loss = min(
            best_loss,
            loss.item(),
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        final_metrics = evaluate_metrics(
            logits,
            masks,
            NUM_CLASSES,
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            iteration == 1
            or iteration % PRINT_EVERY == 0
            or iteration == NUM_ITERATIONS
        ):

            print("\n" + "-" * 60)
            print(
                f"Iteration {iteration}/{NUM_ITERATIONS}"
            )
            print("-" * 60)

            print(f"Loss             : {loss.item():.6f}")
            print(f"Pixel Accuracy   : {final_metrics['pixel_accuracy']:.4f}")
            print(f"Dice Score       : {final_metrics['dice']:.4f}")
            print(f"Mean IoU         : {final_metrics['mean_iou']:.4f}")

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 60)
    print("Overfit Test Summary")
    print("=" * 60)

    print(f"Initial Loss : {initial_loss:.6f}")
    print(f"Best Loss    : {best_loss:.6f}")
    print(f"Final Loss   : {loss_history[-1]:.6f}")

    print()

    print(f"Final Pixel Accuracy : {final_metrics['pixel_accuracy']:.4f}")
    print(f"Final Dice Score     : {final_metrics['dice']:.4f}")
    print(f"Final Mean IoU       : {final_metrics['mean_iou']:.4f}")

    print("\nConfusion Matrix")

    print(final_metrics["confusion_matrix"])

    print()

    if best_loss <= TARGET_LOSS:

        print("✓ Model successfully overfits a single batch.")

    else:

        print(
            "✗ Model did not reach the target loss."
        )
        print(
            "Consider increasing the number of iterations."
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()