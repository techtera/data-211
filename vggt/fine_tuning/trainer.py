"""
Training utilities for SegFormer fine-tuning.
"""

import torch.nn.functional as F

from .config import (
    DEVICE,
    LOG_EVERY,
)

from .validate import validate

from .checkpoints import (
    save_latest_checkpoint,
    save_best_checkpoint,
    save_epoch_checkpoint,
    save_final_checkpoint,
)


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    epoch,
):
    """
    Train the model for one epoch.
    """

    print("\n" + "=" * 60)
    print(f"Epoch {epoch}")
    print("=" * 60)

    model.train()

    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # VGGT expects (B,S,C,H,W)

        images = images.unsqueeze(1)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % LOG_EVERY == 0
            or batch_idx == 0
        ):

            print(
                f"Batch [{batch_idx + 1:03d}/{len(dataloader):03d}] "
                f"Loss : {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(dataloader)

    print(f"\nAverage Training Loss : {epoch_loss:.4f}")

    return epoch_loss


# ============================================================
# Complete Training Loop
# ============================================================

def train(
    model,
    train_loader,
    criterion,
    optimizer,
    writer,
    num_epochs,
    val_loader=None,
):
    """
    Complete training loop.
    """

    history = []

    best_iou = -1.0

    print("\nStarting Fine-Tuning...\n")

    for epoch in range(1, num_epochs + 1):

        # ====================================================
        # Training
        # ====================================================

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        history.append(train_loss)

        # ====================================================
        # Validation
        # ====================================================

        val_results = None

        if val_loader is not None:

            val_results = validate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
            )

        # ====================================================
        # TensorBoard
        # ====================================================

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Learning Rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        if val_results is not None:

            writer.add_scalar(
                "Loss/Validation",
                val_results["loss"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/PixelAccuracy",
                val_results["pixel_accuracy"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/Dice",
                val_results["dice"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/MeanIoU",
                val_results["mean_iou"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/Precision",
                val_results["precision"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/Recall",
                val_results["recall"],
                epoch,
            )

            writer.add_scalar(
                "Metrics/F1",
                val_results["f1"],
                epoch,
            )

        # ====================================================
        # Latest Checkpoint
        # ====================================================

        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=train_loss,
        )

        # ====================================================
        # Best Checkpoint
        # ====================================================

        if val_results is not None:

            if val_results["mean_iou"] > best_iou:

                best_iou = val_results["mean_iou"]

                save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_results["loss"],
                )

        # ====================================================
        # Epoch Checkpoint
        # ====================================================

        save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=train_loss,
        )

        # ====================================================
        # Epoch Summary
        # ====================================================

        print("\n" + "-" * 60)
        print(f"Epoch {epoch} Summary")
        print("-" * 60)

        print(f"Train Loss : {train_loss:.4f}")

        if val_results is not None:

            print(f"Validation Loss : {val_results['loss']:.4f}")
            print(f"Pixel Accuracy  : {val_results['pixel_accuracy']:.4f}")
            print(f"Dice Score      : {val_results['dice']:.4f}")
            print(f"Mean IoU        : {val_results['mean_iou']:.4f}")
            print(f"Precision       : {val_results['precision']:.4f}")
            print(f"Recall          : {val_results['recall']:.4f}")
            print(f"F1 Score        : {val_results['f1']:.4f}")
            print(f"Best IoU        : {best_iou:.4f}")

    # ========================================================
    # Final Checkpoint
    # ========================================================

    save_final_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        loss=history[-1],
    )

    writer.close()

    print("\n✓ TensorBoard writer closed.")
    print("✓ Training Complete.")

    return history