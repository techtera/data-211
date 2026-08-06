"""
Training utilities for SegFormer fine-tuning.
"""

import torch.nn.functional as F

from .config import DEVICE, LOG_EVERY

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
        # Move batch to device
        # ----------------------------------------------------

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # VGGT expects (B, S, C, H, W)
        images = images.unsqueeze(1)

        # ----------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------

        optimizer.zero_grad()

        predictions = model(images)

        logits = predictions["mask_logits"]

        # ----------------------------------------------------
        # Resize predictions to match ground-truth mask
        # ----------------------------------------------------

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
        # Backward Pass
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # ----------------------------------------------------
        # Batch Logging
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

    best_loss = float("inf")

    print("\nStarting Fine-Tuning...\n")

    for epoch in range(1, num_epochs + 1):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        history.append(epoch_loss)

        # ----------------------------------------------------
        # Validation Placeholder
        # ----------------------------------------------------

        if val_loader is not None:
            # Validation loop will be implemented later.
            pass

        # ----------------------------------------------------
        # TensorBoard Logging
        # ----------------------------------------------------

        writer.add_scalar(
            "Loss/Train",
            epoch_loss,
            epoch,
        )

        writer.add_scalar(
            "Learning Rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        # ----------------------------------------------------
        # Latest Checkpoint
        # ----------------------------------------------------

        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_loss,
        )

        # ----------------------------------------------------
        # Best Checkpoint
        # ----------------------------------------------------

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss,
            )

        # ----------------------------------------------------
        # Save Every N Epochs
        # ----------------------------------------------------

        save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=epoch_loss,
        )

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