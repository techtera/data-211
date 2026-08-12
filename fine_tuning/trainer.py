"""
Training utilities for Edge Mask fine-tuning.
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .config import (
    DEVICE,
    GRAD_CLIP_MAX_NORM,
    LOG_EVERY,
    PATIENCE,
)

from .losses import compute_total_loss
from .validate import validate

from .checkpoints import (
    save_latest_checkpoint,
    save_best_checkpoint,
    save_epoch_checkpoint,
)


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epoch,
):
    """
    Train the model for one epoch.

    Steps per batch:
        1. Forward pass in fp16 (autocast)
        2. Compute total loss (final + deep supervision)
        3. Backward with GradScaler
        4. Unscale, clip gradients, optimizer step
        5. Scheduler step (per iteration)

    Returns
    -------
    epoch_loss : float
    max_grad_norm : float
    """

    print("\n" + "=" * 60)
    print(f"Epoch {epoch}")
    print("=" * 60)

    model.train()

    # Encoder stays frozen in eval mode
    model.feature_extractor.aggregator.eval()

    running_loss = 0.0
    max_grad_norm = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):

        # --------------------------------------------------------
        # Move to device
        # --------------------------------------------------------

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # --------------------------------------------------------
        # Forward (mixed precision)
        # --------------------------------------------------------

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):

            logits, ds1_logits, ds2_logits = model(images)

            loss = compute_total_loss(
                logits, ds1_logits, ds2_logits,
                masks, criterion,
            )

        # --------------------------------------------------------
        # Backward
        # --------------------------------------------------------

        scaler.scale(loss).backward()

        # --------------------------------------------------------
        # Gradient Clipping
        # --------------------------------------------------------

        scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_MAX_NORM,
        )

        max_grad_norm = max(max_grad_norm, grad_norm.item())

        # --------------------------------------------------------
        # Optimizer Step
        # --------------------------------------------------------

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per iteration (not per epoch)
        scheduler.step()

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        running_loss += loss.item()

        if (
            (batch_idx + 1) % LOG_EVERY == 0
            or batch_idx == 0
        ):
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Batch [{batch_idx + 1:04d}/{len(dataloader):04d}] "
                f"Loss: {loss.item():.4f}  "
                f"LR: {current_lr:.2e}  "
                f"Grad: {grad_norm.item():.2f}"
            )

    # --------------------------------------------------------
    # Epoch Summary
    # --------------------------------------------------------

    epoch_loss = running_loss / len(dataloader)

    print(f"\nAverage Training Loss : {epoch_loss:.4f}")
    print(f"Max Gradient Norm     : {max_grad_norm:.2f}")

    return epoch_loss, max_grad_norm


# ============================================================
# Complete Training Loop
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
):
    """
    Complete training loop with:
        - Mixed precision (GradScaler)
        - Validation after each epoch
        - Checkpointing (latest, best, periodic)
        - Early stopping on validation loss
        - Collapse detection (edge_ratio)
    """

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting Fine-Tuning...\n")

    # --------------------------------------------------------
    # Epoch Loop
    # --------------------------------------------------------

    for epoch in range(1, num_epochs + 1):

        # ====================================================
        # Training
        # ====================================================

        train_loss, max_grad_norm = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
        )

        # ====================================================
        # Validation
        # ====================================================

        val_results = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
        )

        val_loss = val_results["loss"]
        edge_ratio = val_results["edge_ratio"]

        # ====================================================
        # Collapse Detection
        # ====================================================

        if epoch >= 10 and edge_ratio < 1e-4:
            print("[WARNING] Possible all-zero collapse! edge_ratio ~ 0")

        # ====================================================
        # Latest Checkpoint
        # ====================================================

        save_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            loss=train_loss,
        )

        # ====================================================
        # Best Checkpoint
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                loss=val_loss,
            )

            print(f"  New best val_loss : {val_loss:.4f}")

        else:
            patience_counter += 1

        # ====================================================
        # Epoch Checkpoint
        # ====================================================

        save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            loss=train_loss,
        )

        # ====================================================
        # Epoch Summary
        # ====================================================

        print("\n" + "-" * 60)
        print(f"Epoch {epoch} Summary")
        print("-" * 60)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss         : {train_loss:.4f}")
        print(f"Val Loss           : {val_loss:.4f}")
        print(f"Learning Rate      : {current_lr:.2e}")
        print(f"Max Grad Norm      : {max_grad_norm:.2f}")
        print(f"Edge Ratio         : {edge_ratio:.4f}")
        print(f"Best Val Loss      : {best_val_loss:.4f}")
        print(f"Patience           : {patience_counter}/{PATIENCE}")

        # ====================================================
        # Early Stopping
        # ====================================================

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {PATIENCE} epochs)"
            )
            break

    # ========================================================
    # Training Complete
    # ========================================================

    # Verify encoder stayed frozen
    encoder_has_grad = any(
        p.grad is not None
        for p in model.feature_extractor.aggregator.parameters()
    )

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print(f"Best Val Loss      : {best_val_loss:.4f}")
    print(f"Encoder Frozen     : {'YES' if not encoder_has_grad else 'NO - ERROR!'}")

    return best_val_loss
