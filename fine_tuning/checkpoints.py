"""
Checkpoint utilities for Edge Mask fine-tuning.
"""

from pathlib import Path
import torch

from .config import (
    CHECKPOINT_DIR,
    SAVE_EVERY,
    SAVE_LATEST,
    SAVE_BEST,
)


CHECKPOINT_PATH = Path(CHECKPOINT_DIR)
EPOCH_CHECKPOINT_PATH = CHECKPOINT_PATH / "epochs"

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
EPOCH_CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# Internal Save
# ============================================================

def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    loss,
    filepath,
):
    """
    Save full training state for resuming.
    """

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    torch.save(checkpoint, filepath)

    print(f"  Checkpoint Saved : {filepath}")


# ============================================================
# Save Latest
# ============================================================

def save_latest_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    loss,
):
    if not SAVE_LATEST:
        return

    filepath = CHECKPOINT_PATH / "latest_model.pt"

    _save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, loss, filepath,
    )


# ============================================================
# Save Best
# ============================================================

def save_best_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    loss,
):
    if not SAVE_BEST:
        return

    filepath = CHECKPOINT_PATH / "best_model.pt"

    _save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, loss, filepath,
    )


# ============================================================
# Save Epoch
# ============================================================

def save_epoch_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    loss,
):
    if SAVE_EVERY <= 0:
        return

    if epoch % SAVE_EVERY != 0:
        return

    filepath = (
        EPOCH_CHECKPOINT_PATH
        / f"epoch_{epoch:03d}.pt"
    )

    _save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, loss, filepath,
    )


# ============================================================
# Load Checkpoint
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device="cpu",
):
    """
    Load a checkpoint and restore full training state.

    Returns
    -------
    epoch : int
    loss : float
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(f"Loaded Checkpoint  : {checkpoint_path}")
    print(f"Resumed at Epoch   : {epoch}")

    return epoch, loss
