"""
Checkpoint utilities for SegFormer fine-tuning.
"""

from pathlib import Path
import torch

from .config import (
    CHECKPOINT_DIR,
    SAVE_EVERY,
    SAVE_LATEST,
    SAVE_BEST,
    SAVE_FINAL,
)

CHECKPOINT_PATH = Path(CHECKPOINT_DIR)
EPOCH_CHECKPOINT_PATH = CHECKPOINT_PATH / "epochs"

CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
EPOCH_CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


def _save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    filepath,
):
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, filepath)

    print(f"✓ Checkpoint Saved : {filepath}")


def save_latest_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
):
    if not SAVE_LATEST:
        return

    filepath = CHECKPOINT_PATH / "latest_model.pth"

    _save_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        filepath,
    )


def save_best_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
):
    if not SAVE_BEST:
        return

    filepath = CHECKPOINT_PATH / "best_model.pth"

    _save_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        filepath,
    )


def save_epoch_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
):
    if SAVE_EVERY <= 0:
        return

    if epoch % SAVE_EVERY != 0:
        return

    filepath = (
        EPOCH_CHECKPOINT_PATH
        / f"epoch_{epoch:03d}.pth"
    )

    _save_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        filepath,
    )


def save_final_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
):
    if not SAVE_FINAL:
        return

    filepath = CHECKPOINT_PATH / "final_model.pth"

    _save_checkpoint(
        model,
        optimizer,
        epoch,
        loss,
        filepath,
    )


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device="cpu",
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    print(
        f"✓ Loaded Checkpoint : {checkpoint_path}"
    )

    return (
        model,
        optimizer,
        epoch,
        loss,
    )
