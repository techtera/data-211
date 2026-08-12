"""
Main script for VGGT + UNet++ Edge Mask fine-tuning.
"""

from fine_tuning.config import NUM_EPOCHS
from fine_tuning.dataloader import build_dataloaders
from fine_tuning.losses import build_loss
from fine_tuning.model_builder import build_model
from fine_tuning.optimizer import build_optimizer
from fine_tuning.scheduler import build_scheduler
from fine_tuning.trainer import train


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("VGGT + UNet++ Edge Mask Fine-Tuning")
    print("=" * 60)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader = build_dataloaders()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = build_loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(model)

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    total_steps = NUM_EPOCHS * len(train_loader)

    scheduler, warmup_steps = build_scheduler(
        optimizer,
        total_steps,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Fine-Tuning Summary")
    print("=" * 60)

    print(f"Total Epochs       : {NUM_EPOCHS}")
    print(f"Best Val Loss      : {best_val_loss:.6f}")

    print("\nFine-tuning completed successfully.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
