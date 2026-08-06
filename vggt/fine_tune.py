"""
Main script for SegFormer fine-tuning.
"""

from fine_tuning.config import NUM_EPOCHS

from fine_tuning.dataloader import build_dataloader
from fine_tuning.logger import build_writer
from fine_tuning.losses import build_loss
from fine_tuning.model_builder import build_model
from fine_tuning.optimizer import build_optimizer
from fine_tuning.trainer import train


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SegFormer Fine-Tuning")
    print("=" * 60)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader = build_dataloader()

    # Validation loader will be added later
    val_loader = None

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = build_loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(model)

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = build_writer()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        num_epochs=NUM_EPOCHS,
        val_loader=val_loader,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(f"Total Epochs       : {NUM_EPOCHS}")
    print(f"Final Training Loss: {history[-1]:.6f}")

    print("\n✓ Fine-tuning completed successfully.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()