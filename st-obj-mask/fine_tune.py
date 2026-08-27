"""Main script for Student Encoder + Object Mask fine-tuning."""

from fine_tuning.config import NUM_EPOCHS
from fine_tuning.dataloader import build_dataloaders
from fine_tuning.losses import build_loss
from fine_tuning.model_builder import build_model
from fine_tuning.optimizer import build_optimizer
from fine_tuning.trainer import train


def main():
    print("\n" + "=" * 60)
    print("Student Encoder + Object Mask Fine-Tuning")
    print("=" * 60)

    model = build_model()
    train_loader, val_loader = build_dataloaders()
    criterion = build_loss()
    optimizer = build_optimizer(model)

    history = train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        writer=None,
        num_epochs=NUM_EPOCHS,
        val_loader=val_loader,
    )

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total Epochs       : {NUM_EPOCHS}")
    print(f"Final Training Loss: {history[-1]:.6f}")
    print("\n✓ Fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
