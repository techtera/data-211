"""
Main script for Student Encoder + UNet++ Edge Mask fine-tuning.

Uses the distilled student encoder (255M params) instead of VGGT-1B (909M params).
"""

from fine_tuning.config import NUM_EPOCHS
from fine_tuning.dataloader import build_dataloaders
from fine_tuning.losses import build_loss
from fine_tuning.model_builder import build_model
from fine_tuning.optimizer import build_optimizer
from fine_tuning.scheduler import build_scheduler
from fine_tuning.trainer import train
from fine_tuning.evaluate import evaluate


def main():
    print("\n" + "=" * 60)
    print("Student Encoder + UNet++ Edge Mask Fine-Tuning")
    print("=" * 60)

    model = build_model()
    train_loader, val_loader = build_dataloaders()
    criterion = build_loss()
    optimizer = build_optimizer(model)
    
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps)

    best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
    )

    results = evaluate(model=model, dataloader=val_loader)

    print("\n" + "=" * 60)
    print("Fine-Tuning Summary")
    print("=" * 60)
    print(f"Total Epochs       : {NUM_EPOCHS}")
    print(f"Best Val Loss      : {best_val_loss:.6f}")
    print(f"Dice Score         : {results['dice']:.4f}")
    print(f"BF1 F1             : {results['bf1']['f1']:.4f}")
    print(f"ODS Best F1        : {results['ods']['best_f1']:.4f}")
    print("\nFine-tuning completed successfully.")


if __name__ == "__main__":
    main()
