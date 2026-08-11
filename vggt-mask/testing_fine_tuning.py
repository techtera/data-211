"""
Standalone validation pipeline test.
"""
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from fine_tuning.model_builder import build_model
from fine_tuning.dataloader import build_dataloaders
from fine_tuning.losses import build_loss
from fine_tuning.validate import validate


def main():

    print("=" * 60)
    print("Validation Pipeline Test")
    print("=" * 60)

    model = build_model()

 

    _, val_loader = build_dataloaders()

    small_dataset = Subset(
        val_loader.dataset,
        range(8),      # only 8 samples
    )

    val_loader = DataLoader(
        small_dataset,
        batch_size=2,
        shuffle=False,
    )

    criterion = build_loss()

    results = validate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
    )

    print("\nValidation Results")

    for key, value in results.items():

        if key == "confusion_matrix":

            print(f"\n{key}")
            print(value)

        else:

            print(f"{key:20s}: {value}")


if __name__ == "__main__":

    main()