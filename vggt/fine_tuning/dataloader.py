"""
DataLoader utilities for SegFormer fine-tuning.
"""

import torch

from torch.utils.data import (
    DataLoader,
    random_split,
)

from .config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VALIDATION_SPLIT,
    RANDOM_SEED,
)

from .dataset import SegmentationDataset


# ============================================================
# Train / Validation Split
# ============================================================

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# ============================================================
# Build DataLoaders
# ============================================================

def build_dataloaders():
    """
    Build train and validation dataloaders.

    Returns
    -------
    train_loader
    val_loader
    """

    dataset = SegmentationDataset(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
    )

    dataset_size = len(dataset)

    val_size = int(
        dataset_size * VALIDATION_SPLIT
    )

    train_size = (
        dataset_size - val_size
    )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("=" * 60)
    print("Dataset Split")
    print("=" * 60)

    print(f"Total Samples      : {dataset_size}")
    print(f"Training Samples   : {train_size}")
    print(f"Validation Samples : {val_size}")

    print("\nTraining DataLoader")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Batches            : {len(train_loader)}")

    print("\nValidation DataLoader")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Batches            : {len(val_loader)}")

    return train_loader, val_loader