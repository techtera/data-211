"""
DataLoader utilities for Edge Mask fine-tuning.
"""

import torch

from torch.utils.data import (
    DataLoader,
    random_split,
)

from . import config
from .dataset import EdgeMaskDataset


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

    dataset = EdgeMaskDataset(
        data_dir=config.DATASET_ROOT,
    )

    dataset_size = len(dataset)

    val_size = max(1, int(
        dataset_size * config.VALIDATION_SPLIT
    ))

    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(
        config.RANDOM_SEED
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 60)
    print("Dataset Split")
    print("=" * 60)

    print(f"Total Samples      : {dataset_size}")
    print(f"Training Samples   : {train_size}")
    print(f"Validation Samples : {val_size}")

    print("\nTraining DataLoader")
    print(f"Batch Size         : {config.BATCH_SIZE}")
    print(f"Batches            : {len(train_loader)}")

    print("\nValidation DataLoader")
    print(f"Batch Size         : {config.BATCH_SIZE}")
    print(f"Batches            : {len(val_loader)}")

    return train_loader, val_loader
