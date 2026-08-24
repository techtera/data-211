"""
Image dataset for distillation training.

Loads images, applies preprocessing, returns batches of [B, S, C, H, W].
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import List, Tuple
import glob


class ImageDataset(Dataset):
    """
    Simple image dataset for distillation.

    Expected structure:
        image_dir/
            *.jpg, *.png, *.jpeg

    Returns sequences of S frames (with temporal augmentation if needed).
    """

    def __init__(
        self,
        image_dir: str,
        num_frames: int = 8,
        image_size: int = 518,
        extensions: tuple = ('.jpg', '.jpeg', '.png')
    ):
        """
        Args:
            image_dir: Directory containing images
            num_frames: Number of frames per sequence (S)
            image_size: Input size for VGGT (518×518)
            extensions: Valid image extensions
        """
        self.image_dir = image_dir
        self.num_frames = num_frames
        self.image_size = image_size

        # Find all images
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.image_paths.sort()  # Deterministic order

        # Preprocessing (VGGT standard)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        """Number of samples (each image becomes one sequence)."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess image sequence.

        Returns:
            images: [S, C, H, W] where S=num_frames
        """
        # Load single image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)  # [C, H, W]

        # Replicate to create sequence [S, C, H, W]
        # TODO: For multi-view datasets, load S different views instead
        images = img_tensor.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        return images

    def get_info(self) -> dict:
        """Get dataset statistics."""
        return {
            'num_images': len(self.image_paths),
            'num_frames': self.num_frames,
            'image_size': self.image_size,
            'image_dir': self.image_dir,
        }


def create_dataloader(
    image_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = 8,
    image_size: int = 518,
    shuffle: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create dataloader for training.

    Args:
        image_dir: Path to image directory
        batch_size: Batch size per GPU (effective batch = batch_size × num_gpus)
        num_workers: DataLoader workers
        num_frames: Frames per sequence
        image_size: Image size
        shuffle: Shuffle dataset
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader yielding [B, S, C, H, W] batches
    """
    dataset = ImageDataset(
        image_dir=image_dir,
        num_frames=num_frames,
        image_size=image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True  # Faster GPU transfer
    )

    return dataloader


def verify_dataloader(dataloader: DataLoader, device: str = 'cuda'):
    """Quick test to verify dataloader works."""
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Load one batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")

    # Try moving to device
    batch = batch.to(device)
    print(f"✓ Batch moved to {device}")

    return batch
