"""
Dataset for Edge Mask fine-tuning.

Expected structure:

    data/
    ├── rgb/
    │   ├── abc.png
    │   ├── def.png
    │   └── ...
    │
    └── masks/
        ├── abc_mask.png
        ├── def_mask.png
        └── ...
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .config import IMAGE_SIZE


# ============================================================
# Edge Mask Dataset
# ============================================================

class EdgeMaskDataset(Dataset):
    """
    Loads RGB images and their corresponding binary edge masks.

    - RGB images are resized to IMAGE_SIZE x IMAGE_SIZE
      using bilinear interpolation.

    - Masks are resized using nearest-neighbor
      to preserve binary edges.

    - Mask is binarized at threshold 0.5.

    - Returns tensors with S=1 dimension for VGGT compatibility:
        rgb:  [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        mask: [1, 1, IMAGE_SIZE, IMAGE_SIZE]
    """

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.mask_dir = self.data_dir / "masks"

        all_images = sorted([
            f for f in os.listdir(self.rgb_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        # Only keep images that have a matching mask
        self.image_names = []
        skipped = 0

        for f in all_images:
            stem = Path(f).stem
            suffix = Path(f).suffix
            mask_path = self.mask_dir / f"{stem}_mask{suffix}"
            if mask_path.exists():
                self.image_names.append(f)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} images with no matching mask")

        self.rgb_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img_name = self.image_names[idx]
        stem = Path(img_name).stem
        suffix = Path(img_name).suffix

        # --------------------------------------------------------
        # Load RGB and Mask
        # --------------------------------------------------------

        rgb_path = self.rgb_dir / img_name
        mask_path = self.mask_dir / f"{stem}_mask{suffix}"

        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # --------------------------------------------------------
        # Transform
        # --------------------------------------------------------

        rgb = self.rgb_transform(rgb)        # [3, 518, 518]
        mask = self.mask_transform(mask)      # [1, 518, 518]
        mask = (mask > 0.5).float()          # binarize

        # --------------------------------------------------------
        # Add S=1 Dimension
        # --------------------------------------------------------

        rgb = rgb.unsqueeze(0)               # [1, 3, 518, 518]
        mask = mask.unsqueeze(0)             # [1, 1, 518, 518]

        return rgb, mask
