"""
Lightweight integration test for the VGGT + SegFormer fine-tuning pipeline.

This script verifies:

1. Dataset
2. Model
3. Inference
4. Training (Loss + Backpropagation)
"""

import torch

from fine_tuning.config import (
    DATASET_ROOT,
    IMAGE_SIZE,
)

from fine_tuning.dataset import SegmentationDataset
from fine_tuning.model_builder import build_model
from fine_tuning.losses import build_loss
from fine_tuning.optimizer import build_optimizer


# ============================================================
# Dataset
# ============================================================

print("=" * 60)
print("Testing Dataset")
print("=" * 60)

dataset = SegmentationDataset(
    root_dir=DATASET_ROOT,
    image_size=IMAGE_SIZE,
)

print(f"Dataset Size : {len(dataset)}")

image, mask = dataset[0]

print(f"Image Shape : {image.shape}")
print(f"Mask Shape  : {mask.shape}")
print(f"Mask Labels : {torch.unique(mask)}")

# Add batch and sequence dimensions
image = image.unsqueeze(0)      # (1,3,H,W)
image = image.unsqueeze(1)      # (1,1,3,H,W)

mask = mask.unsqueeze(0)        # (1,H,W)

print(f"\nInput Image Shape : {image.shape}")
print(f"Input Mask Shape  : {mask.shape}")


# ============================================================
# Build Model
# ============================================================

print("\n" + "=" * 60)
print("Building Model")
print("=" * 60)

model = build_model()


# ============================================================
# Inference Test
# ============================================================

print("\n" + "=" * 60)
print("Inference Test")
print("=" * 60)

model.eval()

with torch.no_grad():

    predictions = model(image)

    logits = predictions["mask_logits"]

print(f"Output Shape : {logits.shape}")

print("✓ Inference Successful")


# ============================================================
# Training Test
# ============================================================

print("\n" + "=" * 60)
print("Training Test")
print("=" * 60)

model.train()

predictions = model(image)

logits = predictions["mask_logits"]

print(f"Output Shape      : {logits.shape}")
print(f"Requires Grad     : {logits.requires_grad}")

criterion = build_loss()

loss = criterion(
    logits,
    mask,
)

print(f"Loss : {loss.item():.6f}")

optimizer = build_optimizer(model)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("✓ Backpropagation Successful")


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("PIPELINE VERIFIED")
print("=" * 60)

print("✓ Dataset")
print("✓ Model")
print("✓ Inference")
print("✓ Loss")
print("✓ Optimizer")
print("✓ Backpropagation")

print("\n🚀 Everything is ready for fine-tuning!")