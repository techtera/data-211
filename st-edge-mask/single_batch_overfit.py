#!/usr/bin/env python3
"""
Single batch overfit test for Edge Mask decoder.

Verifies the model can learn by overfitting on a single batch.
Should take ~2-3 minutes.

Usage:
    python sanity_check.py
"""

import torch
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path

sys.path.insert(0, "../kd-encoder")

from student import StudentAggregator
from edge_mask.model import StudentEdgeMask
from fine_tuning.dataset import EdgeMaskDataset
from fine_tuning.losses import build_loss
from fine_tuning.config import STUDENT_CHECKPOINT, DATASET_ROOT


def main():
    print("="*60)
    print("EDGE MASK - SINGLE BATCH OVERFIT TEST")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load student encoder
    print("\n[1/4] Loading student encoder...")
    checkpoint = torch.load(STUDENT_CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict', checkpoint))

    student = StudentAggregator()
    student.load_state_dict(state_dict)
    student.eval()
    student.requires_grad_(False)
    print(f"  ✓ Student loaded: {sum(p.numel() for p in student.parameters()):,} params")

    # Build model
    print("\n[2/4] Building model...")
    model = StudentEdgeMask(student).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model built: {trainable:,} trainable params")

    # Load single batch
    print("\n[3/4] Loading single batch...")
    dataset = EdgeMaskDataset(Path(DATASET_ROOT))
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, shuffle=False)
    images, masks = next(iter(loader))
    images = images.to(device)
    masks = masks.to(device)
    print(f"  ✓ Batch loaded: images {list(images.shape)}, masks {list(masks.shape)}")

    # Setup training
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Overfit test
    print("\n[4/4] Overfitting on single batch (100 steps)...")
    print("-"*60)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(100):
        logits, ds1_logits, ds2_logits = model(images)
        loss = criterion(logits, ds1_logits, ds2_logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        if step % 20 == 0 or step == 99:
            print(f"  Step {step:3d}: Loss = {loss.item():.6f}")

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Initial Loss : {initial_loss:.6f}")
    print(f"Final Loss   : {final_loss:.6f}")
    print(f"Reduction    : {(1 - final_loss/initial_loss)*100:.1f}%")

    # Pass/Fail
    if final_loss < 0.2:
        print("\n✅ PASS - Model can learn!")
        print("  Model successfully overfits on single batch.")
        print("  Ready for full training.")
        return 0
    elif final_loss < initial_loss * 0.5:
        print("\n⚠️  PARTIAL PASS - Model is learning but slowly.")
        print("  Loss decreased but not enough for confidence.")
        print("  Consider checking learning rate or model architecture.")
        return 1
    else:
        print("\n❌ FAIL - Model cannot learn!")
        print("  Loss did not decrease significantly.")
        print("  Check model, loss function, or data pipeline.")
        return 2


if __name__ == "__main__":
    exit(main())
