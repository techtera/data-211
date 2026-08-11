"""
Inference smoke test for VGGTEdgeMask.

Instantiates the model with random VGGT weights, runs a forward pass,
and verifies inference mode works correctly.

Usage:
    python inference_smoke_test.py
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "vggt")

import torch
from vggt.models.aggregator import Aggregator
from edge_mask.model import VGGTEdgeMask


def main():
    print("=" * 60)
    print("  Inference Smoke Test")
    print("=" * 60)

    # Instantiate
    print("\n  Instantiating model...")
    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {frozen:,}")

    # Forward pass
    B, S = 1, 2
    images = torch.rand(B, S, 3, 518, 518)
    print(f"\n  Input: {images.shape}")

    print("  Running forward pass...")
    with torch.no_grad():
        output = model(images)

    # Verify
    print(f"\n  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Value range:  [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Mean:         {output.mean().item():.4f}")

    # Assertions
    assert output.shape == torch.Size([B, S, 1, 518, 518]), f"Wrong shape: {output.shape}"
    assert output.min() >= 0.0, f"Values below 0: {output.min().item()}"
    assert output.max() <= 1.0, f"Values above 1: {output.max().item()}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
