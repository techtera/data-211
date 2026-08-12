"""
Tests for fine_tuning/validate.py

Verifies validation loop returns correct structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import fine_tuning.config as config
from fine_tuning.losses import EdgeLoss
from fine_tuning.validate import validate


# ============================================================
# Mock model that mimics VGGTEdgeMask in train mode
# ============================================================

class MockEdgeMaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        # Mock aggregator with parameters
        self.feature_extractor = type("FE", (), {
            "aggregator": nn.Linear(1, 1)
        })()

    def forward(self, images):
        # images: [B, 1, 3, 518, 518]
        B = images.shape[0]
        # Return 3 tensors like VGGTEdgeMask in train mode
        logits = torch.randn(B, 1, 1, 518, 518, device=images.device)
        ds1 = torch.randn(B, 1, 1, 518, 518, device=images.device)
        ds2 = torch.randn(B, 1, 1, 518, 518, device=images.device)
        return logits, ds1, ds2


# ============================================================
# Tests
# ============================================================

def test_validate_returns_dict():
    # Override device to cpu for testing
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()
    model.to(config.DEVICE)

    # Create dummy dataloader
    images = torch.randn(4, 1, 3, 518, 518)
    masks = (torch.rand(4, 1, 1, 518, 518) > 0.9).float()
    dataset = TensorDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=2)

    criterion = EdgeLoss()

    results = validate(model, dataloader, criterion)

    assert "loss" in results, "Missing 'loss' key"
    assert "edge_ratio" in results, "Missing 'edge_ratio' key"
    assert results["loss"] > 0, f"Loss should be positive: {results['loss']}"
    assert 0 <= results["edge_ratio"] <= 1, f"Edge ratio out of range: {results['edge_ratio']}"

    print("PASSED: test_validate_returns_dict")

    config.DEVICE = original_device


def test_validate_no_gradients():
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()

    images = torch.randn(2, 1, 3, 518, 518)
    masks = (torch.rand(2, 1, 1, 518, 518) > 0.9).float()
    dataset = TensorDataset(images, masks)
    dataloader = DataLoader(dataset, batch_size=2)

    criterion = EdgeLoss()

    validate(model, dataloader, criterion)

    # No gradients should accumulate on model params
    for p in model.parameters():
        assert p.grad is None, "Gradients accumulated during validation"

    print("PASSED: test_validate_no_gradients")

    config.DEVICE = original_device


if __name__ == "__main__":
    test_validate_returns_dict()
    test_validate_no_gradients()
    print("\n=== ALL 2 TESTS PASSED ===")
