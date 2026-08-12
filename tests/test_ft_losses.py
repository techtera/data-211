"""
Tests for fine_tuning/losses.py

Verifies EdgeLoss, compute_total_loss, and build_loss.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fine_tuning.losses import EdgeLoss, compute_total_loss, build_loss


def test_edge_loss_forward():
    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()

    loss = loss_fn(logits, target)

    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    loss.backward()
    assert logits.grad is not None, "No gradient on logits"

    print("PASSED: test_edge_loss_forward")


def test_edge_loss_all_zero_target():
    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64)
    target = torch.zeros(2, 1, 64, 64)

    loss = loss_fn(logits, target)

    assert not torch.isnan(loss), "NaN with all-zero target"
    assert loss.item() >= 0, "Negative loss"

    print("PASSED: test_edge_loss_all_zero_target")


def test_edge_loss_all_one_target():
    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)

    loss = loss_fn(logits, target)

    assert not torch.isnan(loss), "NaN with all-one target"
    assert loss.item() >= 0, "Negative loss"

    print("PASSED: test_edge_loss_all_one_target")


def test_compute_total_loss_weights():
    loss_fn = EdgeLoss()

    final = torch.randn(2, 1, 64, 64)
    ds1 = torch.randn(2, 1, 64, 64)
    ds2 = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()

    total = compute_total_loss(final, ds1, ds2, target, loss_fn)

    assert total.item() > 0, "Total loss should be positive"
    assert not torch.isnan(total), "Total loss is NaN"

    # Final should dominate (weight=1.0 vs 0.1 and 0.2)
    loss_final = loss_fn(final, target)
    assert total.item() >= loss_final.item() * 0.8, "Final should dominate total"

    print("PASSED: test_compute_total_loss_weights")


def test_build_loss():
    criterion = build_loss()
    assert isinstance(criterion, EdgeLoss)
    print("PASSED: test_build_loss")


if __name__ == "__main__":
    test_edge_loss_forward()
    test_edge_loss_all_zero_target()
    test_edge_loss_all_one_target()
    test_compute_total_loss_weights()
    test_build_loss()
    print("\n=== ALL 5 TESTS PASSED ===")
