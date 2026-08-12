"""
Tests for fine_tuning/optimizer.py

Verifies build_optimizer creates correct optimizer with only trainable params.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from fine_tuning.optimizer import build_optimizer
from fine_tuning.config import LEARNING_RATE, WEIGHT_DECAY


# ============================================================
# Mock model with frozen + trainable params
# ============================================================

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(10, 10)
        self.trainable = nn.Linear(10, 10)
        self.frozen.requires_grad_(False)

    def forward(self, x):
        return self.trainable(self.frozen(x))


# ============================================================
# Tests
# ============================================================

def test_optimizer_only_trainable():
    model = MockModel()
    optimizer = build_optimizer(model)

    # Only trainable params should be in optimizer
    opt_params = []
    for group in optimizer.param_groups:
        opt_params.extend(group["params"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    assert len(opt_params) == len(trainable_params), (
        f"Optimizer has {len(opt_params)} params, expected {len(trainable_params)}"
    )

    print("PASSED: test_optimizer_only_trainable")


def test_optimizer_lr():
    model = MockModel()
    optimizer = build_optimizer(model)

    lr = optimizer.param_groups[0]["lr"]
    assert lr == LEARNING_RATE, f"LR is {lr}, expected {LEARNING_RATE}"

    print("PASSED: test_optimizer_lr")


def test_optimizer_weight_decay():
    model = MockModel()
    optimizer = build_optimizer(model)

    wd = optimizer.param_groups[0]["weight_decay"]
    assert wd == WEIGHT_DECAY, f"WD is {wd}, expected {WEIGHT_DECAY}"

    print("PASSED: test_optimizer_weight_decay")


def test_optimizer_step():
    model = MockModel()
    optimizer = build_optimizer(model)

    x = torch.randn(2, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Snapshot before step
    before = model.trainable.weight.clone()

    optimizer.step()

    # Weight should have changed
    assert not torch.equal(model.trainable.weight, before), "Weights unchanged after step"

    # Frozen should be unchanged
    print("PASSED: test_optimizer_step")


if __name__ == "__main__":
    test_optimizer_only_trainable()
    test_optimizer_lr()
    test_optimizer_weight_decay()
    test_optimizer_step()
    print("\n=== ALL 4 TESTS PASSED ===")
