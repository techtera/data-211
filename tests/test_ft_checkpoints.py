"""
Tests for fine_tuning/checkpoints.py

Verifies save/load checkpoint preserves model and optimizer state.
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

import fine_tuning.checkpoints as ckpt_module
from fine_tuning.checkpoints import _save_checkpoint, load_checkpoint


# ============================================================
# Mock model
# ============================================================

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# Tests
# ============================================================

def test_save_and_load_checkpoint():
    tmp_dir = tempfile.mkdtemp()
    try:
        # Override checkpoint paths for test
        original_path = ckpt_module.CHECKPOINT_PATH
        ckpt_module.CHECKPOINT_PATH = __import__("pathlib").Path(tmp_dir)

        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        scaler = GradScaler(enabled=False)

        # Do a step to modify optimizer state
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save
        filepath = os.path.join(tmp_dir, "test_ckpt.pt")
        _save_checkpoint(model, optimizer, scheduler, scaler, 5, 0.123, filepath)

        assert os.path.exists(filepath), "Checkpoint file not created"

        # Load into fresh model
        model2 = SimpleModel()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10)
        scaler2 = GradScaler(enabled=False)

        epoch, loss_val = load_checkpoint(
            model2, optimizer2, scheduler2, scaler2, filepath
        )

        assert epoch == 5, f"Epoch mismatch: {epoch}"
        assert abs(loss_val - 0.123) < 1e-6, f"Loss mismatch: {loss_val}"

        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Model weights don't match after load"

        print("PASSED: test_save_and_load_checkpoint")

        ckpt_module.CHECKPOINT_PATH = original_path

    finally:
        shutil.rmtree(tmp_dir)


def test_checkpoint_contains_all_keys():
    tmp_dir = tempfile.mkdtemp()
    try:
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        scaler = GradScaler(enabled=False)

        filepath = os.path.join(tmp_dir, "test_ckpt.pt")
        _save_checkpoint(model, optimizer, scheduler, scaler, 3, 0.5, filepath)

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        required_keys = [
            "epoch", "loss",
            "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "scaler_state_dict",
        ]

        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"

        print("PASSED: test_checkpoint_contains_all_keys")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_save_and_load_checkpoint()
    test_checkpoint_contains_all_keys()
    print("\n=== ALL 2 TESTS PASSED ===")
