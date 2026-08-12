"""
Tests for fine_tuning/scheduler.py

Verifies warmup + cosine decay produces correct LR curve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fine_tuning.scheduler import build_scheduler


def test_warmup_ramp():
    """LR should ramp from ~0 to base_lr during warmup."""

    params = [torch.nn.Parameter(torch.randn(10))]
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    scheduler, warmup_steps = build_scheduler(optimizer, total_steps=100)

    # Step 0: lr should be near 0
    lr_start = optimizer.param_groups[0]["lr"]
    assert lr_start < 1e-5, f"LR at step 0 should be ~0, got {lr_start}"

    # Step through warmup
    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()

    # After warmup: lr should be at base (3e-4)
    lr_peak = optimizer.param_groups[0]["lr"]
    assert abs(lr_peak - 3e-4) < 1e-5, f"LR at warmup end should be 3e-4, got {lr_peak}"

    print("PASSED: test_warmup_ramp")


def test_cosine_decay():
    """LR should decay after warmup and approach 0 at end."""

    params = [torch.nn.Parameter(torch.randn(10))]
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    total_steps = 100
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps)

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # After warmup, LR should decrease
    assert lrs[50] < lrs[warmup_steps], (
        f"LR should decrease: step{warmup_steps}={lrs[warmup_steps]}, step50={lrs[50]}"
    )

    # At end, LR should be near 0
    assert lrs[-1] < 1e-5, f"LR at end should be ~0, got {lrs[-1]}"

    print("PASSED: test_cosine_decay")


def test_scheduler_never_negative():
    """LR should never go negative."""

    params = [torch.nn.Parameter(torch.randn(10))]
    optimizer = torch.optim.AdamW(params, lr=3e-4)

    total_steps = 200
    scheduler, _ = build_scheduler(optimizer, total_steps)

    for step in range(total_steps):
        lr = optimizer.param_groups[0]["lr"]
        assert lr >= 0, f"Negative LR at step {step}: {lr}"
        optimizer.step()
        scheduler.step()

    print("PASSED: test_scheduler_never_negative")


if __name__ == "__main__":
    test_warmup_ramp()
    test_cosine_decay()
    test_scheduler_never_negative()
    print("\n=== ALL 3 TESTS PASSED ===")
