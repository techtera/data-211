"""
Tests for fine_tuning/evaluate.py

Verifies all evaluation metrics: Dice, BF1, ODS, Confusion Matrix.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fine_tuning.evaluate import (
    confusion_matrix,
    dice_score,
    boundary_f1,
    optimal_dataset_scale,
    evaluate,
)


# ============================================================
# Confusion Matrix Tests
# ============================================================

def test_confusion_matrix_perfect():
    pred = torch.ones(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)

    cm = confusion_matrix(pred, target)

    assert cm["tp"] == 16
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tn"] == 0

    print("PASSED: test_confusion_matrix_perfect")


def test_confusion_matrix_all_wrong():
    pred = torch.ones(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)

    cm = confusion_matrix(pred, target)

    assert cm["tp"] == 0
    assert cm["fp"] == 16
    assert cm["fn"] == 0
    assert cm["tn"] == 0

    print("PASSED: test_confusion_matrix_all_wrong")


def test_confusion_matrix_mixed():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)

    # Top-left 2x2 = edge in both
    pred[0, 0, :2, :2] = 1.0
    target[0, 0, :2, :2] = 1.0

    # Bottom-right 2x2 = pred says edge, gt says no
    pred[0, 0, 2:, 2:] = 1.0

    cm = confusion_matrix(pred, target)

    assert cm["tp"] == 4   # top-left
    assert cm["fp"] == 4   # bottom-right (pred=1, gt=0)
    assert cm["fn"] == 0   # no gt edge missed
    assert cm["tn"] == 8   # rest

    print("PASSED: test_confusion_matrix_mixed")


# ============================================================
# Dice Score Tests
# ============================================================

def test_dice_perfect():
    pred = torch.ones(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)

    d = dice_score(pred, target)

    assert abs(d - 1.0) < 1e-4, f"Expected ~1.0, got {d}"

    print("PASSED: test_dice_perfect")


def test_dice_no_overlap():
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :] = 1.0

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4:, :] = 1.0

    d = dice_score(pred, target)

    assert d < 0.01, f"Expected ~0, got {d}"

    print("PASSED: test_dice_no_overlap")


def test_dice_half_overlap():
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :] = 1.0  # 8 pixels

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1:3, :] = 1.0  # 8 pixels

    # Overlap = row 1, all 4 cols = 4 pixels
    # Dice = 2*4 / (8 + 8) = 0.5
    d = dice_score(pred, target)

    assert abs(d - 0.5) < 0.01, f"Expected ~0.5, got {d}"

    print("PASSED: test_dice_half_overlap")


# ============================================================
# BF1 Tests
# ============================================================

def test_bf1_perfect():
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)

    pred[0, 0, 8, :] = 1.0
    target[0, 0, 8, :] = 1.0

    bf1 = boundary_f1(pred, target, tolerance=2)

    assert bf1["precision"] > 0.99
    assert bf1["recall"] > 0.99
    assert bf1["f1"] > 0.99

    print("PASSED: test_bf1_perfect")


def test_bf1_within_tolerance():
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)

    # Pred edge at row 8, GT at row 9 (1 pixel apart, within tolerance=2)
    pred[0, 0, 8, :] = 1.0
    target[0, 0, 9, :] = 1.0

    bf1 = boundary_f1(pred, target, tolerance=2)

    assert bf1["precision"] > 0.99
    assert bf1["recall"] > 0.99
    assert bf1["f1"] > 0.99

    print("PASSED: test_bf1_within_tolerance")


def test_bf1_outside_tolerance():
    pred = torch.zeros(1, 1, 32, 32)
    target = torch.zeros(1, 1, 32, 32)

    # Pred edge at row 5, GT at row 20 (15 pixels apart, outside tolerance=2)
    pred[0, 0, 5, :] = 1.0
    target[0, 0, 20, :] = 1.0

    bf1 = boundary_f1(pred, target, tolerance=2)

    assert bf1["f1"] < 0.1, f"Expected low F1, got {bf1['f1']}"

    print("PASSED: test_bf1_outside_tolerance")


# ============================================================
# ODS Tests
# ============================================================

def test_ods_returns_best_threshold():
    # Probabilities: top half = 0.9, bottom half = 0.1
    pred_probs = torch.zeros(1, 1, 8, 8)
    pred_probs[0, 0, :4, :] = 0.9
    pred_probs[0, 0, 4:, :] = 0.1

    # GT: top half = edge
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :4, :] = 1.0

    ods = optimal_dataset_scale(pred_probs, target)

    # Best threshold should be around 0.5 (separates 0.9 from 0.1)
    assert 0.1 <= ods["best_threshold"] <= 0.9
    assert ods["best_f1"] > 0.9

    print(f"PASSED: test_ods_returns_best_threshold (t={ods['best_threshold']:.2f}, f1={ods['best_f1']:.4f})")


def test_ods_all_scores_length():
    pred_probs = torch.rand(1, 1, 4, 4)
    target = (torch.rand(1, 1, 4, 4) > 0.5).float()

    thresholds = [0.2, 0.4, 0.6, 0.8]
    ods = optimal_dataset_scale(pred_probs, target, thresholds=thresholds)

    assert len(ods["all_scores"]) == 4

    print("PASSED: test_ods_all_scores_length")


# ============================================================
# Full Evaluate Test
# ============================================================

def test_evaluate_full():
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Simple model that returns sigmoid directly
    class FakeModel(nn.Module):
        def forward(self, x):
            B, S = x.shape[:2]
            # Return constant 0.8 probability (high edge prediction)
            return torch.full((B, S, 1, x.shape[3], x.shape[4]), 0.8)

    model = FakeModel()

    # All-edge targets
    images = torch.rand(4, 1, 3, 32, 32)
    masks = torch.ones(4, 1, 1, 32, 32)

    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=2)

    results = evaluate(model, loader, threshold=0.5)

    assert "dice" in results
    assert "bf1" in results
    assert "ods" in results
    assert "confusion" in results

    # All predictions above 0.5, all targets = 1 -> perfect
    assert results["dice"] > 0.99
    assert results["bf1"]["f1"] > 0.99
    assert results["confusion"]["tp"] == 4 * 32 * 32
    assert results["confusion"]["fp"] == 0

    print("PASSED: test_evaluate_full")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    test_confusion_matrix_perfect()
    test_confusion_matrix_all_wrong()
    test_confusion_matrix_mixed()
    test_dice_perfect()
    test_dice_no_overlap()
    test_dice_half_overlap()
    test_bf1_perfect()
    test_bf1_within_tolerance()
    test_bf1_outside_tolerance()
    test_ods_returns_best_threshold()
    test_ods_all_scores_length()
    test_evaluate_full()
    print("\n=== ALL 12 TESTS PASSED ===")
