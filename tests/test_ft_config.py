"""
Tests for fine_tuning/config.py

Verifies all constants are defined and have correct types/ranges.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fine_tuning.config import (
    PRETRAINED_MODEL,
    IMAGE_SIZE,
    DATASET_ROOT,
    VALIDATION_SPLIT,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP_MAX_NORM,
    WARMUP_FRACTION,
    LOG_EVERY,
    BCE_WEIGHT,
    DICE_WEIGHT,
    POS_WEIGHT_CLAMP,
    DS1_WEIGHT,
    DS2_WEIGHT,
    FINAL_WEIGHT,
    CHECKPOINT_DIR,
    SAVE_EVERY,
    SAVE_LATEST,
    SAVE_BEST,
    PATIENCE,
    DEVICE,
)


def test_pretrained_model():
    assert isinstance(PRETRAINED_MODEL, str)
    assert len(PRETRAINED_MODEL) > 0
    print("PASSED: test_pretrained_model")


def test_image_size():
    assert IMAGE_SIZE == 518
    print("PASSED: test_image_size")


def test_dataset_config():
    assert isinstance(DATASET_ROOT, str)
    assert 0.0 < VALIDATION_SPLIT < 1.0
    assert isinstance(RANDOM_SEED, int)
    print("PASSED: test_dataset_config")


def test_dataloader_config():
    assert BATCH_SIZE > 0
    assert NUM_WORKERS >= 0
    print("PASSED: test_dataloader_config")


def test_training_config():
    assert NUM_EPOCHS > 0
    assert 0 < LEARNING_RATE < 1.0
    assert WEIGHT_DECAY >= 0
    assert GRAD_CLIP_MAX_NORM > 0
    assert 0 < WARMUP_FRACTION < 1.0
    assert LOG_EVERY > 0
    print("PASSED: test_training_config")


def test_loss_config():
    assert BCE_WEIGHT > 0
    assert DICE_WEIGHT > 0
    assert BCE_WEIGHT + DICE_WEIGHT == 1.0
    assert len(POS_WEIGHT_CLAMP) == 2
    assert POS_WEIGHT_CLAMP[0] < POS_WEIGHT_CLAMP[1]
    assert DS1_WEIGHT > 0
    assert DS2_WEIGHT > 0
    assert FINAL_WEIGHT > 0
    # Final should dominate
    assert FINAL_WEIGHT > DS1_WEIGHT
    assert FINAL_WEIGHT > DS2_WEIGHT
    print("PASSED: test_loss_config")


def test_checkpoint_config():
    assert isinstance(CHECKPOINT_DIR, str)
    assert isinstance(SAVE_EVERY, int)
    assert isinstance(SAVE_LATEST, bool)
    assert isinstance(SAVE_BEST, bool)
    print("PASSED: test_checkpoint_config")


def test_early_stopping():
    assert PATIENCE > 0
    print("PASSED: test_early_stopping")


def test_device():
    assert isinstance(DEVICE, torch.device)
    print("PASSED: test_device")


if __name__ == "__main__":
    test_pretrained_model()
    test_image_size()
    test_dataset_config()
    test_dataloader_config()
    test_training_config()
    test_loss_config()
    test_checkpoint_config()
    test_early_stopping()
    test_device()
    print("\n=== ALL 9 TESTS PASSED ===")
