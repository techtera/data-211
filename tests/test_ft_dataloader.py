"""
Tests for fine_tuning/dataloader.py

Verifies build_dataloaders() creates correct train/val split.
"""

import sys
import os
import tempfile
import shutil

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fine_tuning.config as config
from fine_tuning.dataloader import build_dataloaders


# ============================================================
# Helper: create dummy data
# ============================================================

def create_dummy_data(tmp_dir, num_images=20):

    rgb_dir = os.path.join(tmp_dir, "rgb")
    mask_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    for i in range(num_images):
        name = f"img_{i:03d}.png"
        rgb_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr).save(os.path.join(rgb_dir, name))

        mask_arr = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        Image.fromarray(mask_arr, mode="L").save(
            os.path.join(mask_dir, f"img_{i:03d}_mask.png")
        )


# ============================================================
# Tests
# ============================================================

def test_build_dataloaders():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=20)

        # Override config for test
        original_root = config.DATASET_ROOT
        original_batch = config.BATCH_SIZE
        config.DATASET_ROOT = tmp_dir
        config.BATCH_SIZE = 2

        train_loader, val_loader = build_dataloaders()

        # 20 images, 10% val = 2 val, 18 train
        assert len(train_loader.dataset) == 18, f"Train size: {len(train_loader.dataset)}"
        assert len(val_loader.dataset) == 2, f"Val size: {len(val_loader.dataset)}"

        print("PASSED: test_build_dataloaders")

        # Restore
        config.DATASET_ROOT = original_root
        config.BATCH_SIZE = original_batch

    finally:
        shutil.rmtree(tmp_dir)


def test_batch_shapes():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=10)

        original_root = config.DATASET_ROOT
        original_batch = config.BATCH_SIZE
        config.DATASET_ROOT = tmp_dir
        config.BATCH_SIZE = 2

        train_loader, val_loader = build_dataloaders()

        batch_rgb, batch_mask = next(iter(train_loader))

        assert batch_rgb.shape[0] == 2, f"Batch size: {batch_rgb.shape[0]}"
        assert batch_rgb.shape[1] == 1, f"S dim: {batch_rgb.shape[1]}"
        assert batch_rgb.shape[2] == 3, f"Channels: {batch_rgb.shape[2]}"
        assert batch_rgb.shape[3] == 518
        assert batch_rgb.shape[4] == 518

        print("PASSED: test_batch_shapes")

        config.DATASET_ROOT = original_root
        config.BATCH_SIZE = original_batch

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_build_dataloaders()
    test_batch_shapes()
    print("\n=== ALL 2 TESTS PASSED ===")
