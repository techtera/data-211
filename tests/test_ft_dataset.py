"""
Tests for fine_tuning/dataset.py

Verifies EdgeMaskDataset loads and transforms correctly.
"""

import sys
import os
import tempfile
import shutil

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fine_tuning.dataset import EdgeMaskDataset


# ============================================================
# Helper: create dummy data
# ============================================================

def create_dummy_data(tmp_dir, num_images=3, sizes=None):

    rgb_dir = os.path.join(tmp_dir, "rgb")
    mask_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    if sizes is None:
        sizes = [(256, 256)] * num_images

    for i, (h, w) in enumerate(sizes):

        name = f"img_{i:03d}.png"

        rgb_arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr).save(os.path.join(rgb_dir, name))

        mask_arr = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
        stem = f"img_{i:03d}"
        Image.fromarray(mask_arr, mode="L").save(
            os.path.join(mask_dir, f"{stem}_mask.png")
        )


# ============================================================
# Tests
# ============================================================

def test_dataset_length():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=5)
        ds = EdgeMaskDataset(tmp_dir)
        assert len(ds) == 5, f"Expected 5, got {len(ds)}"
        print("PASSED: test_dataset_length")
    finally:
        shutil.rmtree(tmp_dir)


def test_output_shapes():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=2)
        ds = EdgeMaskDataset(tmp_dir)
        rgb, mask = ds[0]
        assert rgb.shape == (1, 3, 518, 518), f"RGB shape {rgb.shape}"
        assert mask.shape == (1, 1, 518, 518), f"Mask shape {mask.shape}"
        print("PASSED: test_output_shapes")
    finally:
        shutil.rmtree(tmp_dir)


def test_mask_binary():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=2)
        ds = EdgeMaskDataset(tmp_dir)
        _, mask = ds[0]
        unique_vals = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals), f"Non-binary: {unique_vals}"
        print("PASSED: test_mask_binary")
    finally:
        shutil.rmtree(tmp_dir)


def test_rgb_range():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=2)
        ds = EdgeMaskDataset(tmp_dir)
        rgb, _ = ds[0]
        assert rgb.min() >= 0.0, f"RGB min {rgb.min()}"
        assert rgb.max() <= 1.0, f"RGB max {rgb.max()}"
        print("PASSED: test_rgb_range")
    finally:
        shutil.rmtree(tmp_dir)


def test_different_input_sizes():
    tmp_dir = tempfile.mkdtemp()
    try:
        sizes = [(128, 200), (400, 600), (1024, 768)]
        create_dummy_data(tmp_dir, num_images=3, sizes=sizes)
        ds = EdgeMaskDataset(tmp_dir)
        for i in range(3):
            rgb, mask = ds[i]
            assert rgb.shape == (1, 3, 518, 518), f"Image {i} RGB {rgb.shape}"
            assert mask.shape == (1, 1, 518, 518), f"Image {i} mask {mask.shape}"
        print("PASSED: test_different_input_sizes")
    finally:
        shutil.rmtree(tmp_dir)


def test_dataloader_batch():
    tmp_dir = tempfile.mkdtemp()
    try:
        create_dummy_data(tmp_dir, num_images=4)
        ds = EdgeMaskDataset(tmp_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        batch_rgb, batch_mask = next(iter(loader))
        assert batch_rgb.shape == (2, 1, 3, 518, 518), f"Batch RGB {batch_rgb.shape}"
        assert batch_mask.shape == (2, 1, 1, 518, 518), f"Batch mask {batch_mask.shape}"
        print("PASSED: test_dataloader_batch")
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    test_dataset_length()
    test_output_shapes()
    test_mask_binary()
    test_rgb_range()
    test_different_input_sizes()
    test_dataloader_batch()
    print("\n=== ALL 6 TESTS PASSED ===")
