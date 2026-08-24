#!/usr/bin/env python3
"""
Load real VGGT teacher encoder from checkpoint.
"""

import sys
import os
import torch

# Add vggt-unified to path
sys.path.insert(0, os.path.abspath('../../vggt-unified'))

try:
    from model import VGGTUnified
except ImportError as e:
    print(f"Error: Could not import VGGTUnified from ../../vggt-unified")
    print(f"Make sure vggt-unified directory exists with model.py")
    raise


def load_real_teacher(checkpoint_path='../../vggt-unified/checkpoints/vggt_unified_fp16.pt', device='cuda'):
    """
    Load real VGGT teacher from unified checkpoint.

    Configures teacher to cache only 4 layers that correspond to student's cached layers:
    - Student layer 3  → Teacher layer 4
    - Student layer 8  → Teacher layer 11
    - Student layer 13 → Teacher layer 17
    - Student layer 17 → Teacher layer 23

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Teacher aggregator (encoder only) with 4 cached layers
    """
    print(f"\nLoading real teacher from: {checkpoint_path}")

    # Load unified model
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(checkpoint_path)

    # Extract just the encoder (aggregator)
    teacher = model.aggregator
    teacher = teacher.to(device).eval()

    # Configure cached layers to match student's proportional mapping
    # Teacher has 24 layers (0-23), student has 18 layers (0-17)
    # Student caches: [3, 8, 13, 17] → Teacher should cache: [4, 11, 17, 23]
    teacher.cached_layer_indices = {4, 11, 17, 23}
    print(f"  Teacher configured to cache layers: {sorted(teacher.cached_layer_indices)}")

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"✓ Real teacher loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Cached layers: 4 (matching student)")

    return teacher


if __name__ == '__main__':
    # Test loading
    teacher = load_real_teacher(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Teacher test successful")
    print(f"  Type: {type(teacher)}")
    print(f"  Device: {next(teacher.parameters()).device}")
