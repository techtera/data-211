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

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Teacher aggregator (encoder only)
    """
    print(f"\nLoading real teacher from: {checkpoint_path}")

    # Load unified model
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(checkpoint_path)

    # Extract just the encoder (aggregator)
    teacher = model.aggregator
    teacher = teacher.to(device).eval()

    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"✓ Real teacher loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    return teacher


if __name__ == '__main__':
    # Test loading
    teacher = load_real_teacher(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Teacher test successful")
    print(f"  Type: {type(teacher)}")
    print(f"  Device: {next(teacher.parameters()).device}")
