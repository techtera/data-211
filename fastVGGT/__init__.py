"""
FastVGGT: Training-Free Acceleration for VGGT

This package provides a training-free token merging implementation
to accelerate VGGT inference by 3-4x with minimal quality loss.

Usage:
    from model import VGGTUnified

    model = VGGTUnified()
    model.load_unified_checkpoint('checkpoint.pt')
    model.aggregator.enable_token_merging(merge_ratio=0.9)

    # Now inference is 3-4x faster!
    result = model(images, task='cascade')
"""

__version__ = "1.0.0"
__author__ = "Based on FastVGGT paper (arXiv:2509.02560)"
