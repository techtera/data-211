#!/usr/bin/env python3
"""
Quick test to verify multi-GPU setup works correctly.
"""

import torch
import torch.nn as nn

# Check GPU availability
print("="*60)
print("Multi-GPU Setup Test")
print("="*60)

cuda_available = torch.cuda.is_available()
print(f"\nCUDA available: {cuda_available}")

if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")

    if num_gpus > 1:
        print(f"\n✓ Multi-GPU training available ({num_gpus} GPUs)")

        # Test DataParallel
        print("\nTesting DataParallel...")
        model = nn.Linear(100, 50)
        model = nn.DataParallel(model)
        model = model.cuda()

        # Test forward pass
        x = torch.randn(8, 100).cuda()
        y = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  ✓ DataParallel works!")

    else:
        print(f"\n⚠ Only 1 GPU available - multi-GPU training disabled")
else:
    print(f"\n⚠ CUDA not available - training will use CPU")

print("\n" + "="*60)
