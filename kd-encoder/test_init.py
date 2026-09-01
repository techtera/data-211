#!/usr/bin/env python3
"""Test if initialization corrupts the model dimensions"""

import sys
sys.path.insert(0, '.')
import torch
from student import StudentAggregator, initialize_student_from_dinov2_large

print("="*60)
print("Testing initialization...")
print("="*60)

# Create student
student = StudentAggregator(embed_dim=768, depth=18)
print(f"\nBEFORE init: {student.frame_blocks[0].attn.qkv.weight.shape}")

# Initialize (will download DINOv2-Large first time)
print("\nRunning initialization (may take 5-10 min first time)...")
initialize_student_from_dinov2_large(student, verbose=True)

# Check after
print(f"\nAFTER init: {student.frame_blocks[0].attn.qkv.weight.shape}")

if student.frame_blocks[0].attn.qkv.weight.shape == torch.Size([2304, 768]):
    print("\n✅ SUCCESS - dimensions preserved!")
else:
    print(f"\n❌ FAIL - dimensions corrupted!")
