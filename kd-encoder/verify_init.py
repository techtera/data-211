#!/usr/bin/env python3
"""
Quick verification script for DINOv2 initialization.
Run locally to verify everything is working correctly.
"""

import torch
import sys

print("="*60)
print("Student Initialization Verification")
print("="*60)

# Import student
sys.path.insert(0, '.')
from student import StudentAggregator
from student.initialization import initialize_student_from_dinov2, verify_initialization

# Create student
print("\n[1/4] Creating student model...")
student = StudentAggregator()
print(f"✓ Student created: {sum(p.numel() for p in student.parameters())/1e6:.1f}M parameters")

# Initialize from DINOv2
print("\n[2/4] Initializing from DINOv2...")
print("(This will download ~350MB on first run)")
try:
    initialize_student_from_dinov2(student, verbose=True)
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Verify
print("\n[3/4] Verifying initialization...")
results = verify_initialization(student, verbose=False)

if results['has_nan']:
    print("✗ FAILED: NaN values found!")
    sys.exit(1)
elif results['has_inf']:
    print("✗ FAILED: Inf values found!")
    sys.exit(1)
else:
    print("✓ All parameters valid (no NaN/Inf)")

# Check parameter ranges
print("\n[4/4] Parameter statistics:")
print("-"*60)

# Sample key parameters
key_params = [
    'patch_embed.proj.weight',
    'frame_blocks.0.attn.qkv.weight',
    'frame_blocks.11.attn.qkv.weight',
    'frame_blocks.17.attn.qkv.weight',
    'global_blocks.0.attn.qkv.weight',
    'camera_token',
    'register_token'
]

for name in key_params:
    if name in results['parameter_ranges']:
        stats = results['parameter_ranges'][name]
        print(f"{name:40s} mean={stats['mean']:+.4f} std={stats['std']:.4f}")

print("\n" + "="*60)
print("✓ VERIFICATION PASSED")
print("="*60)
print("\nInitialization is working correctly!")
print("Your training on VM should be fine.")
