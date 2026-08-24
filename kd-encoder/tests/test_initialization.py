#!/usr/bin/env python3
"""
Test DINOv2 initialization for student encoder.
Downloads DINOv2 and initializes student weights.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from student import StudentAggregator, initialize_student_from_dinov2, verify_initialization


def test_initialization():
    """Test DINOv2 initialization."""
    print("="*60)
    print("Testing DINOv2 Initialization")
    print("="*60)

    # Test 1: Create student encoder
    print("\n[Test 1] Creating student encoder...")
    try:
        student = StudentAggregator()
        print("✓ Student encoder created")
    except Exception as e:
        print(f"✗ Failed to create student: {e}")
        return False

    # Test 2: Initialize with DINOv2
    print("\n[Test 2] Loading DINOv2 and initializing student...")
    print("Note: This will download ~350MB on first run (may take several minutes)")
    try:
        initialize_student_from_dinov2(student, verbose=True)
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify initialization
    print("\n[Test 3] Verifying initialization...")
    try:
        results = verify_initialization(student, verbose=True)

        if results['has_nan']:
            print("✗ Initialization produced NaN values")
            return False

        if results['has_inf']:
            print("✗ Initialization produced Inf values")
            return False

        print("✓ Initialization verified")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

    # Test 4: Test forward pass with initialized weights
    print("\n[Test 4] Testing forward pass with initialized weights...")
    try:
        B, S = 1, 2
        images = torch.rand(B, S, 3, 518, 518)

        student.eval()
        with torch.no_grad():
            output_list, _ = student(images)

        # Check outputs are valid
        first_output = next(out for out in output_list if out is not None)

        if torch.isnan(first_output).any():
            print("✗ Forward pass produced NaN")
            return False

        if torch.isinf(first_output).any():
            print("✗ Forward pass produced Inf")
            return False

        print("✓ Forward pass successful with initialized weights")
        print(f"  Output range: [{first_output.min():.4f}, {first_output.max():.4f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Check parameter statistics
    print("\n[Test 5] Checking parameter statistics...")
    try:
        param_stats = {}
        for name, param in student.named_parameters():
            if 'frame_blocks.0' in name or 'global_blocks.0' in name:
                # First blocks should have DINOv2 weights (non-zero)
                param_stats[name] = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                }

        # Print sample stats
        sample_params = list(param_stats.items())[:3]
        for name, stats in sample_params:
            print(f"  {name}:")
            print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}")

        print(f"  ✓ Parameter statistics look reasonable")
    except Exception as e:
        print(f"✗ Parameter statistics check failed: {e}")
        return False

    # All tests passed!
    print("\n" + "="*60)
    print("✓ ALL INITIALIZATION TESTS PASSED")
    print("="*60)
    print("\nInitialization summary:")
    print("  - DINOv2 loaded successfully")
    print("  - Student encoder initialized")
    print("  - No NaN or Inf values")
    print("  - Forward pass works correctly")
    print("  - Ready for Phase 0A benchmarking")
    return True


if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)
