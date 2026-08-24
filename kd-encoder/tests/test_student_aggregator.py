#!/usr/bin/env python3
"""
Test student aggregator implementation.
Verifies architecture builds correctly and forward pass works.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from student import StudentAggregator


def test_student_aggregator():
    """Test student encoder initialization and forward pass."""
    print("="*60)
    print("Testing Student Aggregator")
    print("="*60)

    # Test 1: Initialization
    print("\n[Test 1] Initializing student encoder...")
    try:
        student = StudentAggregator(
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=18,
            num_heads=12,
            mlp_ratio=4.0,
            num_register_tokens=4,
            cached_layer_indices=(3, 8, 13, 17)
        )
        print("✓ Student encoder initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

    # Test 2: Parameter count
    print("\n[Test 2] Counting parameters...")
    try:
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Check if close to expected ~342M
        if total_params > 500_000_000:
            print(f"  ⚠ Warning: Parameter count ({total_params:,}) seems too high (expected ~342M)")
        else:
            print(f"  ✓ Parameter count seems reasonable")
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
        return False

    # Test 3: Forward pass with small batch
    print("\n[Test 3] Testing forward pass...")
    try:
        B, S = 1, 2  # Small batch for testing
        H, W = 518, 518

        # Create random input (normalized to [0, 1])
        images = torch.rand(B, S, 3, H, W)
        print(f"  Input shape: {list(images.shape)}")

        # Forward pass
        student.eval()
        with torch.no_grad():
            output_list, patch_start_idx = student(images)

        print(f"  ✓ Forward pass completed")
        print(f"  Patch start index: {patch_start_idx}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Verify output shapes
    print("\n[Test 4] Verifying output shapes...")
    try:
        expected_cached = {3, 8, 13, 17}
        expected_P = 1374  # 1 camera + 4 register + 1369 patches
        expected_C = 768 * 2  # frame + global concatenated

        for layer_idx, output in enumerate(output_list):
            if output is not None:
                if layer_idx not in expected_cached:
                    print(f"  ⚠ Warning: Layer {layer_idx} cached but not in expected set")

                expected_shape = (B, S, expected_P, expected_C)
                actual_shape = tuple(output.shape)

                if actual_shape != expected_shape:
                    print(f"  ✗ Layer {layer_idx} shape mismatch:")
                    print(f"    Expected: {expected_shape}")
                    print(f"    Got: {actual_shape}")
                    return False
                else:
                    print(f"  ✓ Layer {layer_idx}: shape {list(actual_shape)} correct")

        # Check all expected layers are cached
        cached_layers = {i for i, output in enumerate(output_list) if output is not None}
        print(f"\n  Cached layers: {sorted(cached_layers)}")
        print(f"  Expected layers: {sorted(expected_cached)}")

        if cached_layers == expected_cached:
            print(f"  ✓ All expected layers cached")
        else:
            missing = expected_cached - cached_layers
            extra = cached_layers - expected_cached
            if missing:
                print(f"  ⚠ Missing layers: {sorted(missing)}")
            if extra:
                print(f"  ⚠ Extra layers: {sorted(extra)}")
    except Exception as e:
        print(f"✗ Output verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Test with different batch sizes
    print("\n[Test 5] Testing different batch/sequence sizes...")
    try:
        test_cases = [
            (1, 1, "Single image"),
            (2, 4, "Small batch"),
            (1, 8, "Long sequence"),
        ]

        for B, S, desc in test_cases:
            images = torch.rand(B, S, 3, 518, 518)
            with torch.no_grad():
                output_list, _ = student(images)

            # Check first cached layer shape
            first_cached = next(out for out in output_list if out is not None)
            expected_shape = (B, S, 1374, 1536)

            if tuple(first_cached.shape) == expected_shape:
                print(f"  ✓ {desc} (B={B}, S={S}): shape correct")
            else:
                print(f"  ✗ {desc} (B={B}, S={S}): shape mismatch")
                print(f"    Expected: {expected_shape}")
                print(f"    Got: {first_cached.shape}")
                return False
    except Exception as e:
        print(f"✗ Batch size testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Verify trainable parameters
    print("\n[Test 6] Verifying trainable parameters...")
    try:
        # All parameters should be trainable (no frozen components)
        non_trainable = sum(p.numel() for p in student.parameters() if not p.requires_grad)

        if non_trainable == 0:
            print(f"  ✓ All parameters trainable ({trainable_params:,})")
        else:
            print(f"  ⚠ Warning: {non_trainable:,} non-trainable parameters found")
    except Exception as e:
        print(f"✗ Trainable parameter check failed: {e}")
        return False

    # All tests passed!
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print(f"\nStudent encoder summary:")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Depth: 18 layers")
    print(f"  - Dimension: 768")
    print(f"  - Heads: 12")
    print(f"  - Cached layers: {sorted(expected_cached)}")
    print(f"  - Token count: 1374 (1 camera + 4 register + 1369 patches)")
    print(f"  - Output dimension: 1536 (768 frame + 768 global)")
    return True


if __name__ == "__main__":
    success = test_student_aggregator()
    sys.exit(0 if success else 1)
