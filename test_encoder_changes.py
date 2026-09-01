#!/usr/bin/env python3
"""
Quick test: verify encoder architecture changes compile correctly.
Tests:
1. Output LayerNorm exists in StudentAggregator
2. DINOv2-Large initialization function exists
3. Forward pass works with new architecture
"""

import torch
import sys
sys.path.insert(0, 'kd-encoder')

from student import StudentAggregator, initialize_student_from_dinov2_large

def test_output_norm():
    """Test 1: Check output_norm exists"""
    print("\n" + "="*60)
    print("Test 1: Output LayerNorm")
    print("="*60)

    student = StudentAggregator(
        embed_dim=768,
        depth=18,
        num_heads=12,
        patch_size=14,
        img_size=518,
    )

    # Check output_norm exists
    assert hasattr(student, 'output_norm'), "❌ output_norm not found!"
    assert isinstance(student.output_norm, torch.nn.LayerNorm), "❌ output_norm is not LayerNorm!"
    assert student.output_norm.normalized_shape == (1536,), f"❌ Wrong dimension: {student.output_norm.normalized_shape}"

    print("✓ output_norm exists")
    print(f"✓ Type: {type(student.output_norm)}")
    print(f"✓ Shape: {student.output_norm.normalized_shape}")
    return True


def test_forward_pass():
    """Test 2: Check forward pass works"""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)

    student = StudentAggregator(
        embed_dim=768,
        depth=18,
        num_heads=12,
        patch_size=14,
        img_size=518,
    )
    student.eval()

    # Create dummy input
    x = torch.randn(2, 1, 3, 518, 518)

    print(f"Input shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output_list, patch_start_idx = student(x)

    print(f"Output list length: {len(output_list)}")
    print(f"Patch start idx: {patch_start_idx}")

    # Check cached outputs
    cached_indices = [3, 8, 13, 17]
    for i, idx in enumerate(cached_indices):
        output = output_list[idx]
        assert output is not None, f"❌ Layer {idx} returned None!"
        print(f"✓ Layer {idx} output shape: {output.shape}")
        assert output.shape[-1] == 1536, f"❌ Wrong output dim: {output.shape[-1]} (expected 1536)"

    print("\n✓ Forward pass successful!")
    print("✓ All cached layers have 1536-dim output")
    return True


def test_dinov2_large_loader():
    """Test 3: Check DINOv2-Large loader exists"""
    print("\n" + "="*60)
    print("Test 3: DINOv2-Large Initialization")
    print("="*60)

    # Just check function exists (don't actually load ~1.2GB model)
    from student.initialization import load_dinov2_vitl14_reg

    print("✓ load_dinov2_vitl14_reg function exists")
    print("✓ initialize_student_from_dinov2_large function exists")
    print("\n⚠ Skipping actual download (would be ~1.2GB)")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Testing Encoder Architecture Changes")
    print("="*70)

    try:
        test_output_norm()
        test_forward_pass()
        test_dinov2_large_loader()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nChanges verified:")
        print("  1. ✓ Output LayerNorm(1536) added to encoder")
        print("  2. ✓ Forward pass works correctly")
        print("  3. ✓ DINOv2-Large initialization function ready")
        print("\nNext: Train encoder from scratch with new architecture")
        return 0

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
