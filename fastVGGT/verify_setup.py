"""
Quick verification script to ensure FastVGGT is set up correctly.
Run this first before testing with actual checkpoints.
"""

import sys
import os

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking imports...")
    errors = []

    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ PyTorch not found: {e}")

    try:
        from token_merging import TokenMerger, TokenMergingConfig
        print("  ✓ token_merging module")
    except ImportError as e:
        errors.append(f"  ✗ token_merging import failed: {e}")

    try:
        from model import VGGTUnified
        print("  ✓ model.VGGTUnified")
    except ImportError as e:
        errors.append(f"  ✗ model import failed: {e}")

    try:
        from encoder import Aggregator
        print("  ✓ encoder.Aggregator")
    except ImportError as e:
        errors.append(f"  ✗ encoder import failed: {e}")

    return errors


def check_methods():
    """Check if FastVGGT methods are available."""
    print("\nChecking FastVGGT methods...")
    errors = []

    try:
        from encoder import Aggregator
        agg = Aggregator(img_size=518, patch_size=14, embed_dim=1024)

        if hasattr(agg, 'enable_token_merging'):
            print("  ✓ enable_token_merging() method exists")
        else:
            errors.append("  ✗ enable_token_merging() method not found")

        if hasattr(agg, 'disable_token_merging'):
            print("  ✓ disable_token_merging() method exists")
        else:
            errors.append("  ✗ disable_token_merging() method not found")

        if hasattr(agg, 'use_token_merging'):
            print("  ✓ use_token_merging attribute exists")
        else:
            errors.append("  ✗ use_token_merging attribute not found")

    except Exception as e:
        errors.append(f"  ✗ Method check failed: {e}")

    return errors


def check_files():
    """Check if all required files are present."""
    print("\nChecking files...")
    errors = []

    required_files = [
        'token_merging.py',
        'model.py',
        'test_fastvggt.py',
        'example_usage.py',
        'README.md',
        'QUICKSTART.md',
        'encoder/aggregator.py',
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            errors.append(f"  ✗ {file} not found")

    return errors


def test_basic_functionality():
    """Test basic functionality without loading checkpoints."""
    print("\nTesting basic functionality...")
    errors = []

    try:
        import torch
        from token_merging import TokenMerger, TokenMergingConfig

        # Test config creation
        config = TokenMergingConfig(merge_ratio=0.9)
        print("  ✓ TokenMergingConfig created")

        # Test merger creation
        merger = TokenMerger(config)
        print("  ✓ TokenMerger created")

        # Test with dummy data
        B, N, C = 2, 100, 1024
        x = torch.rand(B, N, C)
        frame_idx = torch.arange(N).unsqueeze(0).expand(B, -1)

        dst, src, salient, merge_info = merger.partition_tokens(
            x, frame_idx, num_frames=5, tokens_per_frame=20
        )
        print(f"  ✓ Token partitioning works (dst:{dst.shape[1]}, src:{src.shape[1]}, salient:{salient.shape[1]})")

        # Test merging
        merged, mapping = merger.merge_tokens(dst, src)
        print(f"  ✓ Token merging works (merged shape: {merged.shape})")

        # Test unmerging
        merged_full = torch.cat([merged, salient], dim=1)
        merge_info['num_dst'] = merged.shape[1]
        merge_info['num_salient'] = salient.shape[1]
        merge_info['src_to_dst_mapping'] = mapping
        restored = merger.unmerge_tokens(merged_full, merge_info)
        print(f"  ✓ Token unmerging works (restored shape: {restored.shape})")

        # Verify shape preservation
        if restored.shape == x.shape:
            print("  ✓ Shape preservation verified")
        else:
            errors.append(f"  ✗ Shape mismatch: {restored.shape} != {x.shape}")

    except Exception as e:
        errors.append(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()

    return errors


def main():
    print("=" * 80)
    print("FastVGGT Setup Verification")
    print("=" * 80)
    print()

    all_errors = []

    # Run checks
    all_errors.extend(check_files())
    all_errors.extend(check_imports())
    all_errors.extend(check_methods())
    all_errors.extend(test_basic_functionality())

    # Summary
    print("\n" + "=" * 80)
    if not all_errors:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("FastVGGT is correctly set up and ready to use.")
        print()
        print("Next steps:")
        print("  1. Run: python test_fastvggt.py --checkpoint <your_checkpoint> --num-frames 10")
        print("  2. Or check: python example_usage.py 1")
        print()
        return 0
    else:
        print("❌ ERRORS FOUND:")
        print()
        for error in all_errors:
            print(error)
        print()
        print("Please fix the errors above before proceeding.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
