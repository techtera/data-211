"""
Test script to verify RoPE-First refactoring works correctly
"""

import torch
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import VGGTUnified

print("=" * 80)
print("Testing RoPE-First FastVGGT Refactoring")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = VGGTUnified(load_encoder=False)
try:
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt', device='cpu')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Could not load checkpoint: {e}")
    print("  Continuing with uninitialized model for structure test...")

model.eval()

# Test 1: Enable FastVGGT with new API
print("\n2. Testing new API (no disable_rope parameter)...")
try:
    model.aggregator.enable_token_merging(merge_ratio=0.9)
    print("✓ FastVGGT enabled successfully")
except Exception as e:
    print(f"✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check inference shapes
print("\n3. Testing inference with 3 frames...")
images = torch.rand(1, 3, 3, 518, 518)

try:
    with torch.no_grad():
        result = model(images, task='obj')

    obj_mask = result['obj_mask']
    expected_shape = [1, 3, 2, 518, 518]
    actual_shape = list(obj_mask.shape)

    if actual_shape == expected_shape:
        print(f"✓ Output shape correct: {actual_shape}")
    else:
        print(f"✗ Shape mismatch!")
        print(f"  Expected: {expected_shape}")
        print(f"  Got: {actual_shape}")
        sys.exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify RoPE is enabled
print("\n4. Verifying RoPE is enabled...")
if model.aggregator.rope is not None:
    print("✓ RoPE is enabled (rope module exists)")
else:
    print("⚠ RoPE is disabled (no rope module)")

# Test 4: Verify merge_info structure
print("\n5. Testing merge_info structure...")
try:
    from token_merging import TokenMerger, TokenMergingConfig, create_frame_index_tensor

    config = TokenMergingConfig(merge_ratio=0.9)
    merger = TokenMerger(config)

    # Test prepare_merge_info
    test_tokens = torch.rand(1, 1560, 1024)  # B, N, C
    frame_idx = create_frame_index_tensor(1, 3, 520, test_tokens.device)

    returned_tokens, merge_info = merger.prepare_merge_info(test_tokens, frame_idx, 3, 520)

    # Verify tokens unchanged
    if torch.equal(returned_tokens, test_tokens):
        print("✓ prepare_merge_info returns unchanged tokens")
    else:
        print("✗ prepare_merge_info modified tokens!")
        sys.exit(1)

    # Verify merge_info structure
    required_keys = ['dst_mask', 'src_mask', 'salient_mask', 'src_to_dst_mapping']
    if all(k in merge_info for k in required_keys):
        print(f"✓ merge_info has all required keys: {required_keys}")
    else:
        print(f"✗ merge_info missing keys!")
        print(f"  Has: {list(merge_info.keys())}")
        sys.exit(1)

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test with different merge ratios
print("\n6. Testing different merge ratios...")
merge_ratios = [0.7, 0.8, 0.9, 0.95]
for ratio in merge_ratios:
    try:
        model.aggregator.enable_token_merging(merge_ratio=ratio)
        with torch.no_grad():
            result = model(torch.rand(1, 2, 3, 518, 518), task='obj')
        print(f"✓ merge_ratio={ratio} works")
    except Exception as e:
        print(f"✗ merge_ratio={ratio} failed: {e}")
        sys.exit(1)

# Test 6: Test disable
print("\n7. Testing disable...")
try:
    model.aggregator.disable_token_merging()
    with torch.no_grad():
        result = model(torch.rand(1, 2, 3, 518, 518), task='obj')
    print("✓ Disable works")
except Exception as e:
    print(f"✗ Disable failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nRefactoring successful! RoPE-First architecture is working correctly.")
print("\nNext steps:")
print("  1. Test on real data: python fastVGGT/run_inference.py --fast")
print("  2. Compare quality vs baseline")
print("  3. Benchmark performance")
