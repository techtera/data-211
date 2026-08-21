"""Test FastVGGT with RoPE enabled"""
import torch
from model import VGGTUnified

print("Testing FastVGGT with RoPE enabled...")
print("=" * 80)

# Load model
model = VGGTUnified(load_encoder=False)
model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt', device='cpu')
model.eval()

# Enable FastVGGT with RoPE
print("Enabling FastVGGT with RoPE (disable_rope=False)...")
model.aggregator.enable_token_merging(merge_ratio=0.9, disable_rope=False)
print("✓ FastVGGT enabled")

# Test with 5 frames
print("\nTesting with 5 frames...")
images = torch.rand(1, 5, 3, 518, 518)

try:
    with torch.no_grad():
        result = model(images, task='obj')

    obj_mask = result['obj_mask']
    print(f"✓ SUCCESS! Output shape: {list(obj_mask.shape)}")
    print(f"  Expected: [1, 5, 2, 518, 518]")
    print(f"  Match: {list(obj_mask.shape) == [1, 5, 2, 518, 518]}")
    print("\n✅ RoPE + Token Merging working correctly!")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\n❌ RoPE + Token Merging failed!")
