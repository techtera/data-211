"""Quick test for chunking fix"""
import torch
from model import VGGTUnified

print("Loading model...")
model = VGGTUnified(load_encoder=False)
model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt', device='cpu')
model.eval()

# Test with 10 frames (will trigger chunking: 8+2)
print("\nTesting with 10 frames (baseline, no FastVGGT)...")
images = torch.rand(1, 10, 3, 518, 518)

try:
    with torch.no_grad():
        result = model(images, task='obj')

    obj_mask = result['obj_mask']
    print(f"✓ SUCCESS! Output shape: {list(obj_mask.shape)}")
    print(f"  Expected: [1, 10, 2, 518, 518]")
    print(f"  Match: {list(obj_mask.shape) == [1, 10, 2, 518, 518]}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
