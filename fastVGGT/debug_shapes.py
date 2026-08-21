"""
Debug script to check intermediate shapes with/without token merging
"""

import torch
from model import VGGTUnified

print("=" * 80)
print("DEBUG: Checking intermediate shapes")
print("=" * 80)

# Load model
model = VGGTUnified(load_encoder=False)
model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt', device='cpu')
model.eval()

# Create dummy input
images = torch.rand(1, 3, 3, 518, 518)  # Small for testing

print("\n--- WITHOUT Token Merging ---")
model.aggregator.disable_token_merging()
with torch.no_grad():
    aggregated_tokens_list, patch_start_idx = model.aggregator(images)

print(f"Number of cached layers: {len([x for x in aggregated_tokens_list if x is not None])}")
for i, tokens in enumerate(aggregated_tokens_list):
    if tokens is not None:
        print(f"Layer {i}: {list(tokens.shape)}")

print("\n--- WITH Token Merging ---")
model.aggregator.enable_token_merging(merge_ratio=0.9, disable_rope=True)
with torch.no_grad():
    aggregated_tokens_list_merged, patch_start_idx = model.aggregator(images)

print(f"Number of cached layers: {len([x for x in aggregated_tokens_list_merged if x is not None])}")
for i, tokens in enumerate(aggregated_tokens_list_merged):
    if tokens is not None:
        print(f"Layer {i}: {list(tokens.shape)}")

print("\n--- Comparison ---")
for i, (t1, t2) in enumerate(zip(aggregated_tokens_list, aggregated_tokens_list_merged)):
    if t1 is not None and t2 is not None:
        match = "✓ MATCH" if t1.shape == t2.shape else "✗ MISMATCH"
        print(f"Layer {i}: {list(t1.shape)} vs {list(t2.shape)} - {match}")

print("\n" + "=" * 80)
print("If all shapes match, FastVGGT is working correctly!")
print("=" * 80)
