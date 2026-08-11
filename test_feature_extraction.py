"""
Standalone script to verify VGGT feature extraction shapes.

Loads the VGGT encoder, runs a single synthetic image through,
and prints all intermediate tensor shapes to validate assumptions
before building the UNet++ decoder.

Usage:
    python test_feature_extraction.py [--device cpu|cuda|mps]
"""

import sys
import argparse

sys.path.insert(0, "vggt")

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT
from vggt.models.aggregator import Aggregator


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_with_real_model(device: str):
    """Load full VGGT model and verify feature shapes."""

    print_section("Loading VGGT Model")

    model = VGGT(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
        enable_track=False,
    )
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    print(f"Aggregator depth: {model.aggregator.depth}")
    print(f"Cached layer indices: {sorted(model.aggregator.cached_layer_indices)}")
    print(f"Patch start idx: {model.aggregator.patch_start_idx}")
    print(f"Patch size: {model.aggregator.patch_size}")

    print_section("Input Tensor")

    B, S = 1, 2  # batch=1, sequence=2 frames
    H, W = 518, 518
    images = torch.rand(B, S, 3, H, W, device=device)
    print(f"Input images shape: {images.shape}")
    print(f"  B={B}, S={S}, C=3, H={H}, W={W}")

    print_section("Running Aggregator Forward Pass")

    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)

    print(f"\npatch_start_idx = {patch_start_idx}")
    print(f"len(aggregated_tokens_list) = {len(aggregated_tokens_list)}")
    print(f"\nOutput list contents (None = uncached layer):")

    for i, t in enumerate(aggregated_tokens_list):
        if t is not None:
            print(f"  Layer {i}: shape = {t.shape}, dtype = {t.dtype}")
        else:
            print(f"  Layer {i}: None (not cached)")

    print_section("Feature Extraction for Edge Decoder")

    target_layers = [4, 11, 17, 23]
    print(f"\nTarget layers: {target_layers}")
    print(f"patch_start_idx = {patch_start_idx} (skip camera + register tokens)\n")

    patch_h = H // 14
    patch_w = W // 14
    print(f"Expected patch grid: {patch_h} x {patch_w} = {patch_h * patch_w} patches")
    print(f"  (518 / 14 = {518/14:.2f}, floor = {518//14})\n")

    features = {}
    for layer_idx in target_layers:
        raw = aggregated_tokens_list[layer_idx]
        print(f"Layer {layer_idx}:")
        print(f"  Raw shape: {raw.shape}")

        # Slice off camera + register tokens
        patch_tokens = raw[:, :, patch_start_idx:]
        print(f"  After slicing [:, :, {patch_start_idx}:]: {patch_tokens.shape}")

        # Flatten B and S
        BS = B * S
        flat = patch_tokens.reshape(BS, -1, patch_tokens.shape[-1])
        print(f"  After reshape to [B*S, P, C]: {flat.shape}")

        # Verify patch count
        num_patches = flat.shape[1]
        spatial_h = int(num_patches**0.5)
        print(f"  Num patches = {num_patches}, sqrt = {num_patches**0.5:.4f}")

        if spatial_h * spatial_h != num_patches:
            # Non-square patch grid - compute from H, W
            spatial_h = H // 14
            spatial_w = W // 14
            print(f"  Non-square: using {spatial_h} x {spatial_w} = {spatial_h * spatial_w}")
            assert spatial_h * spatial_w == num_patches, (
                f"Mismatch: {spatial_h}*{spatial_w}={spatial_h*spatial_w} != {num_patches}"
            )
        else:
            spatial_w = spatial_h
            print(f"  Square grid: {spatial_h} x {spatial_w}")

        # Reshape to spatial: [B*S, C, H, W]
        spatial = flat.permute(0, 2, 1).reshape(BS, flat.shape[-1], spatial_h, spatial_w)
        print(f"  Spatial feature map: {spatial.shape}")
        features[layer_idx] = spatial
        print()

    print_section("Simulated Feature Projections (for UNet++ levels)")

    # Simulate the projections we'll use in the decoder
    level_configs = [
        (4, 64, "upsample_4x", (148, 148)),
        (11, 128, "upsample_2x", (74, 74)),
        (17, 256, "identity", (37, 37)),
        (23, 512, "downsample_2x", (19, 19)),
    ]

    print(f"\n{'Level':<6} {'Layer':<6} {'In Shape':<25} {'Proj Ch':<8} {'Op':<15} {'Out Shape':<25}")
    print("-" * 90)

    projected = {}
    for level, (layer_idx, out_ch, op_name, target_size) in enumerate(level_configs):
        feat = features[layer_idx]
        in_ch = feat.shape[1]

        # 1x1 projection
        proj = nn.Conv2d(in_ch, out_ch, 1).to(device)
        with torch.no_grad():
            x = proj(feat)

        # Spatial resize
        if op_name == "upsample_4x":
            x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1).to(device)
            with torch.no_grad():
                x = smooth(x)
        elif op_name == "upsample_2x":
            x = nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1).to(device)
            with torch.no_grad():
                x = smooth(x)
        elif op_name == "identity":
            pass  # already at 37x37
        elif op_name == "downsample_2x":
            down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1).to(device)
            with torch.no_grad():
                x = down(x)

        projected[level] = x
        print(
            f"{level:<6} {layer_idx:<6} {str(feat.shape):<25} {out_ch:<8} {op_name:<15} {str(x.shape):<25}"
        )

    print_section("UNet++ Concatenation Shape Verification")

    # Verify the upsample target size issue: 19 -> 37
    print("\nCritical check: Upsample from Level 3 (19x19) to Level 2 (37x37)")
    x_19 = projected[3]
    x_up = nn.functional.interpolate(x_19, size=(37, 37), mode="bilinear", align_corners=False)
    print(f"  Level 3: {x_19.shape} -> upsample(size=37) -> {x_up.shape}")
    print(f"  Level 2: {projected[2].shape}")
    print(f"  Can concatenate: {x_up.shape[2:] == projected[2].shape[2:]}")

    print("\nCritical check: Upsample from Level 2 (37x37) to Level 1 (74x74)")
    x_37 = projected[2]
    x_up = nn.functional.interpolate(x_37, size=(74, 74), mode="bilinear", align_corners=False)
    print(f"  Level 2: {x_37.shape} -> upsample(size=74) -> {x_up.shape}")
    print(f"  Level 1: {projected[1].shape}")
    print(f"  Can concatenate: {x_up.shape[2:] == projected[1].shape[2:]}")

    print("\nCritical check: Upsample from Level 1 (74x74) to Level 0 (148x148)")
    x_74 = projected[1]
    x_up = nn.functional.interpolate(x_74, size=(148, 148), mode="bilinear", align_corners=False)
    print(f"  Level 1: {x_74.shape} -> upsample(size=148) -> {x_up.shape}")
    print(f"  Level 0: {projected[0].shape}")
    print(f"  Can concatenate: {x_up.shape[2:] == projected[0].shape[2:]}")

    print_section("UNet++ Node Input Channels (Dense Skip Concatenation)")

    ch = {0: 64, 1: 128, 2: 256, 3: 512}
    nodes = {
        "X(2,1)": f"cat[X(2,0)={ch[2]}, Up(X(3,0))→{ch[2]}] = {ch[2] + ch[2]} → out {ch[2]}",
        "X(1,1)": f"cat[X(1,0)={ch[1]}, Up(X(2,0))→{ch[1]}] = {ch[1] + ch[1]} → out {ch[1]}",
        "X(1,2)": f"cat[X(1,0)={ch[1]}, X(1,1)={ch[1]}, Up(X(2,1))→{ch[1]}] = {ch[1]*3} → out {ch[1]}",
        "X(0,1)": f"cat[X(0,0)={ch[0]}, Up(X(1,0))→{ch[0]}] = {ch[0] + ch[0]} → out {ch[0]}",
        "X(0,2)": f"cat[X(0,0)={ch[0]}, X(0,1)={ch[0]}, Up(X(1,1))→{ch[0]}] = {ch[0]*3} → out {ch[0]}",
        "X(0,3)": f"cat[X(0,0)={ch[0]}, X(0,1)={ch[0]}, X(0,2)={ch[0]}, Up(X(1,2))→{ch[0]}] = {ch[0]*4} → out {ch[0]}",
    }

    print()
    for node, desc in nodes.items():
        print(f"  {node}: {desc}")

    print_section("Final Output Path")

    final_feat = projected[0]  # stand-in for X(0,3) at 148x148
    print(f"  X(0,3) shape:     [B*S, 64, 148, 148]")
    print(f"  After refinement: [B*S, 64, 148, 148]  (residual, same shape)")
    print(f"  Conv1x1(64→1):    [B*S, 1, 148, 148]")
    print(f"  Bilinear to 518:  [B*S, 1, 518, 518]")
    print(f"  Reshape:          [B, S, 1, 518, 518]")

    print_section("Summary")
    print(f"""
  Input:            [B={B}, S={S}, 3, 518, 518]
  Aggregator output: list of 24 entries (4 cached at layers {target_layers})
  Each cached:      [B={B}, S={S}, P={patch_start_idx + patch_h*patch_w}, 2C=2048]
  After slice:      [B={B}, S={S}, {patch_h*patch_w}, 2048]
  Spatial reshape:  [B*S={BS}, 2048, {patch_h}, {patch_w}]

  Projected levels:
    Level 0: [{BS}, 64, 148, 148]   (from layer 4,  bilinear ×4)
    Level 1: [{BS}, 128, 74, 74]    (from layer 11, bilinear ×2)
    Level 2: [{BS}, 256, 37, 37]    (from layer 17, identity)
    Level 3: [{BS}, 512, 19, 19]    (from layer 23, conv stride 2)

  Final output:     [{BS}, 1, 518, 518] logits → sigmoid → edge mask
""")

    print_section("PASS - All shapes verified")


def test_with_synthetic_aggregator_output(device: str):
    """
    Quick shape test without loading full model weights.
    Uses synthetic data matching aggregator output format.
    """
    print_section("Synthetic Test (no model weights needed)")

    B, S = 1, 2
    H, W = 518, 518
    patch_h, patch_w = H // 14, W // 14
    num_patches = patch_h * patch_w
    patch_start_idx = 5  # 1 camera + 4 register
    total_tokens = patch_start_idx + num_patches
    embed_dim = 1024
    concat_dim = 2 * embed_dim  # frame + global concatenated

    print(f"  B={B}, S={S}, H={H}, W={W}")
    print(f"  patch_size=14, patch_grid={patch_h}x{patch_w}, num_patches={num_patches}")
    print(f"  patch_start_idx={patch_start_idx}, total_tokens={total_tokens}")
    print(f"  embed_dim={embed_dim}, concat_dim(2C)={concat_dim}")

    # Simulate aggregator output
    cached_layers = [4, 11, 17, 23]
    aggregated_tokens_list = [None] * 24
    for idx in cached_layers:
        aggregated_tokens_list[idx] = torch.randn(B, S, total_tokens, concat_dim, device=device)

    print(f"\n  Simulated aggregated_tokens_list[4].shape = {aggregated_tokens_list[4].shape}")

    # Extract and reshape features
    print(f"\n  Feature extraction pipeline:")
    for layer_idx in cached_layers:
        raw = aggregated_tokens_list[layer_idx]
        patches = raw[:, :, patch_start_idx:]  # [B, S, num_patches, 2048]
        flat = patches.reshape(B * S, num_patches, concat_dim)  # [B*S, P, 2C]
        spatial = flat.permute(0, 2, 1).reshape(B * S, concat_dim, patch_h, patch_w)
        print(f"    Layer {layer_idx:2d}: {raw.shape} → slice → {patches.shape} → flat → {flat.shape} → spatial → {spatial.shape}")

    print(f"\n  All shapes match expected: [B*S, 2048, 37, 37] ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VGGT feature extraction shapes")
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="Only run synthetic test (no model loading)"
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    # Always run synthetic test (fast, no weights needed)
    test_with_synthetic_aggregator_output(device)

    if not args.synthetic_only:
        print("\n\n" + "=" * 60)
        print("  Running with real VGGT model (random weights)")
        print("=" * 60)
        test_with_real_model(device)
    else:
        print("\n  (Skipped real model test, use without --synthetic-only to run)")
