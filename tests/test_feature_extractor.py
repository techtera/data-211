"""
Test script for edge_mask/feature_extractor.py

Gates:
- FeatureProjection output shapes correct for all 4 levels
- Outputs have requires_grad=True (trainable projections)
- No NaN/Inf in outputs
- Gradient flow through projections
- VGGTFeatureExtractor with real aggregator (random weights)
- Encoder parameters frozen, projection parameters trainable
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "vggt")

import torch
from edge_mask.feature_extractor import FeatureProjection, VGGTFeatureExtractor


def check_no_nan_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def test_feature_projection_shapes():
    print("=" * 60)
    print("  Test 1: FeatureProjection output shapes (all 4 levels)")
    print("=" * 60)

    BS = 2  # B*S
    x = torch.randn(BS, 2048, 37, 37)

    configs = [
        (64, (148, 148), False, "Level 0: upsample to 148x148"),
        (128, (74, 74), False, "Level 1: upsample to 74x74"),
        (256, None, False, "Level 2: identity (37x37)"),
        (512, None, True, "Level 3: downsample to 19x19"),
    ]

    expected_shapes = [
        (BS, 64, 148, 148),
        (BS, 128, 74, 74),
        (BS, 256, 37, 37),
        (BS, 512, 19, 19),
    ]

    for i, (out_ch, target_size, downsample, desc) in enumerate(configs):
        proj = FeatureProjection(2048, out_ch, target_size=target_size, downsample=downsample)
        out = proj(x)

        print(f"  {desc}")
        print(f"    Input:    {x.shape}")
        print(f"    Output:   {out.shape}")
        print(f"    Expected: {expected_shapes[i]}")

        assert out.shape == torch.Size(expected_shapes[i]), (
            f"Shape mismatch at level {i}: got {out.shape}, expected {expected_shapes[i]}"
        )
        check_no_nan_inf(out, f"level_{i}_output")

    print("  PASSED\n")


def test_projection_gradient_flow():
    print("=" * 60)
    print("  Test 2: Gradient flow through projections")
    print("=" * 60)

    BS = 2
    x = torch.randn(BS, 2048, 37, 37)

    configs = [
        (64, (148, 148), False),
        (128, (74, 74), False),
        (256, None, False),
        (512, None, True),
    ]

    for i, (out_ch, target_size, downsample) in enumerate(configs):
        proj = FeatureProjection(2048, out_ch, target_size=target_size, downsample=downsample)
        out = proj(x)
        loss = out.sum()
        loss.backward()

        has_grad = False
        for name, p in proj.named_parameters():
            assert p.grad is not None, f"Level {i}, param {name}: no gradient"
            check_no_nan_inf(p.grad, f"level_{i}_{name}.grad")
            if (p.grad != 0).any():
                has_grad = True

        assert has_grad, f"Level {i}: all gradients are zero"
        print(f"  Level {i}: gradients verified (non-zero, no NaN/Inf)")

    print("  PASSED\n")


def test_projection_requires_grad():
    print("=" * 60)
    print("  Test 3: Projection outputs have requires_grad=True")
    print("=" * 60)

    BS = 2
    x = torch.randn(BS, 2048, 37, 37)

    proj = FeatureProjection(2048, 64, target_size=(148, 148))
    out = proj(x)

    assert out.requires_grad, "Output does not have requires_grad=True"
    print(f"  Output requires_grad: {out.requires_grad}")
    print("  PASSED\n")


def test_projection_parameter_count():
    print("=" * 60)
    print("  Test 4: Parameter counts")
    print("=" * 60)

    configs = [
        (64, (148, 148), False),
        (128, (74, 74), False),
        (256, None, False),
        (512, None, True),
    ]

    total = 0
    for i, (out_ch, target_size, downsample) in enumerate(configs):
        proj = FeatureProjection(2048, out_ch, target_size=target_size, downsample=downsample)
        count = sum(p.numel() for p in proj.parameters())
        total += count
        print(f"  Level {i} (out_ch={out_ch}): {count:,} parameters")

    print(f"  Total projections: {total:,} parameters")
    print("  PASSED\n")


def test_with_real_aggregator():
    print("=" * 60)
    print("  Test 5: VGGTFeatureExtractor with real aggregator")
    print("=" * 60)

    from vggt.models.aggregator import Aggregator

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    extractor = VGGTFeatureExtractor(aggregator)

    B, S = 1, 2
    images = torch.rand(B, S, 3, 518, 518)

    print(f"  Input: {images.shape}")

    features = extractor(images)

    expected_shapes = [
        (B * S, 64, 148, 148),
        (B * S, 128, 74, 74),
        (B * S, 256, 37, 37),
        (B * S, 512, 19, 19),
    ]

    print(f"  Number of feature levels: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape} (expected {expected_shapes[i]})")
        assert feat.shape == torch.Size(expected_shapes[i]), (
            f"Level {i} shape mismatch: {feat.shape} != {expected_shapes[i]}"
        )
        check_no_nan_inf(feat, f"feature_level_{i}")

    print("  PASSED\n")


def test_encoder_frozen_decoder_trainable():
    print("=" * 60)
    print("  Test 6: Encoder frozen, projections trainable")
    print("=" * 60)

    from vggt.models.aggregator import Aggregator

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    extractor = VGGTFeatureExtractor(aggregator)

    # Check encoder frozen
    encoder_params = list(extractor.aggregator.parameters())
    frozen_count = sum(1 for p in encoder_params if not p.requires_grad)
    print(f"  Encoder parameters: {len(encoder_params)}")
    print(f"  Frozen (requires_grad=False): {frozen_count}")
    assert frozen_count == len(encoder_params), "Some encoder params are not frozen"

    # Check projections trainable
    proj_params = list(extractor.projections.parameters())
    trainable_count = sum(1 for p in proj_params if p.requires_grad)
    print(f"  Projection parameters: {len(proj_params)}")
    print(f"  Trainable (requires_grad=True): {trainable_count}")
    assert trainable_count == len(proj_params), "Some projection params are not trainable"

    # Verify gradient flow only through projections
    B, S = 1, 1
    images = torch.rand(B, S, 3, 518, 518)
    features = extractor(images)

    loss = sum(f.sum() for f in features)
    loss.backward()

    # Encoder should have no grad
    for name, p in extractor.aggregator.named_parameters():
        assert p.grad is None or (p.grad == 0).all(), (
            f"Encoder param {name} has non-zero gradient"
        )

    # Projections should have grad
    for name, p in extractor.projections.named_parameters():
        assert p.grad is not None, f"Projection param {name} has no gradient"
        assert (p.grad != 0).any(), f"Projection param {name} has all-zero gradient"

    print("  Encoder: no gradients (verified)")
    print("  Projections: gradients present (verified)")
    print("  PASSED\n")


def test_batch_sizes():
    print("=" * 60)
    print("  Test 7: Different batch sizes")
    print("=" * 60)

    # Test with synthetic features (skip full aggregator for speed)
    configs = [(1, 1), (1, 2), (2, 1), (2, 3)]

    for B, S in configs:
        BS = B * S
        x = torch.randn(BS, 2048, 37, 37)

        proj = FeatureProjection(2048, 64, target_size=(148, 148))
        out = proj(x)

        expected = (BS, 64, 148, 148)
        assert out.shape == torch.Size(expected), f"B={B},S={S}: {out.shape} != {expected}"
        check_no_nan_inf(out, f"B{B}_S{S}")
        print(f"  B={B}, S={S} (B*S={BS}): {out.shape} ✓")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FEATURE_EXTRACTOR.PY TEST SUITE")
    print("=" * 60 + "\n")

    test_feature_projection_shapes()
    test_projection_gradient_flow()
    test_projection_requires_grad()
    test_projection_parameter_count()
    test_with_real_aggregator()
    test_encoder_frozen_decoder_trainable()
    test_batch_sizes()

    print("=" * 60)
    print("  ALL GATES PASSED - feature_extractor.py is verified")
    print("=" * 60)
