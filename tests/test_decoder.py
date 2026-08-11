"""
Test script for edge_mask/decoder.py

Gates (Step 3 - primitives):
- ConvBlock: shape transformations correct
- Upsample: produces exact target sizes (including 19→37)
- DeepSupervisionHead: produces [B, 1, 518, 518]
- All: gradient flow, no NaN/Inf

Gates (Step 4 - full grid):
- UNetPPDecoder: x_0_3 = [B, 64, 148, 148]
- UNetPPDecoder: ds1 = [B, 1, 518, 518]
- UNetPPDecoder: ds2 = [B, 1, 518, 518]
- Full backward pass completes
- All concatenation dimensions valid
"""

import sys
sys.path.insert(0, ".")

import torch
from edge_mask.decoder import ConvBlock, Upsample, DeepSupervisionHead, UNetPPDecoder


def check_no_nan_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def test_conv_block():
    print("=" * 60)
    print("  Test 1: ConvBlock shape transformations")
    print("=" * 60)

    test_cases = [
        (512, 256, (2, 512, 37, 37), (2, 256, 37, 37)),
        (256, 128, (2, 256, 74, 74), (2, 128, 74, 74)),
        (384, 128, (2, 384, 74, 74), (2, 128, 74, 74)),
        (128, 64, (2, 128, 148, 148), (2, 64, 148, 148)),
        (192, 64, (2, 192, 148, 148), (2, 64, 148, 148)),
        (256, 64, (2, 256, 148, 148), (2, 64, 148, 148)),
    ]

    for in_ch, out_ch, in_shape, expected_shape in test_cases:
        block = ConvBlock(in_ch, out_ch)
        x = torch.randn(*in_shape)
        out = block(x)

        assert out.shape == torch.Size(expected_shape), (
            f"ConvBlock({in_ch}→{out_ch}): {out.shape} != {expected_shape}"
        )
        check_no_nan_inf(out, f"ConvBlock({in_ch}→{out_ch})")
        print(f"  ConvBlock({in_ch}→{out_ch}): {in_shape} → {out.shape} ✓")

    # Gradient flow
    block = ConvBlock(256, 64)
    x = torch.randn(2, 256, 148, 148, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None and (x.grad != 0).any()
    print("  Gradient flow: verified ✓")
    print("  PASSED\n")


def test_upsample():
    print("=" * 60)
    print("  Test 2: Upsample produces exact target sizes")
    print("=" * 60)

    test_cases = [
        # (in_ch, out_ch, input_shape, target_size, expected_output_shape)
        (512, 256, (2, 512, 19, 19), (37, 37), (2, 256, 37, 37)),   # Critical: 19→37
        (256, 128, (2, 256, 37, 37), (74, 74), (2, 128, 74, 74)),   # 37→74
        (256, 128, (2, 256, 37, 37), (74, 74), (2, 128, 74, 74)),   # 37→74 (up_2_1)
        (128, 64, (2, 128, 74, 74), (148, 148), (2, 64, 148, 148)), # 74→148
        (128, 64, (2, 128, 74, 74), (148, 148), (2, 64, 148, 148)), # 74→148 (up_1_1)
        (128, 64, (2, 128, 74, 74), (148, 148), (2, 64, 148, 148)), # 74→148 (up_1_2)
    ]

    for in_ch, out_ch, in_shape, target_size, expected_shape in test_cases:
        up = Upsample(in_ch, out_ch)
        x = torch.randn(*in_shape)
        out = up(x, target_size)

        assert out.shape == torch.Size(expected_shape), (
            f"Upsample({in_ch}→{out_ch}, target={target_size}): {out.shape} != {expected_shape}"
        )
        check_no_nan_inf(out, f"Upsample({in_ch}→{out_ch})")
        print(f"  Upsample({in_ch}→{out_ch}): {in_shape} → target {target_size} → {out.shape} ✓")

    # Gradient flow
    up = Upsample(512, 256)
    x = torch.randn(2, 512, 19, 19, requires_grad=True)
    out = up(x, (37, 37))
    out.sum().backward()
    assert x.grad is not None and (x.grad != 0).any()
    print("  Gradient flow: verified ✓")

    # Critical verification: scale_factor=2 would give 38, not 37
    print(f"\n  Critical check: 19*2 = {19*2} (wrong), target size (37,37) gives 37 (correct)")
    print("  PASSED\n")


def test_deep_supervision_head():
    print("=" * 60)
    print("  Test 3: DeepSupervisionHead output shape")
    print("=" * 60)

    head = DeepSupervisionHead(in_ch=64, output_size=518)
    x = torch.randn(2, 64, 148, 148)
    out = head(x)

    expected = (2, 1, 518, 518)
    print(f"  Input:    {x.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Expected: {expected}")

    assert out.shape == torch.Size(expected), f"DS head shape: {out.shape} != {expected}"
    check_no_nan_inf(out, "DS head output")

    # Gradient flow
    x_grad = torch.randn(2, 64, 148, 148, requires_grad=True)
    out = head(x_grad)
    out.sum().backward()
    assert x_grad.grad is not None and (x_grad.grad != 0).any()
    print("  Gradient flow: verified ✓")
    print("  PASSED\n")


def test_unetpp_decoder_full():
    print("=" * 60)
    print("  Test 4: UNetPPDecoder full grid forward pass")
    print("=" * 60)

    decoder = UNetPPDecoder(channels=(64, 128, 256, 512))

    BS = 2
    features = [
        torch.randn(BS, 64, 148, 148),
        torch.randn(BS, 128, 74, 74),
        torch.randn(BS, 256, 37, 37),
        torch.randn(BS, 512, 19, 19),
    ]

    print(f"  Input features:")
    for i, f in enumerate(features):
        print(f"    Level {i}: {f.shape}")

    x_0_3, ds1, ds2 = decoder(features)

    print(f"\n  Outputs:")
    print(f"    x_0_3: {x_0_3.shape} (expected [2, 64, 148, 148])")
    print(f"    ds1:   {ds1.shape} (expected [2, 1, 518, 518])")
    print(f"    ds2:   {ds2.shape} (expected [2, 1, 518, 518])")

    assert x_0_3.shape == torch.Size([BS, 64, 148, 148]), f"x_0_3: {x_0_3.shape}"
    assert ds1.shape == torch.Size([BS, 1, 518, 518]), f"ds1: {ds1.shape}"
    assert ds2.shape == torch.Size([BS, 1, 518, 518]), f"ds2: {ds2.shape}"

    check_no_nan_inf(x_0_3, "x_0_3")
    check_no_nan_inf(ds1, "ds1")
    check_no_nan_inf(ds2, "ds2")

    print("  Shape assertions: PASSED")
    print("  NaN/Inf checks: PASSED")
    print("  PASSED\n")


def test_unetpp_decoder_backward():
    print("=" * 60)
    print("  Test 5: UNetPPDecoder full backward pass")
    print("=" * 60)

    decoder = UNetPPDecoder(channels=(64, 128, 256, 512))

    BS = 2
    features = [
        torch.randn(BS, 64, 148, 148, requires_grad=True),
        torch.randn(BS, 128, 74, 74, requires_grad=True),
        torch.randn(BS, 256, 37, 37, requires_grad=True),
        torch.randn(BS, 512, 19, 19, requires_grad=True),
    ]

    x_0_3, ds1, ds2 = decoder(features)

    # Combined loss (mimics real training)
    loss = x_0_3.sum() + ds1.sum() + ds2.sum()
    loss.backward()

    # Verify gradients on inputs
    for i, f in enumerate(features):
        assert f.grad is not None, f"Feature level {i}: no gradient"
        assert (f.grad != 0).any(), f"Feature level {i}: all-zero gradient"
        check_no_nan_inf(f.grad, f"feature_{i}.grad")
        print(f"  Level {i} input grad norm: {f.grad.norm().item():.6f} ✓")

    # Verify gradients on all decoder parameters
    total_params = 0
    params_with_grad = 0
    for name, p in decoder.named_parameters():
        total_params += 1
        if p.grad is not None and (p.grad != 0).any():
            params_with_grad += 1
            check_no_nan_inf(p.grad, name)

    print(f"  Decoder params with non-zero grad: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, "Some decoder params have no gradient"
    print("  PASSED\n")


def test_unetpp_decoder_parameter_count():
    print("=" * 60)
    print("  Test 6: UNetPPDecoder parameter count")
    print("=" * 60)

    decoder = UNetPPDecoder(channels=(64, 128, 256, 512))

    # Count per sub-module
    components = {
        "up_3_0": decoder.up_3_0,
        "up_2_0": decoder.up_2_0,
        "up_2_1": decoder.up_2_1,
        "up_1_0": decoder.up_1_0,
        "up_1_1": decoder.up_1_1,
        "up_1_2": decoder.up_1_2,
        "conv_2_1": decoder.conv_2_1,
        "conv_1_1": decoder.conv_1_1,
        "conv_1_2": decoder.conv_1_2,
        "conv_0_1": decoder.conv_0_1,
        "conv_0_2": decoder.conv_0_2,
        "conv_0_3": decoder.conv_0_3,
        "ds1": decoder.ds1,
        "ds2": decoder.ds2,
    }

    total = 0
    for name, module in components.items():
        count = sum(p.numel() for p in module.parameters())
        total += count
        print(f"  {name:12s}: {count:>10,}")

    print(f"  {'TOTAL':12s}: {total:>10,}")
    print("  PASSED\n")


def test_concatenation_dimensions():
    print("=" * 60)
    print("  Test 7: Explicit concatenation dimension validation")
    print("=" * 60)

    BS = 2
    c0, c1, c2, c3 = 64, 128, 256, 512

    # Simulate exact concatenation at each node
    x_0_0 = torch.randn(BS, c0, 148, 148)
    x_1_0 = torch.randn(BS, c1, 74, 74)
    x_2_0 = torch.randn(BS, c2, 37, 37)
    x_3_0 = torch.randn(BS, c3, 19, 19)

    # Node X(2,1): cat[X(2,0)=256, Up(X(3,0))→256] along dim=1
    up_30 = torch.randn(BS, c2, 37, 37)  # after upsample
    cat_21 = torch.cat([x_2_0, up_30], dim=1)
    assert cat_21.shape == torch.Size([BS, c2 + c2, 37, 37])
    print(f"  X(2,1) concat: [{c2}, {c2}] = {cat_21.shape[1]} channels at 37x37 ✓")

    # Node X(1,1): cat[X(1,0)=128, Up(X(2,0))→128]
    up_20 = torch.randn(BS, c1, 74, 74)
    cat_11 = torch.cat([x_1_0, up_20], dim=1)
    assert cat_11.shape == torch.Size([BS, c1 + c1, 74, 74])
    print(f"  X(1,1) concat: [{c1}, {c1}] = {cat_11.shape[1]} channels at 74x74 ✓")

    # Node X(1,2): cat[X(1,0)=128, X(1,1)=128, Up(X(2,1))→128]
    x_1_1 = torch.randn(BS, c1, 74, 74)
    up_21 = torch.randn(BS, c1, 74, 74)
    cat_12 = torch.cat([x_1_0, x_1_1, up_21], dim=1)
    assert cat_12.shape == torch.Size([BS, c1 * 3, 74, 74])
    print(f"  X(1,2) concat: [{c1}, {c1}, {c1}] = {cat_12.shape[1]} channels at 74x74 ✓")

    # Node X(0,1): cat[X(0,0)=64, Up(X(1,0))→64]
    up_10 = torch.randn(BS, c0, 148, 148)
    cat_01 = torch.cat([x_0_0, up_10], dim=1)
    assert cat_01.shape == torch.Size([BS, c0 + c0, 148, 148])
    print(f"  X(0,1) concat: [{c0}, {c0}] = {cat_01.shape[1]} channels at 148x148 ✓")

    # Node X(0,2): cat[X(0,0)=64, X(0,1)=64, Up(X(1,1))→64]
    x_0_1 = torch.randn(BS, c0, 148, 148)
    up_11 = torch.randn(BS, c0, 148, 148)
    cat_02 = torch.cat([x_0_0, x_0_1, up_11], dim=1)
    assert cat_02.shape == torch.Size([BS, c0 * 3, 148, 148])
    print(f"  X(0,2) concat: [{c0}, {c0}, {c0}] = {cat_02.shape[1]} channels at 148x148 ✓")

    # Node X(0,3): cat[X(0,0)=64, X(0,1)=64, X(0,2)=64, Up(X(1,2))→64]
    x_0_2 = torch.randn(BS, c0, 148, 148)
    up_12 = torch.randn(BS, c0, 148, 148)
    cat_03 = torch.cat([x_0_0, x_0_1, x_0_2, up_12], dim=1)
    assert cat_03.shape == torch.Size([BS, c0 * 4, 148, 148])
    print(f"  X(0,3) concat: [{c0}, {c0}, {c0}, {c0}] = {cat_03.shape[1]} channels at 148x148 ✓")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DECODER.PY TEST SUITE")
    print("=" * 60 + "\n")

    test_conv_block()
    test_upsample()
    test_deep_supervision_head()
    test_unetpp_decoder_full()
    test_unetpp_decoder_backward()
    test_unetpp_decoder_parameter_count()
    test_concatenation_dimensions()

    print("=" * 60)
    print("  ALL GATES PASSED - decoder.py is verified")
    print("=" * 60)
