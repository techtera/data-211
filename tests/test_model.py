"""
Test script for edge_mask/model.py

Gates:
- Full forward pass produces [B, S, 1, 518, 518]
- Training mode returns 3 tensors (logits, ds1, ds2)
- Eval mode returns 1 tensor with values in [0, 1]
- Encoder parameters have no grad after backward
- Decoder parameters have grad after backward
- loss.backward() completes without error
- No NaN/Inf anywhere
- Different B, S combinations work
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "vggt")

import torch
from vggt.models.aggregator import Aggregator
from edge_mask.model import VGGTEdgeMask
from edge_mask.losses import EdgeLoss, compute_total_loss


def check_no_nan_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def test_forward_training_mode():
    print("=" * 60)
    print("  Test 1: Forward pass (training mode)")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.train()

    B, S = 1, 2
    images = torch.rand(B, S, 3, 518, 518)

    print(f"  Input: {images.shape}")

    outputs = model(images)

    assert isinstance(outputs, tuple), "Training mode should return tuple"
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    logits, ds1, ds2 = outputs
    print(f"  logits: {logits.shape}")
    print(f"  ds1:    {ds1.shape}")
    print(f"  ds2:    {ds2.shape}")

    expected = torch.Size([B, S, 1, 518, 518])
    assert logits.shape == expected, f"logits shape: {logits.shape} != {expected}"
    assert ds1.shape == expected, f"ds1 shape: {ds1.shape} != {expected}"
    assert ds2.shape == expected, f"ds2 shape: {ds2.shape} != {expected}"

    check_no_nan_inf(logits, "logits")
    check_no_nan_inf(ds1, "ds1")
    check_no_nan_inf(ds2, "ds2")

    print("  Shape assertions: PASSED")
    print("  NaN/Inf checks: PASSED")
    print("  PASSED\n")


def test_forward_eval_mode():
    print("=" * 60)
    print("  Test 2: Forward pass (eval mode)")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.eval()

    B, S = 1, 2
    images = torch.rand(B, S, 3, 518, 518)

    with torch.no_grad():
        output = model(images)

    assert isinstance(output, torch.Tensor), "Eval mode should return single tensor"
    expected = torch.Size([B, S, 1, 518, 518])
    print(f"  Output: {output.shape}")
    assert output.shape == expected, f"Output shape: {output.shape} != {expected}"

    # Values should be in [0, 1] (sigmoid applied)
    assert output.min() >= 0.0, f"Min value {output.min().item()} < 0"
    assert output.max() <= 1.0, f"Max value {output.max().item()} > 1"
    print(f"  Value range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    check_no_nan_inf(output, "output")
    print("  PASSED\n")


def test_gradient_flow():
    print("=" * 60)
    print("  Test 3: Gradient flow (encoder frozen, decoder trainable)")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.train()

    B, S = 1, 1
    images = torch.rand(B, S, 3, 518, 518)
    target = (torch.rand(B, S, 1, 518, 518) > 0.9).float()

    logits, ds1, ds2 = model(images)

    loss_fn = EdgeLoss()
    total_loss = compute_total_loss(
        logits.view(-1, 1, 518, 518),
        ds1.view(-1, 1, 518, 518),
        ds2.view(-1, 1, 518, 518),
        target.view(-1, 1, 518, 518),
        loss_fn,
    )

    print(f"  Loss value: {total_loss.item():.6f}")
    check_no_nan_inf(total_loss, "total_loss")

    total_loss.backward()

    # Encoder: should have NO gradients
    encoder_has_grad = False
    for name, p in model.feature_extractor.aggregator.named_parameters():
        if p.grad is not None and (p.grad != 0).any():
            encoder_has_grad = True
            print(f"  ERROR: Encoder param {name} has non-zero gradient!")
            break

    assert not encoder_has_grad, "Encoder should not receive gradients"
    print("  Encoder: no gradients ✓")

    # Decoder (projections + decoder + refinement + final_conv): should have gradients
    trainable_modules = [
        ("feature_extractor.projections", model.feature_extractor.projections),
        ("decoder", model.decoder),
        ("refinement", model.refinement),
        ("final_conv", model.final_conv),
    ]

    for mod_name, module in trainable_modules:
        has_grad = False
        for name, p in module.named_parameters():
            if p.grad is not None and (p.grad != 0).any():
                has_grad = True
                break
        assert has_grad, f"{mod_name}: no parameters with gradient"
        print(f"  {mod_name}: has gradients ✓")

    print("  PASSED\n")


def test_loss_backward_completes():
    print("=" * 60)
    print("  Test 4: Full loss.backward() completes")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01,
    )

    B, S = 1, 1
    images = torch.rand(B, S, 3, 518, 518)
    target = (torch.rand(B, S, 1, 518, 518) > 0.9).float()

    # Forward
    logits, ds1, ds2 = model(images)

    # Loss
    loss_fn = EdgeLoss()
    total_loss = compute_total_loss(
        logits.view(-1, 1, 518, 518),
        ds1.view(-1, 1, 518, 518),
        ds2.view(-1, 1, 518, 518),
        target.view(-1, 1, 518, 518),
        loss_fn,
    )

    # Backward
    optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        max_norm=1.0,
    )

    print(f"  Loss: {total_loss.item():.6f}")
    print(f"  Grad norm (before clip): {grad_norm.item():.6f}")

    # Step
    optimizer.step()
    print("  optimizer.step() completed ✓")

    # Verify no NaN in params after step
    for name, p in model.named_parameters():
        if p.requires_grad:
            check_no_nan_inf(p.data, f"param {name}")

    print("  All parameters finite after step ✓")
    print("  PASSED\n")


def test_different_batch_sizes():
    print("=" * 60)
    print("  Test 5: Different B, S combinations")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)
    model.eval()

    configs = [(1, 1), (1, 2), (1, 3)]

    for B, S in configs:
        images = torch.rand(B, S, 3, 518, 518)
        with torch.no_grad():
            output = model(images)
        expected = torch.Size([B, S, 1, 518, 518])
        assert output.shape == expected, f"B={B},S={S}: {output.shape} != {expected}"
        check_no_nan_inf(output, f"output_B{B}_S{S}")
        print(f"  B={B}, S={S}: {output.shape} ✓")

    print("  PASSED\n")


def test_total_parameter_count():
    print("=" * 60)
    print("  Test 6: Total parameter count")
    print("=" * 60)

    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)
    model = VGGTEdgeMask(aggregator)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable (decoder):  {trainable_params:>12,}")
    print(f"  Frozen (encoder):     {frozen_params:>12,}")

    # Breakdown
    proj_params = sum(p.numel() for p in model.feature_extractor.projections.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    ref_params = sum(p.numel() for p in model.refinement.parameters())
    final_params = sum(p.numel() for p in model.final_conv.parameters())

    print(f"\n  Trainable breakdown:")
    print(f"    Projections:  {proj_params:>10,}")
    print(f"    Decoder:      {dec_params:>10,}")
    print(f"    Refinement:   {ref_params:>10,}")
    print(f"    Final conv:   {final_params:>10,}")
    print(f"    Sum:          {proj_params + dec_params + ref_params + final_params:>10,}")

    assert trainable_params == proj_params + dec_params + ref_params + final_params
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MODEL.PY TEST SUITE")
    print("=" * 60 + "\n")

    test_forward_training_mode()
    test_forward_eval_mode()
    test_gradient_flow()
    test_loss_backward_completes()
    test_different_batch_sizes()
    test_total_parameter_count()

    print("=" * 60)
    print("  ALL GATES PASSED - model.py is verified")
    print("=" * 60)
