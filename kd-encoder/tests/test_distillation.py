#!/usr/bin/env python3
"""
Test distillation components: token sampling, projection, and loss.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation import (
    sample_tokens,
    sample_tokens_with_indices,
    get_sampling_stats,
    ProjectionHead,
    MultiLayerProjection,
    DistillationLoss
)


def test_distillation():
    """Test all distillation components."""
    print("="*60)
    print("Testing Distillation Components")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test 1: Token Sampling
    print("\n[Test 1] Token sampling...")
    try:
        B, S, P, C = 2, 4, 1374, 1536  # Student dimension
        features = torch.randn(B, S, P, C, device=device)

        # Sample tokens
        sampled, indices = sample_tokens(features, patch_start_idx=5, num_patch_samples=128)

        # Check shape
        expected_shape = (B, S, 133, C)  # 5 special + 128 patches
        if sampled.shape != expected_shape:
            print(f"  ✗ Shape mismatch: {sampled.shape} != {expected_shape}")
            return False

        print(f"  Original shape: {features.shape}")
        print(f"  Sampled shape: {sampled.shape}")
        print(f"  Indices shape: {indices.shape}")

        # Test sampling with pre-computed indices
        sampled2 = sample_tokens_with_indices(features, indices, patch_start_idx=5)

        if sampled2.shape != sampled.shape:
            print(f"  ✗ sample_tokens_with_indices shape mismatch")
            return False

        # Get stats
        stats = get_sampling_stats(features.shape, sampled.shape)
        print(f"  Reduction: {stats['reduction_ratio']:.2f}x")
        print(f"  Memory savings: {stats['memory_savings_pct']:.1f}%")

        print(f"  ✓ Token sampling works")
    except Exception as e:
        print(f"✗ Token sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Projection Head
    print("\n[Test 2] Projection head...")
    try:
        projection = ProjectionHead(student_dim=1536, teacher_dim=2048).to(device)

        # Test forward
        x = torch.randn(B, S, 133, 1536, device=device)
        proj = projection(x)

        expected_shape = (B, S, 133, 2048)
        if proj.shape != expected_shape:
            print(f"  ✗ Shape mismatch: {proj.shape} != {expected_shape}")
            return False

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {proj.shape}")
        print(f"  Parameters: {sum(p.numel() for p in projection.parameters()):,}")
        print(f"  ✓ Projection head works")
    except Exception as e:
        print(f"✗ Projection head failed: {e}")
        return False

    # Test 3: Multi-Layer Projection
    print("\n[Test 3] Multi-layer projection...")
    try:
        multi_proj = MultiLayerProjection(
            num_layers=4,
            student_dim=1536,
            teacher_dim=2048
        ).to(device)

        # Create student features (4 layers)
        student_feats = [
            torch.randn(B, S, 133, 1536, device=device)
            for _ in range(4)
        ]

        # Project all layers
        projected = multi_proj(student_feats)

        if len(projected) != 4:
            print(f"  ✗ Expected 4 layers, got {len(projected)}")
            return False

        for i, proj in enumerate(projected):
            expected_shape = (B, S, 133, 2048)
            if proj.shape != expected_shape:
                print(f"  ✗ Layer {i} shape mismatch: {proj.shape}")
                return False

        total_params = sum(p.numel() for p in multi_proj.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Parameters per head: {total_params // 4:,}")
        print(f"  ✓ Multi-layer projection works")
    except Exception as e:
        print(f"✗ Multi-layer projection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Distillation Loss
    print("\n[Test 4] Distillation loss...")
    try:
        loss_fn = DistillationLoss(
            student_dim=1536,
            teacher_dim=2048,
            num_layers=4
        ).to(device)

        # Create student and teacher features
        student_feats = [
            torch.randn(B, S, 133, 1536, device=device)
            for _ in range(4)
        ]
        teacher_feats = [
            torch.randn(B, S, 133, 2048, device=device)
            for _ in range(4)
        ]

        # Compute loss
        loss, metrics = loss_fn(student_feats, teacher_feats)

        # Check loss is scalar
        if loss.dim() != 0:
            print(f"  ✗ Loss should be scalar, got shape {loss.shape}")
            return False

        # Check loss is positive
        if loss.item() <= 0:
            print(f"  ✗ Loss should be positive, got {loss.item()}")
            return False

        print(f"  Total loss: {loss.item():.6f}")
        print(f"  Layer-wise losses:")
        for i in range(4):
            mse = metrics[f'layer_{i}_mse']
            cos_sim = metrics[f'layer_{i}_cosine_sim']
            print(f"    Layer {i}: MSE={mse:.6f}, Cosine Sim={cos_sim:.6f}")

        print(f"  ✓ Distillation loss works")
    except Exception as e:
        print(f"✗ Distillation loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Backward pass (gradient flow)
    print("\n[Test 5] Gradient flow...")
    try:
        # Create fresh tensors with requires_grad
        student_feats = [
            torch.randn(B, S, 133, 1536, device=device, requires_grad=True)
            for _ in range(4)
        ]
        teacher_feats = [
            torch.randn(B, S, 133, 2048, device=device)
            for _ in range(4)
        ]

        # Compute loss and backward
        loss, _ = loss_fn(student_feats, teacher_feats)
        loss.backward()

        # Check gradients
        for i, s_feat in enumerate(student_feats):
            if s_feat.grad is None:
                print(f"  ✗ Layer {i} has no gradient")
                return False
            if torch.isnan(s_feat.grad).any():
                print(f"  ✗ Layer {i} has NaN gradient")
                return False

        print(f"  ✓ Gradients computed successfully")
        print(f"  ✓ No NaN gradients")
        print(f"  ✓ Backward pass works")
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed!
    print("\n" + "="*60)
    print("✓ ALL DISTILLATION TESTS PASSED")
    print("="*60)
    print("\nDistillation components ready:")
    print("  ✓ Token sampling (1374 → 133 tokens)")
    print("  ✓ Projection heads (1536 → 2048)")
    print("  ✓ Distillation loss (MSE + Cosine)")
    print("  ✓ Gradient flow verified")
    print("\nReady for Phase 1 training!")
    return True


if __name__ == "__main__":
    success = test_distillation()
    sys.exit(0 if success else 1)
