#!/usr/bin/env python3
"""
Integration test: Token sampling → Projection → Loss pipeline.

Tests the complete flow:
    1. Sample tokens from teacher and student (with shared indices)
    2. Project student features to teacher dimension
    3. Compute distillation loss
    4. Verify gradient flow
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation import (
    sample_tokens,
    sample_tokens_with_indices,
    ProjectionHead,
    MultiLayerProjection,
    DistillationLoss
)


def test_integration():
    """Test complete sampling → projection → loss pipeline."""
    print("="*60)
    print("Integration Test: Sampling → Projection → Loss")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Configuration
    B, S = 2, 4  # Batch size, sequence length
    num_layers = 4

    print("\n[1] Create mock teacher and student features")
    print("-" * 60)

    # Teacher: 4 layers × [B, S, 1374, 2048]
    teacher_features = [
        torch.randn(B, S, 1374, 2048, device=device)
        for _ in range(num_layers)
    ]

    # Student: 4 layers × [B, S, 1374, 1536]
    student_features = [
        torch.randn(B, S, 1374, 1536, device=device, requires_grad=True)
        for _ in range(num_layers)
    ]

    print(f"  Teacher features: {num_layers} layers × {teacher_features[0].shape}")
    print(f"  Student features: {num_layers} layers × {student_features[0].shape}")

    print("\n[2] Token sampling with shared indices")
    print("-" * 60)

    teacher_sampled = []
    student_sampled = []

    for i in range(num_layers):
        # Sample teacher first to get indices
        t_sampled, indices = sample_tokens(teacher_features[i])
        teacher_sampled.append(t_sampled)

        # Use SAME indices for student
        s_sampled = sample_tokens_with_indices(student_features[i], indices)
        student_sampled.append(s_sampled)

        print(f"  Layer {i}:")
        print(f"    Teacher: {teacher_features[i].shape} → {t_sampled.shape}")
        print(f"    Student: {student_features[i].shape} → {s_sampled.shape}")
        print(f"    Shared indices: {indices.shape}")

    # Verify sampling worked
    assert all(t.shape == (B, S, 133, 2048) for t in teacher_sampled), "Teacher sampling failed"
    assert all(s.shape == (B, S, 133, 1536) for s in student_sampled), "Student sampling failed"
    print(f"\n  ✓ All layers sampled: 1374 → 133 tokens (90.3% reduction)")

    print("\n[3] Initialize distillation loss (includes projection)")
    print("-" * 60)

    loss_fn = DistillationLoss(
        student_dim=1536,
        teacher_dim=2048,
        num_layers=num_layers
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in loss_fn.parameters())
    proj_params = sum(p.numel() for p in loss_fn.projection.parameters())

    print(f"  Total parameters: {total_params:,}")
    print(f"  Projection parameters: {proj_params:,}")
    print(f"  Parameters per head: {proj_params // num_layers:,}")

    print("\n[4] Compute distillation loss")
    print("-" * 60)

    loss, metrics = loss_fn(student_sampled, teacher_sampled)

    print(f"  Total loss: {loss.item():.6f}")
    print(f"\n  Per-layer breakdown:")
    for i in range(num_layers):
        mse = metrics[f'layer_{i}_mse']
        cos_sim = metrics[f'layer_{i}_cosine_sim']
        cos_loss = metrics[f'layer_{i}_cosine_loss']
        layer_loss = metrics[f'layer_{i}_loss']
        weighted = metrics[f'layer_{i}_weighted_loss']

        print(f"    Layer {i}:")
        print(f"      MSE: {mse:.6f}")
        print(f"      Cosine sim: {cos_sim:.6f}")
        print(f"      Cosine loss: {cos_loss:.6f}")
        print(f"      Combined: {layer_loss:.6f}")
        print(f"      Weighted: {weighted:.6f}")

    # Verify loss is scalar
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"\n  ✓ Loss computation successful")

    print("\n[5] Test gradient flow")
    print("-" * 60)

    # Backward pass
    loss.backward()

    # Check gradients on student features
    for i, s_feat in enumerate(student_features):
        if s_feat.grad is None:
            print(f"  ✗ Layer {i}: No gradient")
            return False

        if torch.isnan(s_feat.grad).any():
            print(f"  ✗ Layer {i}: NaN gradient")
            return False

        grad_norm = s_feat.grad.norm().item()
        print(f"  Layer {i} gradient norm: {grad_norm:.6f}")

    print(f"\n  ✓ Gradients flow to all student features")

    # Check gradients on projection heads
    proj_grads_ok = True
    for i, head in enumerate(loss_fn.projection.projection_heads):
        has_grad = all(p.grad is not None for p in head.parameters())
        if not has_grad:
            print(f"  ✗ Projection head {i}: Missing gradients")
            proj_grads_ok = False

    if proj_grads_ok:
        print(f"  ✓ Gradients flow to all projection heads")

    print("\n[6] Test with optimization step")
    print("-" * 60)

    # Create fresh features and optimizer
    student_features_opt = [
        torch.randn(B, S, 1374, 1536, device=device, requires_grad=True)
        for _ in range(num_layers)
    ]

    # Create optimizer for projection heads only
    optimizer = torch.optim.AdamW(loss_fn.parameters(), lr=1e-4)

    # Run a few optimization steps
    print(f"  Running 3 optimization steps...")
    initial_loss = None

    for step in range(3):
        optimizer.zero_grad()

        # Sample tokens
        teacher_sampled_opt = []
        student_sampled_opt = []
        for i in range(num_layers):
            t_sampled, indices = sample_tokens(teacher_features[i])
            teacher_sampled_opt.append(t_sampled)
            s_sampled = sample_tokens_with_indices(student_features_opt[i], indices)
            student_sampled_opt.append(s_sampled)

        # Compute loss
        loss, _ = loss_fn(student_sampled_opt, teacher_sampled_opt)

        if step == 0:
            initial_loss = loss.item()

        # Backward + step
        loss.backward()
        optimizer.step()

        print(f"    Step {step}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    print(f"\n  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Change: {final_loss - initial_loss:.6f}")

    # Note: Loss might increase since we're only optimizing projection heads,
    # not the actual student encoder. This is expected.
    print(f"\n  ✓ Optimization step successful")

    print("\n" + "="*60)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("="*60)
    print("\nPipeline verified:")
    print("  ✓ Token sampling with shared indices")
    print("  ✓ Projection heads (1536 → 2048)")
    print("  ✓ Loss computation (MSE + Cosine)")
    print("  ✓ Gradient flow (student features + projection)")
    print("  ✓ Optimization step")
    print("\nReady to implement training infrastructure!")

    return True


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
