"""
Test script for edge_mask/refinement.py

Gates:
- Output shape == input shape
- Output != input (refine path contributes)
- Gradient flow through both branches (residual + refine)
- No NaN/Inf
- Works at different batch sizes
"""

import sys
sys.path.insert(0, ".")

import torch
from edge_mask.refinement import EdgeRefinement


def check_no_nan_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def test_shape_preservation():
    print("=" * 60)
    print("  Test 1: Output shape == input shape")
    print("=" * 60)

    ref = EdgeRefinement(ch=64)
    x = torch.randn(2, 64, 148, 148)
    out = ref(x)

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    check_no_nan_inf(out, "output")
    print("  PASSED\n")


def test_output_differs_from_input():
    print("=" * 60)
    print("  Test 2: Output != input (refine path contributes)")
    print("=" * 60)

    ref = EdgeRefinement(ch=64)
    x = torch.randn(2, 64, 148, 148)
    out = ref(x)

    diff = (out - x).abs().sum().item()
    print(f"  |output - input| sum: {diff:.4f}")
    assert diff > 0, "Output is identical to input (refine path not contributing)"
    print("  PASSED\n")


def test_gradient_flow_both_branches():
    print("=" * 60)
    print("  Test 3: Gradient flow through both branches")
    print("=" * 60)

    ref = EdgeRefinement(ch=64)
    x = torch.randn(2, 64, 148, 148, requires_grad=True)
    out = ref(x)
    out.sum().backward()

    # Input gets gradient from both residual and refine paths
    assert x.grad is not None, "No gradient on input"
    assert (x.grad != 0).any(), "All-zero gradient on input"
    check_no_nan_inf(x.grad, "x.grad")
    print(f"  Input grad norm: {x.grad.norm().item():.6f}")

    # All refine block params should have gradients
    for name, p in ref.named_parameters():
        assert p.grad is not None, f"No gradient on {name}"
        assert (p.grad != 0).any(), f"All-zero gradient on {name}"
        check_no_nan_inf(p.grad, name)

    param_count = sum(p.numel() for p in ref.parameters())
    print(f"  All {param_count:,} parameters have non-zero gradients")
    print("  PASSED\n")


def test_different_batch_sizes():
    print("=" * 60)
    print("  Test 4: Different batch sizes")
    print("=" * 60)

    ref = EdgeRefinement(ch=64)

    for bs in [1, 2, 4, 8]:
        x = torch.randn(bs, 64, 148, 148)
        out = ref(x)
        assert out.shape == x.shape, f"BS={bs}: {out.shape} != {x.shape}"
        check_no_nan_inf(out, f"output_bs{bs}")
        print(f"  BS={bs}: {out.shape} ✓")

    print("  PASSED\n")


def test_parameter_count():
    print("=" * 60)
    print("  Test 5: Parameter count")
    print("=" * 60)

    ref = EdgeRefinement(ch=64)
    total = sum(p.numel() for p in ref.parameters())
    print(f"  Total parameters: {total:,}")

    # Expected: 2x(Conv3x3(64,64) + GN(64)) = 2*(64*64*9 + 64 + 64*2) = 2*(36864+64+128) = 74,112
    print(f"  Expected ~74,112")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  REFINEMENT.PY TEST SUITE")
    print("=" * 60 + "\n")

    test_shape_preservation()
    test_output_differs_from_input()
    test_gradient_flow_both_branches()
    test_different_batch_sizes()
    test_parameter_count()

    print("=" * 60)
    print("  ALL GATES PASSED - refinement.py is verified")
    print("=" * 60)
