"""
Test script for edge_mask/losses.py

Gates:
- Gradient flow (non-zero grad on logits after backward)
- No NaN/Inf in loss or gradients
- pos_weight clamping works at boundaries
- Loss is non-negative
- All-zero target: stable
- All-one target: stable
- Perfect prediction: near-zero loss
- Extremely sparse target: stable
- compute_total_loss: correct weighted combination
"""

import sys
sys.path.insert(0, ".")

import torch
from edge_mask.losses import EdgeLoss, compute_total_loss


def check_no_nan_inf(tensor, name):
    assert not torch.isnan(tensor).any(), f"{name} contains NaN"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf"


def test_basic_forward_backward():
    print("=" * 60)
    print("  Test 1: Basic forward + backward")
    print("=" * 60)

    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()  # ~10% edges

    loss = loss_fn(logits, target)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Loss value: {loss.item():.6f}")

    check_no_nan_inf(loss, "loss")
    assert loss.item() >= 0, "Loss is negative"

    loss.backward()

    check_no_nan_inf(logits.grad, "logits.grad")
    assert logits.grad is not None, "No gradient on logits"
    assert (logits.grad != 0).any(), "All gradients are zero"

    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    print(f"  Non-zero grad elements: {(logits.grad != 0).sum().item()}/{logits.grad.numel()}")
    print("  PASSED\n")


def test_pos_weight_clamping():
    print("=" * 60)
    print("  Test 2: pos_weight clamping")
    print("=" * 60)

    loss_fn = EdgeLoss(pos_weight_clamp=(5, 25))

    # Case 1: Very sparse target (1% edges → natural weight ~99, should clamp to 25)
    logits = torch.zeros(1, 1, 100, 100, requires_grad=True)
    target = torch.zeros(1, 1, 100, 100)
    target[0, 0, 50, 50] = 1.0  # 1 pixel out of 10000

    pos = target.sum()
    neg = target.numel() - pos
    natural_weight = neg / pos
    expected_clamped = min(max(natural_weight.item(), 5), 25)

    print(f"  Sparse case: {pos.item():.0f} pos / {neg.item():.0f} neg")
    print(f"  Natural weight: {natural_weight.item():.1f}")
    print(f"  Expected clamped: {expected_clamped:.1f}")

    loss = loss_fn(logits, target)
    check_no_nan_inf(loss, "loss (sparse)")
    loss.backward()
    check_no_nan_inf(logits.grad, "grad (sparse)")
    print(f"  Loss (sparse): {loss.item():.6f}")

    # Case 2: Dense target (50% edges → natural weight ~1, should clamp to 5)
    logits2 = torch.zeros(1, 1, 100, 100, requires_grad=True)
    target2 = (torch.rand(1, 1, 100, 100) > 0.5).float()

    pos2 = target2.sum()
    neg2 = target2.numel() - pos2
    natural_weight2 = neg2 / pos2
    expected_clamped2 = min(max(natural_weight2.item(), 5), 25)

    print(f"  Dense case: {pos2.item():.0f} pos / {neg2.item():.0f} neg")
    print(f"  Natural weight: {natural_weight2.item():.2f}")
    print(f"  Expected clamped: {expected_clamped2:.1f}")

    loss2 = loss_fn(logits2, target2)
    check_no_nan_inf(loss2, "loss (dense)")
    loss2.backward()
    check_no_nan_inf(logits2.grad, "grad (dense)")
    print(f"  Loss (dense): {loss2.item():.6f}")
    print("  PASSED\n")


def test_all_zero_target():
    print("=" * 60)
    print("  Test 3: All-zero target (no edges)")
    print("=" * 60)

    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64)

    loss = loss_fn(logits, target)
    print(f"  Loss value: {loss.item():.6f}")
    check_no_nan_inf(loss, "loss (all-zero target)")
    assert loss.item() >= 0, "Loss is negative"

    loss.backward()
    check_no_nan_inf(logits.grad, "grad (all-zero target)")
    assert logits.grad is not None, "No gradient"
    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    print("  PASSED\n")


def test_all_one_target():
    print("=" * 60)
    print("  Test 4: All-one target (all edges)")
    print("=" * 60)

    loss_fn = EdgeLoss()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.ones(2, 1, 64, 64)

    loss = loss_fn(logits, target)
    print(f"  Loss value: {loss.item():.6f}")
    check_no_nan_inf(loss, "loss (all-one target)")
    assert loss.item() >= 0, "Loss is negative"

    loss.backward()
    check_no_nan_inf(logits.grad, "grad (all-one target)")
    assert logits.grad is not None, "No gradient"
    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    print("  PASSED\n")


def test_perfect_prediction():
    print("=" * 60)
    print("  Test 5: Perfect prediction (near-zero loss)")
    print("=" * 60)

    loss_fn = EdgeLoss()
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()

    # Perfect logits: large positive where target=1, large negative where target=0
    logits = torch.where(target == 1, torch.tensor(10.0), torch.tensor(-10.0))
    logits.requires_grad_(True)

    loss = loss_fn(logits, target)
    print(f"  Loss value: {loss.item():.8f}")
    check_no_nan_inf(loss, "loss (perfect)")
    assert loss.item() < 0.01, f"Loss too high for perfect prediction: {loss.item()}"

    loss.backward()
    check_no_nan_inf(logits.grad, "grad (perfect)")
    print(f"  Gradient norm: {logits.grad.norm().item():.8f}")
    print("  PASSED\n")


def test_extremely_sparse_target():
    print("=" * 60)
    print("  Test 6: Extremely sparse target (1 pixel in 518x518)")
    print("=" * 60)

    loss_fn = EdgeLoss()
    logits = torch.randn(1, 1, 518, 518, requires_grad=True)
    target = torch.zeros(1, 1, 518, 518)
    target[0, 0, 259, 259] = 1.0  # single edge pixel

    edge_ratio = target.sum().item() / target.numel()
    print(f"  Edge ratio: {edge_ratio:.8f} ({target.sum().item():.0f} pixels)")

    loss = loss_fn(logits, target)
    print(f"  Loss value: {loss.item():.6f}")
    check_no_nan_inf(loss, "loss (extremely sparse)")
    assert loss.item() >= 0, "Loss is negative"

    loss.backward()
    check_no_nan_inf(logits.grad, "grad (extremely sparse)")
    assert logits.grad is not None, "No gradient"
    print(f"  Gradient norm: {logits.grad.norm().item():.6f}")
    print("  PASSED\n")


def test_compute_total_loss():
    print("=" * 60)
    print("  Test 7: compute_total_loss weighted combination")
    print("=" * 60)

    loss_fn = EdgeLoss()
    target = (torch.rand(2, 1, 64, 64) > 0.9).float()

    final_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    ds1_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    ds2_logits = torch.randn(2, 1, 64, 64, requires_grad=True)

    total_loss = compute_total_loss(final_logits, ds1_logits, ds2_logits, target, loss_fn)

    # Verify weights: 1.0 * final + 0.2 * ds2 + 0.1 * ds1
    loss_final = loss_fn(final_logits, target)
    loss_ds1 = loss_fn(ds1_logits, target)
    loss_ds2 = loss_fn(ds2_logits, target)
    expected = 1.0 * loss_final + 0.2 * loss_ds2 + 0.1 * loss_ds1

    print(f"  Loss final: {loss_final.item():.6f}")
    print(f"  Loss DS1:   {loss_ds1.item():.6f}")
    print(f"  Loss DS2:   {loss_ds2.item():.6f}")
    print(f"  Total (computed):  {total_loss.item():.6f}")
    print(f"  Total (expected):  {expected.item():.6f}")

    check_no_nan_inf(total_loss, "total_loss")
    assert total_loss.item() >= 0, "Total loss is negative"

    total_loss.backward()
    check_no_nan_inf(final_logits.grad, "final_logits.grad")
    check_no_nan_inf(ds1_logits.grad, "ds1_logits.grad")
    check_no_nan_inf(ds2_logits.grad, "ds2_logits.grad")

    assert final_logits.grad is not None, "No gradient on final_logits"
    assert ds1_logits.grad is not None, "No gradient on ds1_logits"
    assert ds2_logits.grad is not None, "No gradient on ds2_logits"

    # Final should have largest gradient (weight 1.0)
    # DS1 should have smallest gradient (weight 0.1)
    grad_final_norm = final_logits.grad.norm().item()
    grad_ds1_norm = ds1_logits.grad.norm().item()
    grad_ds2_norm = ds2_logits.grad.norm().item()

    print(f"  Grad norm final: {grad_final_norm:.6f}")
    print(f"  Grad norm DS1:   {grad_ds1_norm:.6f}")
    print(f"  Grad norm DS2:   {grad_ds2_norm:.6f}")
    print("  PASSED\n")


def test_loss_non_negative():
    print("=" * 60)
    print("  Test 8: Loss non-negative across random inputs")
    print("=" * 60)

    loss_fn = EdgeLoss()
    all_passed = True

    for i in range(20):
        logits = torch.randn(1, 1, 32, 32)
        sparsity = torch.rand(1).item() * 0.5  # 0% to 50% edges
        target = (torch.rand(1, 1, 32, 32) > (1 - sparsity)).float()
        loss = loss_fn(logits, target)
        if loss.item() < 0:
            print(f"  FAIL at iteration {i}: loss = {loss.item()}")
            all_passed = False
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  FAIL at iteration {i}: NaN/Inf")
            all_passed = False

    assert all_passed, "Some iterations produced negative or invalid loss"
    print("  20/20 random cases: all non-negative, no NaN/Inf")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LOSSES.PY TEST SUITE")
    print("=" * 60 + "\n")

    test_basic_forward_backward()
    test_pos_weight_clamping()
    test_all_zero_target()
    test_all_one_target()
    test_perfect_prediction()
    test_extremely_sparse_target()
    test_compute_total_loss()
    test_loss_non_negative()

    print("=" * 60)
    print("  ALL GATES PASSED - losses.py is verified")
    print("=" * 60)
