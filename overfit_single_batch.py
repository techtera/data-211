"""
Single batch overfit test for VGGT + UNet++ Edge Mask model.

This script verifies the model can memorize a single batch
by training on it for 500 steps. It checks:

    1. Loss decreases toward 0
    2. Encoder (aggregator) stays frozen
    3. Decoder weights update

Usage:
    python overfit_single_batch.py
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from fine_tuning.config import (
    DEVICE,
    GRAD_CLIP_MAX_NORM,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from fine_tuning.model_builder import build_model
from fine_tuning.dataloader import build_dataloaders
from fine_tuning.losses import build_loss, compute_total_loss
from fine_tuning.evaluate import dice_score, boundary_f1, confusion_matrix


# ============================================================
# Config
# ============================================================

OVERFIT_STEPS = 500

LOG_EVERY = 25


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("Single Batch Overfit Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Data (grab one batch)
    # --------------------------------------------------------

    train_loader, _ = build_dataloaders()

    images, masks = next(iter(train_loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    print(f"\nBatch Shape (images) : {images.shape}")
    print(f"Batch Shape (masks)  : {masks.shape}")
    print(f"Edge Pixel Ratio     : {masks.mean().item():.4f}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = build_loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # --------------------------------------------------------
    # Snapshot Encoder Weights
    # --------------------------------------------------------

    encoder_snapshot = {
        name: p.clone()
        for name, p in model.feature_extractor.aggregator.named_parameters()
    }

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"Training for {OVERFIT_STEPS} steps on single batch")
    print("=" * 60)

    model.train()
    model.feature_extractor.aggregator.eval()

    losses = []

    for step in range(OVERFIT_STEPS):

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):

            logits, ds1, ds2 = model(images)

            loss = compute_total_loss(
                logits, ds1, ds2, masks, criterion,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_MAX_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        if step % LOG_EVERY == 0 or step == OVERFIT_STEPS - 1:

            preds = torch.sigmoid(logits)
            edge_ratio = (preds > 0.5).float().mean().item()

            print(
                f"  Step [{step + 1:3d}/{OVERFIT_STEPS}]  "
                f"Loss: {loss.item():.6f}  "
                f"Grad: {grad_norm.item():.2f}  "
                f"Edge Ratio: {edge_ratio:.4f}"
            )

    # ============================================================
    # Results
    # ============================================================

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Loss Decrease
    # --------------------------------------------------------

    initial_loss = losses[0]
    final_loss = losses[-1]
    decrease_pct = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial Loss       : {initial_loss:.6f}")
    print(f"Final Loss         : {final_loss:.6f}")
    print(f"Decrease           : {decrease_pct:.1f}%")

    if final_loss < 0.1:
        print("PASS : Loss converged near 0")
    elif final_loss < initial_loss:
        print("PARTIAL : Loss decreased but did not converge to 0")
    else:
        print("FAIL : Loss did not decrease")

    # --------------------------------------------------------
    # 2. Encoder Frozen
    # --------------------------------------------------------

    encoder_changed = False

    for name, p in model.feature_extractor.aggregator.named_parameters():
        if not torch.equal(p, encoder_snapshot[name]):
            encoder_changed = True
            print(f"FAIL : Encoder param changed: {name}")
            break

    if not encoder_changed:
        print("PASS : Encoder stayed frozen")

    # --------------------------------------------------------
    # 3. Decoder Updated
    # --------------------------------------------------------

    decoder_has_grad = any(
        p.grad is not None
        for p in model.decoder.parameters()
    )

    if decoder_has_grad:
        print("PASS : Decoder received gradients")
    else:
        print("FAIL : Decoder has no gradients")

    # --------------------------------------------------------
    # 4. Evaluation Metrics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Evaluation Metrics (on overfitted batch)")
    print("=" * 60)

    model.eval()

    with torch.no_grad():
        probs = model(images)

    # [B, S, 1, H, W] -> [B*S, 1, H, W]
    B, S = probs.shape[:2]
    probs_flat = probs.view(B * S, 1, probs.shape[3], probs.shape[4])
    masks_flat = masks.view(B * S, 1, masks.shape[3], masks.shape[4])

    pred_binary = (probs_flat > 0.5).float()
    target_binary = masks_flat.float()

    # Dice
    dice = dice_score(pred_binary, target_binary)
    print(f"\nDice Score         : {dice:.4f}")

    # BF1
    bf1 = boundary_f1(pred_binary, target_binary, tolerance=2)
    print(f"BF1 Precision      : {bf1['precision']:.4f}")
    print(f"BF1 Recall         : {bf1['recall']:.4f}")
    print(f"BF1 F1             : {bf1['f1']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(pred_binary, target_binary)
    print(f"Confusion Matrix   : TP={cm['tp']:,}  FP={cm['fp']:,}  FN={cm['fn']:,}  TN={cm['tn']:,}")

    # --------------------------------------------------------
    # Final Verdict
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    all_pass = (
        final_loss < initial_loss
        and not encoder_changed
        and decoder_has_grad
    )

    if all_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")

    if dice > 0.9:
        print("OVERFIT CHECK: PASS (Dice > 0.9)")
    else:
        print(f"OVERFIT CHECK: PARTIAL (Dice = {dice:.4f}, expected > 0.9)")

    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
