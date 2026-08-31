#!/usr/bin/env python3
"""
Quick evaluation of a saved edge mask checkpoint.

Usage:
    python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint_best.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.insert(0, "../kd-encoder")

from student import StudentAggregator
from edge_mask.model import StudentEdgeMask
from fine_tuning.dataset import EdgeMaskDataset
from fine_tuning.metrics import compute_complete_edge_metrics
from fine_tuning.config import DATASET_ROOT, VALIDATION_SPLIT, RANDOM_SEED


def evaluate_checkpoint(checkpoint_path, device='cuda'):
    print("="*60)
    print("EDGE MASK CHECKPOINT EVALUATION")
    print("="*60)
    print(f"\nCheckpoint: {checkpoint_path}")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print("\n[1/4] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', '?')
    print(f"  Epoch: {epoch}")
    print(f"  Saved Loss: {loss:.4f}" if isinstance(loss, float) else f"  Saved Loss: {loss}")

    # Load student encoder
    print("\n[2/4] Loading student encoder...")
    from fine_tuning.config import STUDENT_CHECKPOINT
    student_ckpt = torch.load(STUDENT_CHECKPOINT, map_location='cpu')
    state_dict = student_ckpt.get('student_state_dict', student_ckpt.get('model_state_dict', student_ckpt))

    student = StudentAggregator()
    student.load_state_dict(state_dict)
    student.eval()
    student.requires_grad_(False)
    print(f"  ✓ Student encoder loaded")

    # Build model and load checkpoint
    print("\n[3/4] Building model...")
    model = StudentEdgeMask(student).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded from checkpoint")

    # Load validation data
    print("\n[4/4] Loading validation data...")
    dataset = EdgeMaskDataset(Path(DATASET_ROOT))
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size

    torch.manual_seed(RANDOM_SEED)
    from torch.utils.data import random_split
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"  ✓ Validation set: {len(val_dataset)} images")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING...")
    print("="*60)

    # Warmup
    print("\n[1/2] Warmup (5 iterations)...")
    dummy_input = torch.randn(1, 1, 3, 518, 518).to(device)  # [B, S, C, H, W]
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    print("  ✓ Warmup complete")

    # Latency measurement
    print("\n[2/2] Measuring latency (100 iterations)...")
    import time
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)  # ms

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    throughput = 1000 / avg_latency  # images/sec

    print(f"  ✓ Latency measured: {avg_latency:.2f}ms avg")

    # Accuracy evaluation
    print("\nComputing metrics on validation set...")
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions (eval mode returns sigmoid output)
            logits = model(images)

            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx+1}/{len(val_loader)} batches...")

    # Compute metrics
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    # Convert sigmoid outputs back to logits for metric computation
    # logit = log(p / (1-p))
    eps = 1e-7
    all_logits = torch.clamp(all_logits, eps, 1-eps)
    all_logits = torch.log(all_logits / (1 - all_logits))

    metrics = compute_complete_edge_metrics(all_logits, all_targets)

    # Confusion matrix at 0.5 threshold
    preds = (torch.sigmoid(all_logits) > 0.5).float().view(-1)
    targets = all_targets.view(-1)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    # Per-class metrics
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    # Print results
    print("\n" + "="*60)
    print("EDGE DETECTION METRICS")
    print("="*60)
    print(f"Checkpoint Epoch    : {epoch}")
    print(f"Dice Score          : {metrics['dice_score']:.4f}")
    print(f"BF1 Precision       : {metrics['bf1_precision']:.4f}")
    print(f"BF1 Recall          : {metrics['bf1_recall']:.4f}")
    print(f"BF1 F1              : {metrics['bf1_f1']:.4f}")
    print(f"ODS Best Threshold  : {metrics['ods_threshold']:.2f}")
    print(f"ODS Best F1         : {metrics['ods_f1']:.4f}")

    print(f"\n" + "="*60)
    print("CONFUSION MATRIX (Threshold=0.5)")
    print("="*60)
    print(f"TP (Edge pixels correctly predicted)    : {tp:,}")
    print(f"FP (Non-edge predicted as edge)         : {fp:,}")
    print(f"FN (Edge pixels missed)                 : {fn:,}")
    print(f"TN (Non-edge correctly predicted)       : {tn:,}")
    print(f"\nPixel-wise Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1_score:.4f}")
    print(f"  IoU:       {iou:.4f}")

    print(f"\n" + "="*60)
    print("INFERENCE PERFORMANCE")
    print("="*60)
    print(f"Avg Latency         : {avg_latency:.2f} ms")
    print(f"Min Latency         : {min_latency:.2f} ms")
    print(f"Max Latency         : {max_latency:.2f} ms")
    print(f"Throughput          : {throughput:.2f} images/sec")
    print(f"Model Size          : {sum(p.numel() for p in model.parameters()):,} params")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params    : {trainable:,}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.device)
