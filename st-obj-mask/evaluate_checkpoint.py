#!/usr/bin/env python3
"""
Quick evaluation of a saved checkpoint.

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
from obj_mask.model import StudentObjMask
from fine_tuning.dataset import SegmentationDataset
from fine_tuning.metrics import compute_segmentation_metrics
from fine_tuning.config import DATASET_ROOT, VALIDATION_SPLIT, RANDOM_SEED


def evaluate_checkpoint(checkpoint_path, device='cuda'):
    print("="*60)
    print("OBJECT MASK CHECKPOINT EVALUATION")
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
    model = StudentObjMask(student).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded from checkpoint")

    # Load validation data
    print("\n[4/4] Loading validation data...")
    dataset = SegmentationDataset(Path(DATASET_ROOT))
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

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx+1}/{len(val_loader)} batches...")

    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    logits_dummy = torch.zeros(all_preds.shape[0], 2, all_preds.shape[1], all_preds.shape[2])
    logits_dummy.scatter_(1, all_preds.unsqueeze(1), 1)
    metrics = compute_segmentation_metrics(logits_dummy, all_targets, num_classes=2)

    # Confusion matrix
    tp = ((all_preds == 1) & (all_targets == 1)).sum().item()
    fp = ((all_preds == 1) & (all_targets == 0)).sum().item()
    fn = ((all_preds == 0) & (all_targets == 1)).sum().item()
    tn = ((all_preds == 0) & (all_targets == 0)).sum().item()

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Checkpoint Epoch    : {epoch}")
    print(f"Mean IoU            : {metrics['miou']:.4f}")
    print(f"Dice Score          : {metrics['mean_dice']:.4f}")
    print(f"Pixel Accuracy      : {metrics['pixel_accuracy']:.4f}")
    print(f"IoU Background      : {metrics['iou_background']:.4f}")
    print(f"IoU Object          : {metrics['iou_object']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  Background: TP={tn:,}, FP={fn:,}")
    print(f"  Object:     TP={tp:,}, FP={fp:,}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.device)
