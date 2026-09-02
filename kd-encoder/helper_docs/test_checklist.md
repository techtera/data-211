# Test Checklist

## Sanity Check

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| DDP sanity check | Verify training pipeline works end-to-end | `torchrun --nproc_per_node=2 sanity_check_ddp.py --image_dir train_images --batch_size 64` | Completes 3-5 epochs without error | — | — | — |
| Feature evaluation | Assess student quality vs teacher | `python evaluate_features.py --student checkpoints_full/checkpoint_epoch_36.pt --teacher ../../vggt-unified/checkpoints/vggt_unified_fp16.pt --images "test_images/*.jpg"` | Correlation >0.75, Variance Ratio >0.7 | Score 1.49 | PASS | Epoch 36 |

## Training Validation

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| Full training | Train student for 80 epochs | `torchrun --nproc_per_node=2 train_ddp.py --image_dir train_images --epochs 80 --batch_size 64` | Loss decreases, checkpoints saved | — | IN PROGRESS | — |
| Checkpoint resume | Verify training can resume from checkpoint | `torchrun ... --resume_from checkpoints/checkpoint_last.pt` | Training continues from saved epoch | — | — | — |

## Pre-Decoder Checks

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| Final evaluation | Evaluate epoch 80 checkpoint | `python evaluate_features.py --student checkpoints_full/student_final.pt ...` | Correlation >0.75 | — | PENDING | — |
| Feature shape check | Verify student output matches expected shape | — | [B, S, 1374, 1536] at cached layers | — | PENDING | — |
