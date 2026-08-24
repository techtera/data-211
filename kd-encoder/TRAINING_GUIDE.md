# Knowledge Distillation Training Guide

**Quick Start:** Train student encoder from teacher in ~10 hours on 2×A100 80GB.

## Overview

- **Teacher**: 909M param VGGT encoder (frozen, FP16)
- **Student**: 255M param encoder (DINOv2 pretrained, FP32)
- **Method**: Layer-wise feature distillation (4 cached layers)
- **Dataset**: 23,687 unrelated images
- **Hardware**: 2×A100 80GB GPUs

## Key Optimizations

### 1. Single Frame (CRITICAL)
```python
num_frames=1  # NOT 8!
```
**Why**: Images are unrelated (not video sequences). Using 8 frames was replicating each image 8× and wasting compute.
**Impact**: 10× speedup (40s/step → 4s/step)

### 2. Large Batch Size
```python
batch_size=32  # Use 60-75GB per GPU
```
**Why**: With single frames, memory usage is low. Increase batch to maximize GPU utilization.
**Impact**: 4× fewer steps per epoch (11 min/epoch vs 43 min)

### 3. Optimal Configuration

**Recommended (batch=32):**
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_full \
    --log_every 50
```

**Stats:**
- Effective batch: 32×2×2 = 128
- Steps per epoch: ~185
- Time per epoch: ~11 minutes
- **Total (35 epochs): ~6.5 hours**
- GPU memory: ~60GB per GPU (75% utilization)

**Aggressive (batch=40):**
- Change `--batch_size 40`
- GPU memory: ~75GB per GPU (95% utilization)
- Time per epoch: ~9 minutes
- **Total (35 epochs): ~5.2 hours**

## Training Workflow

### Step 1: Create Dataset Subset (Optional)

For faster testing, use half the dataset:
```bash
python create_subset.py \
    --input train_images \
    --output train_images_half \
    --ratio 0.5
```

### Step 2: Sanity Check (3 epochs)

Test setup with small run:
```bash
torchrun --nproc_per_node=2 sanity_check_ddp.py \
    --image_dir train_images \
    --batch_size 32 \
    --gradient_accumulation_steps 2
```

**Expected output:**
```
Step 10/185: Loss=1.7XX, Speed=0.28 steps/sec (3.5s/step)
```

**Check GPUs:**
```bash
nvidia-smi
# Should show: GPU 0 & 1 at ~60GB, 95-100% utilization
```

### Step 3: Full Training (35 epochs)

After sanity check passes:
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints \
    --log_every 50
```

### Step 4: Monitor Progress

**Terminal 1:** Training logs
- Loss should decrease smoothly
- Speed should stabilize around 3-4s/step

**Terminal 2:** GPU monitoring
```bash
watch -n 1 nvidia-smi
```

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| num_frames | 1 | Critical: images are unrelated |
| batch_size | 32-40 | Maximize GPU usage |
| gradient_accumulation | 2 | Effective batch = 128-160 |
| learning_rate | 1e-4 | Standard for distillation |
| warmup_epochs | 3 | LR warmup for stability |
| epochs | 30-35 | Pretrained student converges fast |
| num_workers | 12 | I/O prefetching |
| prefetch_factor | 4 | Preload batches |

## Expected Timeline

| Stage | Time |
|-------|------|
| Sanity check (3 epochs) | ~35 minutes |
| Full training (35 epochs, batch=32) | ~6.5 hours |
| **Total** | **~7 hours** |

## Checkpoints

Saved to `checkpoints/` directory:
- `checkpoint_last.pt` - Last epoch (for resuming)
- `checkpoint_best.pt` - Best validation loss
- `student_final.pt` - Final student model (use for inference)

## Troubleshooting

### OOM (Out of Memory)
- Reduce `batch_size` from 32 → 24 → 16
- Check `num_frames=1` (not 8!)

### Slow Training (>10s/step)
- Check `num_frames=1` (most common issue)
- Increase `num_workers` to 12-16
- Verify GPU utilization (should be 95-100%)

### Loss Not Decreasing
- Check learning rate (1e-4 is good baseline)
- Verify warmup (3 epochs recommended)
- Ensure teacher checkpoint loaded correctly

### DDP Not Using Both GPUs
- One GPU shows low memory/utilization
- Check `torchrun --nproc_per_node=2` (not `python`)
- Verify NCCL backend initialized

## Performance Comparison

| Config | Step Time | Epoch Time | 35 Epochs | GPU Memory |
|--------|-----------|------------|-----------|------------|
| **Original (8 frames, batch=4)** | 60s | 28 hours | **34 days** | 60GB |
| **Optimized (1 frame, batch=8)** | 3.5s | 43 min | 25 hours | 15GB |
| **Recommended (1 frame, batch=32)** | 3.5s | 11 min | **6.5 hours** | 60GB ✅ |

**100× speedup from original plan!**

## Model Architecture

**Teacher (909M):**
- 24 transformer layers
- Cached layers: [4, 11, 17, 23]
- Hidden dim: 2048
- Frozen during training

**Student (255M):**
- 18 transformer layers (DINOv2 pretrained)
- Cached layers: [3, 8, 13, 17]
- Hidden dim: 1536
- Gradient checkpointing for non-cached layers

**Distillation Loss:**
- Projection: 1536 → 2048 (learnable)
- MSE loss on projected features
- Cosine similarity regularization
- Per-layer alignment

## Resume Training

If training is interrupted:
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --resume_from checkpoints/checkpoint_last.pt \
    --checkpoint_dir checkpoints \
    --log_every 50
```

## Next Steps After Training

1. **Quantization**: Convert to INT8 for deployment
2. **Evaluation**: Test on validation set
3. **Deployment**: Export to TensorRT for Jetson Orin NX
4. **Fine-tuning**: Optional fine-tuning on task-specific data
