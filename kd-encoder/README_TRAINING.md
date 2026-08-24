# Phase 1: Distillation Training Guide

## 📋 Prerequisites

1. **Teacher checkpoint**: `../../vggt-unified/checkpoints/vggt_unified_fp16.pt`
2. **Image directory**: Folder with RGB images (see `DATA_REQUIREMENTS.md`)
3. **GPU**: Recommended (CPU is very slow)

## 🚀 Quick Start

### Step 1: Sanity Check (ALWAYS RUN FIRST!)

Quick 3-5 epoch test to verify everything works:

```bash
python sanity_check.py --image_dir /path/to/images
```

**Expected output:**
- Training starts and completes 5 epochs
- Loss decreases over epochs
- Checkpoints saved to `checkpoints_sanity/`

**If sanity check fails:** Fix issues before full training!

### Step 2: Full Training (50 epochs)

After sanity check passes:

```bash
python train.py --image_dir /path/to/images --epochs 50 --batch_size 4
```

**Training time:** ~8-12 hours on GPU (depends on dataset size)

## 📝 Script Options

### `sanity_check.py`

```bash
python sanity_check.py \
    --image_dir /path/to/images \
    --teacher_checkpoint ../../vggt-unified/checkpoints/vggt_unified_fp16.pt \
    --epochs 5 \
    --batch_size 2 \
    --device cuda \
    --checkpoint_dir checkpoints_sanity
```

### `train.py`

```bash
python train.py \
    --image_dir /path/to/images \
    --teacher_checkpoint ../../vggt-unified/checkpoints/vggt_unified_fp16.pt \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --warmup_epochs 5 \
    --device cuda \
    --checkpoint_dir checkpoints \
    --num_workers 4 \
    --log_every 10
```

## 🔄 Resuming Training

If training is interrupted:

```bash
python train.py --resume_from checkpoints/checkpoint_last.pt
```

This will:
- Load student weights
- Load optimizer state
- Load scheduler state
- Continue from last epoch

## 💾 Output Checkpoints

After training, you'll find:

```
checkpoints/
├── checkpoint_last.pt      # Most recent epoch (for resuming)
├── checkpoint_best.pt      # Best loss (USE THIS ONE!)
└── student_final.pt        # Student weights only
```

**Which checkpoint to use?**
- For decoder training: `checkpoint_best.pt` (best student encoder)
- For resuming: `checkpoint_last.pt`

## 📊 Monitoring Training

### Watch logs in real-time:

```bash
python train.py --image_dir /path/to/images 2>&1 | tee training.log
```

### What to monitor:

1. **Loss decreasing**: Should go from ~1.2 → 0.1-0.3
2. **Learning rate**: Warms up, then decays
3. **Best checkpoint updates**: Saved when loss improves

### Expected loss curve:

```
Epoch 1-10:   High loss (0.5-1.0)  - Student learning basics
Epoch 10-30:  Medium (0.1-0.3)     - Refining features
Epoch 30-50:  Low (<0.1)           - Fine-tuning
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python train.py --image_dir /path/to/images --batch_size 2

# Or use gradient accumulation (TODO: not implemented yet)
```

### Issue: Training too slow

```bash
# Use more workers
python train.py --image_dir /path/to/images --num_workers 8

# Use smaller dataset for testing
python sanity_check.py --image_dir /path/to/images --epochs 3
```

### Issue: Loss not decreasing

Possible causes:
1. Learning rate too high/low (try 5e-5 or 2e-4)
2. Dataset too small (need >1k images)
3. Bad teacher checkpoint (verify it loads)

## 🎯 Next Steps After Training

1. **Extract best student encoder:**
   ```bash
   # checkpoint_best.pt already contains student weights
   ```

2. **Train decoder heads:**
   - Edge-mask decoder (like `vggt-edge-mask`)
   - Object-mask decoder (like `vggt-obj-mask`)

3. **Compare performance:**
   - Original (909M) vs Distilled (255M)
   - Accuracy, latency, memory

## 📦 Full Example

Complete workflow:

```bash
# 1. Sanity check
python sanity_check.py --image_dir /data/images --epochs 3

# 2. If sanity check passes, run full training
python train.py --image_dir /data/images --epochs 50 --batch_size 4

# 3. If interrupted, resume
python train.py --resume_from checkpoints/checkpoint_last.pt

# 4. After training, verify checkpoints exist
ls -lh checkpoints/
# Should see: checkpoint_last.pt, checkpoint_best.pt, student_final.pt

# 5. Ready for Phase 2: decoder training!
```

## 🔧 Advanced Options

### Multi-GPU Training

Automatically uses all available GPUs:

```bash
python train.py --image_dir /path/to/images --use_multi_gpu
```

Effective batch size = `batch_size × num_gpus`

### Custom Learning Rate Schedule

```bash
python train.py \
    --image_dir /path/to/images \
    --learning_rate 2e-4 \
    --warmup_epochs 10
```

### Different Teacher Checkpoint

```bash
python train.py \
    --image_dir /path/to/images \
    --teacher_checkpoint /path/to/custom/teacher.pt
```

## 📈 Expected Results

After 50 epochs:

| Metric | Target |
|--------|--------|
| Final loss | <0.15 |
| Best loss | <0.10 |
| Student params | 255M |
| Speedup | ~2.5x vs teacher |

## ❓ FAQ

**Q: How long does training take?**  
A: ~8-12 hours for 50 epochs on modern GPU (V100/A100)

**Q: Can I train on CPU?**  
A: Yes, but very slow (~10x slower). Not recommended.

**Q: Do I need labeled data?**  
A: No! Distillation is unsupervised (see `DATA_REQUIREMENTS.md`)

**Q: Can I use a different teacher?**  
A: Yes, as long as it outputs features in the same format (4 layers, 2048 dim)

**Q: What if I run out of GPU memory?**  
A: Reduce `--batch_size` to 2 or 1
