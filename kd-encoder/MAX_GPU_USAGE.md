# Maximum GPU Usage Guide

## Current Bottleneck

With `num_frames=1`, memory usage is **very low** (~10-15GB per GPU).

**You have 80GB per GPU but only using 15-20%!** 😱

## Strategy: Increase Batch Size

With single frames, you can fit **MUCH larger batches** in memory.

## Recommended Configurations

### Option 1: Conservative (batch=32)
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
- Memory per GPU: ~60GB (safe)
- Effective batch: 32×2×2 = **128**
- Steps per epoch: 23,687 / 128 = **~185 steps**
- Time per epoch: 185 × 3.5s = **~11 minutes** 🚀
- 35 epochs: **~6.5 hours** 🚀🚀🚀

### Option 2: Aggressive (batch=40)
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 40 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_full \
    --log_every 50
```

**Stats:**
- Memory per GPU: ~75GB (close to limit)
- Effective batch: 40×2×2 = **160**
- Steps per epoch: 23,687 / 160 = **~148 steps**
- Time per epoch: 148 × 3.5s = **~9 minutes** 🚀
- 35 epochs: **~5.2 hours** 🚀🚀🚀

### Option 3: Maximum (batch=48, accum=1)
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 48 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_full \
    --log_every 50
```

**Stats:**
- Memory per GPU: ~78GB (max)
- Effective batch: 48×2×1 = **96**
- Steps per epoch: 23,687 / 96 = **~247 steps**
- Time per epoch: 247 × 3.5s = **~14 minutes**
- 35 epochs: **~8.2 hours**

## Comparison

| Config | Memory/GPU | Effective Batch | Steps/Epoch | Time/Epoch | 35 Epochs |
|--------|-----------|-----------------|-------------|------------|-----------|
| batch=8 (current) | 15GB | 32 | 740 | 43 min | 25 hours |
| **batch=32 (recommended)** | **60GB** | **128** | **185** | **11 min** | **6.5 hours** ✅ |
| batch=40 (aggressive) | 75GB | 160 | 148 | 9 min | 5.2 hours |
| batch=48 (max) | 78GB | 96 | 247 | 14 min | 8.2 hours |

## Why Larger Batch is Good Here

**For knowledge distillation:**
- ✅ Stable gradients (averaging over more samples)
- ✅ Faster convergence (fewer steps, more efficient updates)
- ✅ Better GPU utilization (less overhead)
- ✅ Teacher signal is fixed (not training teacher, so large batch OK)

**Caveat:**
- Very large batches (>256) can reduce generalization
- But 128-160 is sweet spot for distillation

## Recommended Approach

### Step 1: Test batch=32 (safe, 4× speedup)

After sanity check completes:

```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 5 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 1 \
    --checkpoint_dir checkpoints_test \
    --log_every 20
```

**Watch for:**
- GPU memory at step 10 (should be ~60GB)
- Loss converging normally
- No OOM errors

**Expected output:**
```
Step 10/185: Loss=X.XXX, Speed=0.28 steps/sec (3.5s/step)
```

**Check memory:**
```bash
nvidia-smi
# Should show: GPU 0: ~60GB, GPU 1: ~60GB
```

### Step 2: If successful, try batch=40 (5× speedup)

Stop test run, restart with batch=40:

```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 5 \
    --batch_size 40 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 1 \
    --checkpoint_dir checkpoints_test \
    --log_every 20
```

**Check memory again** - should be ~75GB

### Step 3: Use best config for full 35 epochs

Once you find the max stable batch size, run full training:

```bash
# Use whatever worked (32 or 40)
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 32 \  # or 40
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_full \
    --log_every 50
```

## Timeline with Full Dataset + batch=32

| Stage | Time |
|-------|------|
| Sanity check (half, batch=8) | ~2 hours (running now) |
| Test batch=32 (5 epochs) | ~1 hour |
| Full training (35 epochs, batch=32) | **~6.5 hours** |
| **Total** | **~9.5 hours** |

**You'll finish in less than 12 hours instead of 12 days!** 🚀

## Learning Rate Considerations

With larger batch size, you might want to scale learning rate:

**Rule of thumb:** LR scales with sqrt(batch_size) for stability

- batch=32, effective=128 → LR = 1e-4 ✅ (your current)
- batch=40, effective=160 → LR = 1.1e-4 or 1.2e-4
- batch=48, effective=96 → LR = 9e-5 or 1e-4

**Start with 1e-4, only adjust if loss is unstable**

## What to Monitor

```bash
# Terminal 1: Training
torchrun --nproc_per_node=2 train_ddp.py ...

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Look for:
# - GPU 0 & 1: ~60-75GB memory (not 15GB!)
# - GPU 0 & 1: 95-100% utilization
# - Loss decreasing smoothly
```

## Expected Results

**With batch=32:**
- **Training time: ~6.5 hours** (was 2 days!)
- Memory efficiency: 75% usage (was 20%)
- Effective batch: 128 (was 32)
- **100× faster than original plan with num_frames=8!**
