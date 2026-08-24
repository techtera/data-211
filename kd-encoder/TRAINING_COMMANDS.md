# Training Commands for 2×A100 80GB

## 🚀 Maximum GPU Utilization (~60GB per GPU)

### Option 1: Balanced (Recommended)
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 50 \
    --batch_size 7 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 5 \
    --log_every 100
```
**Stats:**
- Memory per GPU: ~54GB
- Effective batch: 7×2×2 = **28**
- Training time: ~4-5 hours
- Safe margin from OOM

### Option 2: Maximum (Aggressive)
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 50 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 5 \
    --log_every 100
```
**Stats:**
- Memory per GPU: ~62GB
- Effective batch: 8×2×2 = **32**
- Training time: ~4 hours
- Close to limit (may OOM on some steps)

### Option 3: Conservative
```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 50 \
    --batch_size 6 \
    --gradient_accumulation_steps 3 \
    --learning_rate 1e-4 \
    --warmup_epochs 5 \
    --log_every 100
```
**Stats:**
- Memory per GPU: ~46GB
- Effective batch: 6×2×3 = **36** (largest!)
- Training time: ~4.5 hours
- Very safe from OOM

## 📊 Comparison

| Config | Memory/GPU | Effective Batch | Speed | Risk |
|--------|-----------|-----------------|-------|------|
| batch=7, accum=2 | 54GB | 28 | Fast | Low ✅ |
| batch=8, accum=2 | 62GB | 32 | Faster | Medium |
| batch=6, accum=3 | 46GB | 36 | Medium | Very Low |

## 🔍 Monitoring

```bash
# In another terminal
watch -n 1 nvidia-smi

# You should see:
# GPU 0: ~55-60GB, 100% util
# GPU 1: ~55-60GB, 100% util  ← Both working!
```

## ⚠️ Current Issue

Your current training only uses GPU 0!
- Stop it: Ctrl+C
- Run DDP command above
- Verify both GPUs are utilized

## 📈 Expected Output

```
DDP Training: 2 GPUs
Batch size per GPU: 7
Gradient accumulation: 2
Effective batch size: 28 (7×2×2)
```

Both GPUs should show ~55-60GB usage.
