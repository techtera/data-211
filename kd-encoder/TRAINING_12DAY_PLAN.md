# 12-Day Training Plan (Half Dataset)

## Goal
Complete distillation training in **max 12 days** using half dataset (~11,844 images)

## Time Budget
- Steps per epoch: ~846 (half of 1692)
- Time per epoch: ~8.25 hours (at 35 sec/step)
- 12 days = 288 hours
- **Max epochs: 35** (with safety buffer)
- **Recommended: 30-33 epochs**

## Setup (Run on VM)

### 1. Create Half Dataset (~5 min)
```bash
cd /home/terafacdata_gmail_com/dikshit/vggt-KD/kd-encoder
git pull origin vggt-KD

# Create subset with 50% of images
python create_subset.py \
    --input train_images \
    --output train_images_half \
    --ratio 0.5 \
    --seed 42

# Verify
ls train_images_half | wc -l
# Should show ~11,844 images
```

### 2. Run Sanity Check (1 day)
```bash
# Stop current training (Ctrl+C)

# Run 3-epoch sanity check on half dataset
torchrun --nproc_per_node=2 sanity_check_ddp.py \
    --image_dir train_images_half \
    --batch_size 7 \
    --gradient_accumulation_steps 2
```

**Expected:**
- Steps per epoch: ~846
- Time: ~8.25 hours/epoch × 3 = ~25 hours (~1 day)
- Loss should decrease smoothly

### 3. Full Training (11 days)
```bash
# After sanity check passes, run full training
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images_half \
    --epochs 33 \
    --batch_size 7 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_half \
    --log_every 50
```

**Expected:**
- Time per epoch: ~8.25 hours
- 33 epochs × 8.25 hours = **272 hours (~11.3 days)**
- Total with sanity: ~12.3 days ✅

## Configuration Details

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Dataset | ~11,844 images | 50% of full (23,687) |
| Epochs | 33 | Max that fits in 12 days |
| Batch/GPU | 7 | Memory optimized for 2×A100 |
| Grad Accum | 2 | Effective batch = 7×2×2 = 28 |
| Memory/GPU | ~54-60GB | Safe margin from 80GB |
| Steps/epoch | ~846 | Half of full dataset |
| Time/epoch | ~8.25 hours | At 35 sec/step |

## Alternative: Faster Training (9-10 days)

If you want to finish even faster, increase batch size:

```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images_half \
    --epochs 30 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --checkpoint_dir checkpoints_half \
    --log_every 50
```

**Changes:**
- Batch size: 7 → 8 (uses ~62GB/GPU, close to limit)
- Steps per epoch: 846 → 740
- Time per epoch: 8.25 hrs → 7.2 hrs
- Total: 30 epochs × 7.2 hrs = **216 hours (~9 days)**

## Monitoring

```bash
# Check GPU usage (other terminal)
watch -n 1 nvidia-smi

# Check training progress
tail -f nohup.out  # if running in background

# Expected output every 50 steps:
# Step 50/846: Loss=1.234, LR=0.000045, Speed=0.028 steps/sec (35.7s/step)
```

## After Training

**Option 1: Use directly**
- Student model ready at: `checkpoints_half/student_final.pt`
- Trained on 11,844 images × 33 epochs = ~390K samples

**Option 2: Fine-tune on full dataset (optional)**
```bash
# If model is good but you want more diversity, fine-tune:
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 10 \
    --batch_size 7 \
    --gradient_accumulation_steps 2 \
    --resume_from checkpoints_half/checkpoint_best.pt \
    --checkpoint_dir checkpoints_finetuned \
    --learning_rate 5e-5  # Lower LR for fine-tuning
```
- Time: 10 epochs × 16.5 hours = ~7 days
- Total: 12 + 7 = 19 days (still < 20 days target)

## Fallback: If Running Slow

If steps are slower than 35 sec (e.g., 40-45 sec):

**Plan B:**
- 30 epochs instead of 33
- Still fits in 12 days with buffer

**Plan C:**
- Keep dataset at half
- Reduce to 28 epochs
- Guaranteed to finish in 11 days

## Timeline Summary

| Stage | Duration | Total |
|-------|----------|-------|
| Create subset | 5 min | 5 min |
| Sanity check (3 epochs) | 25 hours | 1 day |
| Full training (33 epochs) | 272 hours | 11.3 days |
| **Total** | | **12.3 days** ✅ |

With buffer for slowdowns: Should finish in **11-13 days**
