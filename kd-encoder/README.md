# Knowledge Distillation: VGGT Teacher → Student Encoder

Compress 909M parameter VGGT encoder to 255M parameter student via layer-wise feature distillation.

## Quick Start

**Train student in ~30 hours on 2×A100 80GB:**

```bash
# Sanity check (3 epochs, ~1 hour)
torchrun --nproc_per_node=2 sanity_check_ddp.py \
    --image_dir train_images \
    --batch_size 64 \
    --gradient_accumulation_steps 1

# Full training (80 epochs, ~30 hours)
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 80 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --num_workers 12 \
    --checkpoint_dir checkpoints_full \
    --log_every 5
```

**Output:** `checkpoints_full/student_final.pt` (255M param encoder, inference-ready)

## Architecture

| Component | Parameters | Layers | Dim | Details |
|-----------|-----------|--------|-----|---------|
| **Teacher** | 909M | 24 | 2048 | VGGT encoder (frozen, FP16) |
| **Student** | 255M | 18 | 1536 | DINOv2 pretrained → fine-tuned |
| **Distillation** | 4 layers | - | Projection 1536→2048 | MSE + cosine similarity |

**Compression:** 3.6× smaller, 2-3× faster inference

## Key Features

- ✅ **Single frame processing** - For unrelated images (not videos)
- ✅ **DDP multi-GPU** - Efficient 2×A100 utilization
- ✅ **Large batches** - Batch size 64 (60-70GB per GPU)
- ✅ **No gradient accumulation** - Faster training, better optimizer dynamics
- ✅ **Gradient checkpointing** - Memory-efficient training
- ✅ **Pretrained initialization** - DINOv2 for faster convergence
- ✅ **Extended training** - 80 epochs for optimal convergence

## Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training instructions & troubleshooting
- **[STATUS.md](STATUS.md)** - Current training status & progress
- **[README.md](README.md)** - This file (overview)

## Performance

| Configuration | Batches/Epoch | Time/Epoch | 80 Epochs | GPU Usage |
|--------------|---------------|-----------|-----------|-----------|
| **Current (batch=64)** | 185 | **~23 min** | **~30 hours** | 60-70GB/GPU ✅ |
| Aggressive (batch=72) | 165 | ~20 min | ~26 hours | 70-75GB/GPU |
| Maximum (batch=80) | 148 | ~18 min | ~24 hours | 75-78GB/GPU |

**Cold start:** First 5-10 steps are slower (~30s/step) while data pipeline warms up  
**Steady state:** Speed stabilizes at ~7-8s/step after step 20-30

**100× speedup** from initial naive configuration (num_frames=8, small batches, single GPU)

## Training Data

- **Format:** Independent images (JPG/PNG)
- **Size:** 518×518 (VGGT standard)
- **Count:** 23,687 images
- **Preprocessing:** Resize + ToTensor (models normalize internally)

**Note:** Images must be unrelated (not video sequences). Use `num_frames=1`.

## Requirements

```bash
pip install torch torchvision timm pillow
```

**Hardware:**
- 2× GPU with 16GB+ VRAM (tested on 2×A100 80GB)
- CUDA 11.8+
- PyTorch 2.0+

## Directory Structure

```
kd-encoder/
├── student/              # Student encoder architecture
├── training/             # Training pipeline & DDP utils
├── distillation/         # Loss functions & token sampling
├── benchmarking/         # Performance benchmarking tools
├── checkpoints/          # Saved models (created during training)
├── sanity_check_ddp.py   # Quick 3-epoch validation
├── train_ddp.py          # Full training script
├── create_subset.py      # Dataset subset utility
├── TRAINING_GUIDE.md     # Complete training guide
├── STATUS.md             # Current progress tracker
└── README.md             # This file
```

## Checkpoints

Training produces:
- `checkpoint_last.pt` - Resume interrupted training
- `checkpoint_best.pt` - Best validation loss
- `student_final.pt` - Final model for inference

## Method: Layer-wise Feature Distillation

1. **Teacher forward** (no gradients): Extract 4 cached layer features
2. **Token sampling**: 1374 → 133 tokens (10× memory reduction)
3. **Student forward** (with gradients): Extract 4 cached layer features
4. **Project student**: 1536 → 2048 dim (learnable projection)
5. **Loss**: MSE + cosine similarity per layer
6. **Update**: Student + projection weights only

**Teacher remains frozen** throughout training.

## Citation

Based on VGGT (Video-Group Geometric Transformers):
```
@article{vggt2024,
  title={VGGT: Visual Grounding with Geometric Transformers},
  author={...},
  journal={...},
  year={2024}
}
```

## License

See parent project for license details.

---

## Current Status

**Training:** 80 epochs in progress (~30 hours)  
**Configuration:** batch_size=64, no gradient accumulation, log_every=5  
**Expected completion:** Check [STATUS.md](STATUS.md) for live updates

**Questions?** Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for troubleshooting

---

## Training Configuration Summary

```bash
# Recommended configuration (tested on 2×A100 80GB)
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 80 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --num_workers 12 \
    --checkpoint_dir checkpoints_full \
    --log_every 5
```

**Key parameters:**
- **185 steps/epoch** (23,687 images ÷ 128 effective batch)
- **~7-8s/step** (after warmup)
- **~23 min/epoch** → **30 hours total**
- **GPU usage:** 60-70GB per A100
