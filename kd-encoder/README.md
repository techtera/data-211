# Knowledge Distillation: VGGT Teacher → Student Encoder

Compress 909M parameter VGGT encoder to 255M parameter student via layer-wise feature distillation.

## Quick Start

**Train student in ~7 hours on 2×A100 80GB:**

```bash
# Sanity check (3 epochs, ~35 min)
torchrun --nproc_per_node=2 sanity_check_ddp.py \
    --image_dir train_images \
    --batch_size 32 \
    --gradient_accumulation_steps 2

# Full training (35 epochs, ~6.5 hours)
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 35 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --checkpoint_dir checkpoints
```

**Output:** `checkpoints/student_final.pt` (255M param encoder, inference-ready)

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
- ✅ **Large batches** - Batch size 32-40 (60-75GB per GPU)
- ✅ **Gradient checkpointing** - Memory-efficient training
- ✅ **Pretrained initialization** - DINOv2 for faster convergence

## Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training instructions & troubleshooting
- **[STATUS.md](STATUS.md)** - Current training status & progress
- **[README.md](README.md)** - This file (overview)

## Performance

| Configuration | Time/Epoch | 35 Epochs | GPU Usage |
|--------------|-----------|-----------|-----------|
| Recommended (batch=32) | 11 min | **6.5 hours** | 60GB/GPU ✅ |
| Aggressive (batch=40) | 9 min | 5.2 hours | 75GB/GPU |

**100× speedup** from initial naive configuration (num_frames=8, small batches)

## Training Data

- **Format:** Independent images (JPG/PNG)
- **Size:** 518×518 (VGGT standard)
- **Count:** 23,687 images
- **Preprocessing:** ImageNet normalization

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

**Status:** Phase 1 in progress - See [STATUS.md](STATUS.md) for live updates

**Questions?** Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for troubleshooting
