# Student Encoder + Object Mask Decoder Training

Object segmentation using the distilled student encoder (255M params, 1536-dim output).

## ⚠️ IMPORTANT: Before Training

Edit `obj_mask/segformer_head.py` line ~63 to update defaults:

```python
# Change:
def __init__(self, dim_in: int, patch_size: int = 14, output_dim: int = 4, ...

# To:
def __init__(self, dim_in: int = 1536, patch_size: int = 14, output_dim: int = 2, ...
```

And line ~69:
```python
# Change:
intermediate_layer_idx: List[int] = [4, 11, 17, 23],

# To:
intermediate_layer_idx: List[int] = [3, 8, 13, 17],
```

**✅ ALREADY DONE!**

## Quick Start

```bash
# 1. Prepare data
mkdir -p data/images data/masks
# Copy your images to data/images/ and masks to data/masks/

# 2. Verify student checkpoint
ls -lh ../kd-encoder/checkpoints_full/student_final.pt

# 3. Train

# Multi-GPU (DDP) - RECOMMENDED
torchrun --nproc_per_node=2 train_ddp.py --epochs 100

# Single GPU (fallback)
python fine_tune.py
```

## Data Format

- **Images**: `data/images/*.png` (518×518, RGB)
- **Masks**: `data/masks/*.png` (518×518, binary 0=bg, 255=object)

## Configuration

Edit `fine_tuning/config.py`:
- `BATCH_SIZE`: 4 (default)
- `NUM_EPOCHS`: 100
- `LEARNING_RATE`: 1e-4

## Output

Checkpoints saved to `checkpoints/`:
- `checkpoint_last.pt` - Latest
- `checkpoint_best.pt` - Best validation loss

## Architecture

Student (255M, 1536-dim) → SegFormer Decoder (4-level pyramid) → Output (518×518, 2 classes)

Trainable: ~15M params (decoder only)
Frozen: 255M params (student encoder)

See main vggt-KD/README_main.md for full pipeline details.
