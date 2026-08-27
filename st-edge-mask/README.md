# Student Encoder + Edge Mask Decoder Training

Edge mask prediction using the distilled student encoder (255M params, 1536-dim output).

## Quick Start

```bash
# 1. Prepare data
mkdir -p data/rgb data/masks
# Copy your images to data/rgb/ and masks to data/masks/

# 2. Verify student checkpoint exists
ls -lh ../kd-encoder/checkpoints_full/student_final.pt

# 3. Train
python fine_tune.py
```

## Data Format

- **RGB images**: `data/rgb/*.png` (518×518, [0,1] range)
- **Masks**: `data/masks/*_mask.png` (518×518, binary 0/255)

## Configuration

Edit `fine_tuning/config.py`:
- `BATCH_SIZE`: 4 (default)
- `NUM_EPOCHS`: 100
- `LEARNING_RATE`: 3e-4

## Output

Checkpoints saved to `checkpoints/`:
- `checkpoint_last.pt` - Latest
- `checkpoint_best.pt` - Best validation loss

## Architecture

Student (255M, 1536-dim) → Feature Projections → UNet++ → Edge Refinement → Output (518×518)

Trainable: ~10M params (decoder only)
Frozen: 255M params (student encoder)

See main vggt-KD/README_main.md for full pipeline details.
