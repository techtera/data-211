# Student Encoder + Object Mask Decoder

Object segmentation decoder trained on top of the distilled student encoder (255M params, 1536-dim).

## Architecture

```
Student Encoder (255M, frozen) -> SegFormer Decoder (4-level pyramid) -> Output (518x518, 2 classes)
```

- Trainable: ~15M params (decoder only)
- Frozen: 255M params (student encoder)
- Cached layers: `[3, 8, 13, 17]`
- Input dim: 1536, Output: 2 classes (background / object)

## Quick Start

```bash
# 1. Prepare data
mkdir -p data/images data/masks
# Images: data/images/*.png (518x518, RGB)
# Masks:  data/masks/*.png  (518x518, binary 0=bg, 255=object)

# 2. Verify student checkpoint exists
ls ../kd-encoder/checkpoints/student_final.pt

# 3. Train (multi-GPU recommended)
torchrun --nproc_per_node=2 train_ddp.py --epochs 100

# Single GPU fallback
python fine_tune.py
```

## Configuration

Edit `fine_tuning/config.py`:

| Parameter | Default |
|-----------|---------|
| BATCH_SIZE | 4 |
| NUM_EPOCHS | 100 |
| LEARNING_RATE | 1e-4 |

## Checkpoints

Saved to `checkpoints/`:
- `checkpoint_last.pt` - Latest
- `checkpoint_best.pt` - Best validation loss

## Evaluation

```bash
python evaluate_checkpoint.py
python inference.py              # Run inference on images
python infer_standalone.py       # Standalone inference (no training deps)
```

## Directory Structure

```
st-obj-mask/
├── obj_mask/              # Decoder architecture
│   ├── segformer_head.py  # SegFormer head (1536-dim, layers [3,8,13,17])
│   ├── segformer_decoder.py
│   ├── model.py
│   └── head_utils.py
├── fine_tuning/           # Training pipeline
│   ├── config.py, trainer.py, dataset.py
│   ├── dataloader.py, model_builder.py
│   ├── losses.py, metrics.py, optimizer.py
│   └── checkpoints.py, ddp_utils.py
├── train_ddp.py           # Multi-GPU training
├── fine_tune.py           # Single-GPU training
├── evaluate_checkpoint.py
├── inference.py
└── infer_standalone.py
```
