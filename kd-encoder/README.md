# Knowledge Distillation: VGGT Teacher -> Student Encoder

Compress the 909M parameter VGGT encoder to a 255M parameter student via layer-wise feature distillation.

## Architecture

| Component | Params | Layers | Dim  |
|-----------|--------|--------|------|
| Teacher   | 909M   | 24     | 2048 |
| Student   | 255M   | 18     | 1536 |

Distillation: 4 cached layers with learnable 1536->2048 projection, MSE + cosine similarity loss.

## Quick Start

```bash
# Full training (80 epochs, ~30 hours on 2x A100)
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 80 --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --warmup_epochs 3 \
    --num_workers 12 \
    --checkpoint_dir checkpoints_full --log_every 5
```

## Training Data

- Independent images (not video sequences), JPG/PNG
- Resized to 518x518
- 23,687 images in the training set
- Uses `num_frames=1` (single frame processing)

## Performance

| Config | Steps/Epoch | Time/Epoch | Total (80 ep) | GPU Mem |
|--------|-------------|------------|----------------|---------|
| batch=64 (default) | 185 | ~23 min | ~30 hours | 60-70 GB |
| batch=72 | 165 | ~20 min | ~26 hours | 70-75 GB |
| batch=80 | 148 | ~18 min | ~24 hours | 75-78 GB |

Steady state: ~7-8s/step (first 5-10 steps are slower during data pipeline warmup).

## Checkpoints

- `checkpoint_last.pt` - Resume interrupted training
- `checkpoint_best.pt` - Best validation loss
- `student_final.pt` - Final model for downstream use

## Method

1. Teacher forward (frozen, FP16): extract features at cached layers `[4, 11, 17, 23]`
2. Token sampling: 1374 -> 133 tokens (10x memory reduction)
3. Student forward: extract features at cached layers `[3, 8, 13, 17]`
4. Project student features: 1536 -> 2048 dim (learnable projection)
5. Loss: MSE + cosine similarity per layer
6. Update: student + projection weights only

## Evaluation

```bash
python evaluate_features.py
```

Compares student features against teacher to assess distillation quality before decoder training.

## Directory Structure

```
kd-encoder/
├── student/           # Student encoder architecture
│   ├── aggregator.py
│   ├── initialization.py
│   └── layers/
├── training/          # Training pipeline & DDP utilities
│   ├── config.py, trainer.py, dataset.py
│   ├── ddp_utils.py, optimizer.py, scheduler.py
│   └── checkpoints.py
├── distillation/      # Loss functions & token sampling
│   ├── loss.py, projection.py
│   └── token_sampling.py
├── train_ddp.py       # Full training script (DDP via torchrun)
├── evaluate_features.py  # Feature quality evaluation vs teacher
├── load_real_teacher.py  # Load VGGT teacher from unified checkpoint
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Hardware: 2x GPU with 16GB+ VRAM, CUDA 11.8+, PyTorch 2.0+
