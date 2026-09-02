# VGGT Knowledge Distillation Pipeline

Distill VGGT-1B (909M) into a compact student encoder (255M) and train downstream decoders for edge and object segmentation. Designed for deployment on Jetson Orin NX 16GB with <1s latency.

## Directory Structure

```
vggt-KD/
├── kd-encoder/       # Knowledge distillation: teacher (909M) -> student (255M)
├── st-edge-mask/     # Edge mask decoder training (~10M params)
├── st-obj-mask/      # Object mask decoder training (~15M params)
├── fastVGGT/         # Token-merging acceleration (3x faster, no retraining)
└── test_encoder_changes.py
```

## Architecture

| Component | Params | Layers | Dim  | Details |
|-----------|--------|--------|------|---------|
| Teacher   | 909M   | 24     | 2048 | VGGT encoder (frozen) |
| Student   | 255M   | 18     | 1536 | DINOv2 pretrained, distilled |
| Edge Dec  | ~10M   | -      | -    | UNet++ decoder |
| Obj Dec   | ~15M   | -      | -    | SegFormer decoder |

Student cached layers: `[3, 8, 13, 17]` (teacher: `[4, 11, 17, 23]`)

## Pipeline

```
1. Train student encoder via KD    -> kd-encoder/checkpoints_full/student_final.pt
2. Train edge decoder              -> st-edge-mask/checkpoints/checkpoint_best.pt
3. Train object decoder            -> st-obj-mask/checkpoints/checkpoint_best.pt
4. (Optional) Apply FastVGGT       -> 3x encoder speedup, no retraining
5. Merge into unified checkpoint   -> deploy to Jetson Orin NX (TensorRT, INT8)
```

## Quick Start

### 1. Student Encoder (KD)

```bash
cd kd-encoder
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 80 --batch_size 64 \
    --learning_rate 1e-4 --warmup_epochs 3 \
    --num_workers 12 --checkpoint_dir checkpoints_full
```

### 2. Edge Decoder

```bash
cd st-edge-mask
# Place images in data/rgb/, masks in data/masks/
torchrun --nproc_per_node=2 train_ddp.py --epochs 100
```

### 3. Object Decoder

```bash
cd st-obj-mask
# Place images in data/images/, masks in data/masks/
torchrun --nproc_per_node=2 train_ddp.py --epochs 100
```

### 4. FastVGGT (Optional)

```bash
cd fastVGGT
python run_inference.py --task cascade
```

## Requirements

- 2x GPU with 16GB+ VRAM (tested on 2x A100 80GB)
- CUDA 11.8+, PyTorch 2.0+
- `pip install torch torchvision timm pillow`
