# Architecture

## System Overview

Knowledge distillation system that compresses a 909M-parameter VGGT encoder (teacher) into a 255M-parameter student encoder. The student learns to replicate the teacher's intermediate feature representations through layer-wise distillation with MSE + cosine similarity loss.

## Project Goals

1. Compress VGGT encoder from 909M to 255M parameters (~3.6x reduction)
2. Preserve feature quality sufficient for downstream decoder tasks (object masks, edge masks)
3. Enable deployment on Jetson Orin NX 16GB with <1s latency (via future TensorRT conversion)

## Major Components

### Teacher Encoder (Frozen)
- **Source**: VGGT unified model (`vggt-unified/`)
- **Architecture**: ViT with 24 layers, 2048-dim (1024-dim per branch, frame+global concatenated)
- **Cached layers**: [4, 11, 17, 23]
- **Parameters**: 909M
- **Role**: Provides target features for distillation (frozen, FP16)

### Student Encoder (Trainable)
- **Module**: `student/aggregator.py` → `StudentAggregator`
- **Architecture**: ViT with 18 layers, 1536-dim (768-dim per branch, frame+global concatenated)
- **Cached layers**: [3, 8, 13, 17]
- **Parameters**: 255M
- **Initialization**: DINOv2 ViT-Large (1024-dim) projected to 768-dim
- **Attention pattern**: Alternating frame attention (per-image) and global attention (cross-image)

### Distillation Pipeline
- **Projection Heads**: 4 separate LayerNorm+Linear heads (1536→2048), ~12.6M params total, training-only
- **Loss Function**: MSE (70%) + Cosine (30%) per layer, progressive layer weighting [1.0, 1.5, 2.0, 2.5]
- **Token Sampling**: 1374→133 tokens (90% memory reduction), shared indices between teacher and student

### Training Infrastructure
- **DDP**: DistributedDataParallel via `torchrun`, `train_ddp.py`
- **Dataset**: ImageDataset loading JPG/PNG images, resized to 518x518, `num_frames=1`
- **Optimizer**: AdamW with cosine LR schedule + linear warmup
- **Checkpointing**: Last, best (lowest loss), and periodic saves

## External Dependencies

- PyTorch 2.0+ with CUDA 11.8+
- DINOv2 ViT-Large (via `torch.hub`, ~1.2GB download)
- VGGT unified model from `../../vggt-unified/` (teacher checkpoint)

## Interfaces

- **Input**: Images [B, S=1, 3, 518, 518] in [0,1] range
- **Output**: List of cached layer features [B, S, P=1374, 1536] (4 layers, rest are None)
- **Checkpoint format**: `{student_state_dict, optimizer_state_dict, scheduler_step, loss, epoch, projection_state_dict}`

## Data Flow

```
Input Images [B, S, 3, 518, 518]
    ↓
ImageNet Normalization
    ↓
Patch Embedding (Conv2d 14x14) → [B*S, 1369, 768]
    ↓
Prepend Special Tokens (1 camera + 4 register) → [B*S, 1374, 768]
    ↓
RoPE Position Encoding
    ↓
18× Alternating Attention Layers:
    Frame Block (per-image attention) → [B*S, 1374, 768]
    Global Block (cross-image attention) → [B, S*1374, 768]
    ↓
    At cached layers [3, 8, 13, 17]:
        Concatenate frame+global → [B, S, 1374, 1536]
        LayerNorm → output_list
    ↓
Output: List[Optional[Tensor]], patch_start_idx=5
```

## Design Principles

1. Memory efficiency: Token sampling reduces distillation memory 10x
2. Teacher features are sampled and detached immediately after forward pass
3. Gradient checkpointing for non-cached layers during training
4. Projection heads are training-only artifacts, discarded after distillation

## Architectural Constraints

- Student must produce features compatible with teacher's decoder heads
- Token structure must match: [camera, 4 registers, 1369 patches]
- Output dimension must be 1536 (768 frame + 768 global concatenated)
- ImageNet normalization is handled internally by the model

## Known Weaknesses

- Single-frame training (`num_frames=1`): student may not learn temporal relationships
- DINOv2-Large initialization uses truncation (first 768 of 1024), which discards information
- Token sampling is random per step (not deterministic per image)
