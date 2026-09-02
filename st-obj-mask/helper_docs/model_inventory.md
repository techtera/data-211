# Model Inventory

## StudentObjMask

### Model Architecture

**Type**: Encoder-Decoder for binary object segmentation
**Wrapper**: `obj_mask/model.py` → `StudentObjMask`

### Encoder

- **Name**: StudentAggregator
- **Source**: `../kd-encoder/student/aggregator.py`
- **Parameters**: 255M (frozen during decoder training)
- **Architecture**: ViT with 18 transformer layers, 768-dim per branch
- **Initialization**: DINOv2 ViT-Large (1024→768 truncation projection)
- **Attention**: Alternating frame (per-image) and global (cross-image) attention
- **Cached Layers**: [3, 8, 13, 17]
- **Output**: List of [B, S, 1374, 1536] tensors at cached layers + patch_start_idx=5
- **Token Structure**: [1 camera, 4 registers, 1369 patches] = 1374 tokens
- **Output Dimension**: 1536 (768 frame + 768 global concatenated)

### Decoder

- **Name**: ObjMaskDecoder (DPTHead with SegFormer)
- **Source**: `obj_mask/segformer_head.py` → `DPTHead`
- **Components**:
  - LayerNorm (1536-dim)
  - 4x 1x1 Conv projection layers (1536 → [256, 512, 1024, 1024])
  - 4x Resize layers (ConvTranspose2d stride 4, ConvTranspose2d stride 2, Identity, Conv2d stride 2)
  - Sinusoidal positional embedding injection
  - SegFormer decoder (4x MLP linear embedding + concat + Conv1x1 fusion + classification)
- **SegFormer Decoder**: `obj_mask/segformer_decoder.py` → `SegFormerDecoder`
  - Embedding dim: 256
  - Dropout: 0.1
  - Final prediction: Conv2d(256, 2, 1)

### Prediction Heads

- **Output**: 2-class logits (background, object)
- **Upsampling**: Bilinear interpolation from decoder output to 518x518

### Loss Functions

- **CrossEntropy Loss**: Standard pixel-wise classification (weight: 1.0)
- **Dice Loss**: Multi-class Dice with smoothing=1.0 (weight: 1.0)
- **Combined**: `SegmentationLoss` = CE + Dice

### Datasets

- **Format**: YOLO segmentation annotations (polygon coordinates)
- **Structure**: `data/images/` + `data/labels/` (*.txt with class_id x1 y1 x2 y2 ...)
- **Preprocessing**: Resize to 518x518, polygon → binary mask via cv2.fillPoly
- **Split**: 90% train, 10% validation (seed=42)
- **No augmentation** currently applied

### Augmentations

- None (Albumentations transform parameter exists but is unused)

### Metrics

- **mIoU**: Mean Intersection over Union (2-class)
- **Dice Score**: Mean Dice coefficient
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Per-class IoU**: Background IoU + Object IoU

### Training Configuration

- **Framework**: DDP via torchrun
- **Batch Size**: 2 per GPU
- **Optimizer**: AdamW (LR=1e-4, weight_decay=1e-2)
- **Scheduler**: Linear warmup (5% of steps) + cosine decay to 0
- **Gradient Clipping**: Max norm 1.0
- **Epochs**: Up to 100 (early stopping at patience=15 on val mIoU)
- **Encoder**: Frozen (no gradients)

### Inference Configuration

- **Input**: [B, 3, 518, 518] in [0, 1] range
- **Output**: [B, 2, 518, 518] logits → argmax for binary mask
- **Chunked inference**: Supported for sequences > frames_chunk_size (default=8)

### Checkpoint Locations

- **Student Encoder**: `../kd-encoder/checkpoints_v2/student_final.pt`
- **Best Decoder (mIoU)**: `checkpoints/checkpoint_best.pt`
- **Best Decoder (loss)**: `checkpoints/checkpoint_best_loss.pt`
- **Latest Decoder**: `checkpoints/checkpoint_last.pt`

### Known Limitations

- No data augmentation — may overfit on small datasets
- Single-frame only (S=1) — no temporal context
- Encoder quality depends on KD training (not yet evaluated)
- Decoder channel counts are large (1024) — may need reduction for Orin NX
- Shared LayerNorm across all pyramid levels in SegFormer
- No mixed precision (AMP) — runs in FP32
