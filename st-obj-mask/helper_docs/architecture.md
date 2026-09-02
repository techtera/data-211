# Architecture

## System Overview

Object mask segmentation system using a distilled student encoder (255M params) with a SegFormer-based decoder. The encoder is frozen during decoder training — only the decoder learns to map encoder features to segmentation masks.

## Project Goals

1. Train a decoder that produces accurate object segmentation masks from student encoder features
2. Achieve mIoU > 0.7 on the validation set
3. Keep decoder lightweight for Orin NX deployment (combined with encoder under 16GB)

## Major Components

### Student Encoder (Frozen)
- **Module**: `student/aggregator.py` → `StudentAggregator` (from `../kd-encoder/`)
- **Architecture**: ViT with 18 layers, 768-dim per branch (1536-dim concatenated frame+global)
- **Cached layers**: [3, 8, 13, 17]
- **Parameters**: 255M (all frozen)
- **Output**: List of cached features [B, S, 1374, 1536] at 4 layers, plus patch_start_idx=5

### Object Mask Decoder (Trainable)
- **Module**: `obj_mask/segformer_head.py` → `DPTHead` (aliased as `ObjMaskDecoder`)
- **Pipeline**:
  1. Extract patch tokens (strip 5 special tokens: 1 camera + 4 registers)
  2. LayerNorm on transformer embeddings
  3. 1x1 Conv projection per layer (1536 → [256, 512, 1024, 1024])
  4. Positional embedding injection (sinusoidal UV grid)
  5. Resize to multi-scale feature pyramid
  6. SegFormer decoder (MLP embedding + concatenation + fusion + classification)
  7. Bilinear upsample to original resolution (518x518)

### Feature Pyramid Levels
| Level | Channels | Spatial Size | Stride |
|-------|----------|-------------|--------|
| c1    | 256      | 84 x 148   | 4      |
| c2    | 512      | 42 x 74    | 8      |
| c3    | 1024     | 21 x 37    | 16     |
| c4    | 1024     | 11 x 19    | 32     |

### Wrapper Model
- **Module**: `obj_mask/model.py` → `StudentObjMask`
- **Role**: Wraps encoder + decoder, handles 4D/5D input conversion
- **Input**: [B, 3, 518, 518] or [B, S, 3, 518, 518]
- **Output**: [B, 2, 518, 518] or [B, S, 2, 518, 518]

## External Dependencies

- PyTorch 2.0+ with CUDA
- Student encoder checkpoint from `../kd-encoder/checkpoints/student_final.pt`
- OpenCV (cv2) for mask polygon rasterization in dataset

## Interfaces

- **Training Input**: Images [B, 3, 518, 518] in [0,1], Masks [B, 518, 518] as long tensor (0=background, 1=object)
- **Training Output**: Logits [B, 2, 518, 518]
- **Inference Input**: Images [B, 3, 518, 518] or [B, S, 3, 518, 518]
- **Inference Output**: Mask logits [B, 2, 518, 518] → argmax for binary mask

## Data Flow

```
Input Images [B, 3, 518, 518]
    ↓
unsqueeze → [B, 1, 3, 518, 518]
    ↓
StudentAggregator (frozen, no_grad)
    ↓
Cached Features at layers [3, 8, 13, 17]
    → 4 tensors of shape [B, 1, 1374, 1536]
    ↓
Strip Special Tokens (first 5) → [B*S, 1369, 1536]
    ↓
LayerNorm → [B*S, 1369, 1536]
    ↓
Reshape to Spatial → [B*S, 1536, 37, 37]
    ↓
1x1 Conv Projection → per-level channels
    ↓
Positional Embedding (sinusoidal UV)
    ↓
Resize Layers → Feature Pyramid [c1, c2, c3, c4]
    ↓
SegFormer Decoder → [B*S, 2, 84, 148]
    ↓
Bilinear Upsample → [B*S, 2, 518, 518]
    ↓
Reshape → [B, S, 2, 518, 518]
    ↓
Squeeze if input was 4D → [B, 2, 518, 518]
```

## Design Principles

1. **Frozen encoder**: No gradients flow through the encoder — decoder-only training
2. **Positional embeddings**: Injected into feature maps since spatial info is lost after ViT tokenization
3. **Chunked inference**: Long sequences split into chunks to control memory usage
4. **Multi-scale features**: 4-level pyramid captures both fine and coarse spatial information

## Architectural Constraints

- Student encoder output must be 1536-dim (768 frame + 768 global concatenated)
- Token structure: [1 camera, 4 registers, 1369 patches] = 1374 tokens
- Patch size: 14x14 → 518/14 = 37 patches per spatial dimension
- ImageNet normalization handled inside the encoder

## Known Weaknesses

- No data augmentation in the training pipeline
- Single-frame only (S=1) — no temporal context
- Decoder channel counts are large (1024 at deeper levels) — may need reduction for deployment
- SegFormer decoder uses a single shared LayerNorm across all pyramid levels
