# Architecture

## System Overview

Edge mask prediction system using a distilled student encoder (255M params, frozen) paired with a UNet++ decoder. The system predicts binary edge masks from input images, designed for deployment on Jetson Orin NX 16GB.

## Project Goals

1. Train a lightweight edge decoder on top of the distilled student encoder
2. Achieve competitive edge detection metrics (F1, BF1, ODS) compared to the teacher-based pipeline
3. Fit within the Orin NX 16GB memory budget alongside the object mask decoder

## Major Components

### Student Encoder (Frozen)
- **Module**: `student.StudentAggregator` (imported from `../kd-encoder`)
- **Parameters**: 255M (all frozen)
- **Architecture**: ViT with 18 layers, 768-dim per branch (1536 concatenated frame+global)
- **Cached layers**: [3, 8, 13, 17]
- **Output**: List of 18 tensors (14 None, 4 cached) of shape [B, S, 1374, 1536]
- **patch_start_idx**: 5 (1 camera + 4 register tokens)

### StudentFeatureExtractor (`edge_mask/feature_extractor.py`)
- **Role**: Wraps the frozen aggregator, extracts patch tokens from cached layers, projects them to multi-scale spatial feature maps
- **Projections** (4 FeatureProjection modules):
  - Layer 3 → 1536→64, target 148x148 (Conv2d 1x1 + GroupNorm(8) + SiLU + Conv2d 3x3 + interpolate)
  - Layer 8 → 1536→128, target 74x74
  - Layer 13 → 1536→256, native 37x37 (no resize)
  - Layer 17 → 1536→512, downsampled via stride-2 Conv2d 3x3
- **Output**: List of 4 feature maps [(B*S, 64, 148, 148), (B*S, 128, 74, 74), (B*S, 256, 37, 37), (B*S, 512, ~19, ~19)]

### UNetPPDecoder (`edge_mask/decoder.py`)
- **Role**: Nested dense skip connections for multi-scale feature fusion
- **Channels**: (64, 128, 256, 512)
- **Structure**: 6 Upsample blocks, 6 ConvBlocks (Conv2d 3x3 + GroupNorm + SiLU, doubled)
- **Deep Supervision**: 2 DeepSupervisionHead modules at nodes x_0_1, x_0_2 (Conv2d→SiLU→Conv2d 1→interpolate to 518x518)
- **Output**: (x_0_3, ds1_logits, ds2_logits)

### EdgeRefinement (`edge_mask/refinement.py`)
- **Role**: Post-decode spatial refinement of edge features
- **Channels**: 64

### Final Convolution
- **Module**: `nn.Conv2d(64, 1, 1)` — single channel edge logits
- **Followed by**: `F.interpolate` to 518x518

## External Dependencies

- PyTorch 2.0+ with CUDA 11.8+
- Student encoder checkpoint from `../kd-encoder/checkpoints/student_final.pt`
- StudentAggregator class from `../kd-encoder/student/`

## Interfaces

- **Input**: Images [B, 3, 518, 518] or [B, S, 3, 518, 518] in [0, 1] range
- **Training Output**: (logits, ds1_logits, ds2_logits) — each [B, 1, 518, 518] or [B, S, 1, 518, 518]
- **Eval Output**: sigmoid(logits) — edge probability map [B, 1, 518, 518]
- **Checkpoint format**: `{epoch, loss, model_state_dict, optimizer_state_dict, scheduler_state_dict}`

## Data Flow

```
Input Images [B, S, 3, 518, 518]
    ↓
StudentAggregator (frozen, no_grad)
    ↓
Cached Features at [3, 8, 13, 17] → [B, S, 1374, 1536]
    ↓
Strip special tokens (first 5) → [B, S, 1369, 1536]
    ↓
Reshape to spatial: [B*S, 1536, 37, 37]
    ↓
4× FeatureProjection:
    [B*S, 64, 148, 148]   (interpolate up)
    [B*S, 128, 74, 74]    (interpolate up)
    [B*S, 256, 37, 37]    (native)
    [B*S, 512, ~19, ~19]  (stride-2 down)
    ↓
UNet++ Decoder (nested skip connections)
    → x_0_3: [B*S, 64, 148, 148]
    → ds1:   [B*S, 1, 518, 518]
    → ds2:   [B*S, 1, 518, 518]
    ↓
EdgeRefinement → [B*S, 64, 148, 148]
    ↓
Conv2d(64, 1, 1) → [B*S, 1, ~148, ~148]
    ↓
Interpolate to 518x518
    ↓
Reshape to [B, S, 1, 518, 518]
    ↓
Training: return (logits, ds1, ds2)
Eval:     return sigmoid(logits)
```

## Design Principles

1. Encoder is always frozen — gradients flow only through the decoder
2. Multi-scale feature extraction from same-resolution encoder outputs via learned projections
3. Deep supervision provides auxiliary gradient paths for stable training
4. All normalization uses GroupNorm (not BatchNorm) for compatibility with small batch sizes
5. SiLU activation throughout (smooth, non-monotonic, good gradient flow)

## Architectural Constraints

- Student encoder output dimension must be 1536 (768 frame + 768 global concatenated)
- Cached layer indices [3, 8, 13, 17] are fixed by the student encoder architecture
- patch_start_idx = 5 (1 camera + 4 register tokens)
- Image size must be 518x518 (VGGT standard)

## Known Weaknesses

- No data augmentation in current pipeline
- Single-frame training (S=1) — no temporal edge consistency
- Validation uses model.train() mode to get 3 outputs, which means BatchNorm statistics differ from eval mode
- Feature projections add ~3M trainable parameters on top of the decoder
