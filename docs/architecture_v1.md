# Architecture v1: VGGT + UNet++ Edge Masking

## Objective

Build a high-quality edge masking model using a fixed VGGT encoder and a custom UNet++ decoder. The model predicts only edge masks. This is not a semantic segmentation task. The primary goal is accurate reconstruction of thin boundaries and edge structures.

---

## Full Architecture Diagram

```
Input: [B, S, 3, 518, 518]
│
▼ torch.no_grad(), model.eval()
┌─────────────────────────────────────────────────────────────┐
│  VGGT Encoder (Frozen)                                      │
│                                                             │
│  Patch Embed (DINOv2 ViT-L, patch_size=14)                  │
│       ↓                                                     │
│  Aggregator (24 layers, alternating frame/global attention)  │
│       ↓                                                     │
│  Output: aggregated_tokens_list[i] = [B, S, 1374, 2048]     │
│          (1 camera + 4 register + 1369 patch tokens, 2×1024) │
└─────────────────────────────────────────────────────────────┘
│
├── Layer 4  → [B, S, 1374, 2048]
├── Layer 11 → [B, S, 1374, 2048]
├── Layer 17 → [B, S, 1374, 2048]
└── Layer 23 → [B, S, 1374, 2048]
│
▼ slice [:, :, 5:] → [B, S, 1369, 2048]
▼ reshape → [B*S, 2048, 37, 37]
▼ .detach()
│
┌─────────────────────────────────────────────────────────────┐
│  Feature Projection (4 levels)                              │
│                                                             │
│  Level 0 (Layer 4):  Conv1×1(2048→64)  + GN8 + SiLU        │
│                      bilinear(size=148) + Conv3×3 + GN8+SiLU│
│                      → [B*S, 64, 148, 148]                  │
│                                                             │
│  Level 1 (Layer 11): Conv1×1(2048→128) + GN8 + SiLU        │
│                      bilinear(size=74)  + Conv3×3 + GN8+SiLU│
│                      → [B*S, 128, 74, 74]                   │
│                                                             │
│  Level 2 (Layer 17): Conv1×1(2048→256) + GN8 + SiLU        │
│                      → [B*S, 256, 37, 37]                   │
│                                                             │
│  Level 3 (Layer 23): Conv1×1(2048→512) + GN8 + SiLU        │
│                      Conv3×3(stride=2)  + GN8 + SiLU        │
│                      → [B*S, 512, 19, 19]                   │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│  UNet++ Dense Decoder                                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Level 0 (148×148, 64ch)                              │   │
│  │                                                      │   │
│  │ X(0,0)───→X(0,1)────→X(0,2)─────→X(0,3)            │   │
│  │   │  ↗      │  ↗        │  ↗                        │   │
│  │   │ /       │ /         │ /                          │   │
│  └───┼─────────┼───────────┼────────────────────────────┘   │
│      │ Up      │ Up        │ Up                             │
│  ┌───┼─────────┼───────────┼────────────────────────────┐   │
│  │ Level 1 (74×74, 128ch)                               │   │
│  │                                                      │   │
│  │ X(1,0)───→X(1,1)────→X(1,2)                         │   │
│  │   │  ↗      │  ↗                                    │   │
│  │   │ /       │ /                                      │   │
│  └───┼─────────┼───────────────────────────────────────-┘   │
│      │ Up      │ Up                                         │
│  ┌───┼─────────┼───────────────────────────────────────-┐   │
│  │ Level 2 (37×37, 256ch)                               │   │
│  │                                                      │   │
│  │ X(2,0)───→X(2,1)                                    │   │
│  │   │  ↗                                              │   │
│  │   │ /                                                │   │
│  └───┼──────────────────────────────────────────────────┘   │
│      │ Up                                                   │
│  ┌───┼──────────────────────────────────────────────────┐   │
│  │ Level 3 (19×19, 512ch)                               │   │
│  │                                                      │   │
│  │ X(3,0)                                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
│
├── DS Head 1 @ X(0,1): Conv3×3(64→32) + SiLU + Conv1×1(32→1) + bilinear(518)
├── DS Head 2 @ X(0,2): Conv3×3(64→32) + SiLU + Conv1×1(32→1) + bilinear(518)
│
▼ X(0,3): [B*S, 64, 148, 148]
┌─────────────────────────────────────────────────────────────┐
│  Edge Refinement (Residual)                                 │
│                                                             │
│  x + (Conv3×3 + GN8 + SiLU + Conv3×3 + GN8 + SiLU)(x)     │
│                                                             │
│  → [B*S, 64, 148, 148]                                     │
└─────────────────────────────────────────────────────────────┘
│
▼
Conv1×1(64 → 1) → [B*S, 1, 148, 148]
│
▼
F.interpolate(size=518, bilinear) → [B*S, 1, 518, 518]
│
▼ reshape → [B, S, 1, 518, 518]
│
▼ sigmoid (inference only; logits to loss during training)
│
Edge Mask: [B, S, 1, 518, 518]
```

---

## VGGT Encoder Details

| Parameter | Value |
|-----------|-------|
| Input image size | 518 x 518 |
| Patch size | 14 |
| Patch grid | 37 x 37 (1369 patches) |
| Embed dimension | 1024 |
| Aggregator depth | 24 layers |
| Attention heads | 16 |
| MLP ratio | 4.0 |
| Register tokens | 4 |
| Camera tokens | 1 |
| Total tokens per frame | 1374 (1 + 4 + 1369) |
| Output dim (frame + global concat) | 2048 |
| Intermediate layer indices | [4, 11, 17, 23] |
| patch_start_idx | 5 |

The encoder is completely frozen. No modifications, no added layers, no fine-tuning.

---

## Synthetic Feature Pyramid

All VGGT intermediate features have the same spatial resolution (37x37). The multi-scale pyramid is constructed synthetically by assigning layers to different spatial levels:

| Level | Source Layer | Channels | Spatial Size | Resize Operation |
|-------|-------------|----------|--------------|-----------------|
| 0 (finest) | Layer 4 | 64 | 148x148 | Bilinear upsample (size=148) + Conv3x3 |
| 1 | Layer 11 | 128 | 74x74 | Bilinear upsample (size=74) + Conv3x3 |
| 2 | Layer 17 | 256 | 37x37 | Identity (native resolution) |
| 3 (coarsest) | Layer 23 | 512 | 19x19 | Conv3x3 stride=2 |

Rationale: Early transformer layers (4) encode low-level patterns and are assigned to the finest level. Deep layers (23) encode high-level semantics and are assigned to the coarsest level. The decoder reconstructs edge detail through learned convolutions at each spatial scale.

---

## Decoder Structure

### UNet++ Node Definitions

| Node | Inputs (concatenated) | In Channels | Out Channels |
|------|----------------------|-------------|--------------|
| X(3,0) | Level 3 projection | 512 | 512 |
| X(2,0) | Level 2 projection | 256 | 256 |
| X(2,1) | cat[X(2,0), Up(X(3,0))→256] | 512 | 256 |
| X(1,0) | Level 1 projection | 128 | 128 |
| X(1,1) | cat[X(1,0), Up(X(2,0))→128] | 256 | 128 |
| X(1,2) | cat[X(1,0), X(1,1), Up(X(2,1))→128] | 384 | 128 |
| X(0,0) | Level 0 projection | 64 | 64 |
| X(0,1) | cat[X(0,0), Up(X(1,0))→64] | 128 | 64 |
| X(0,2) | cat[X(0,0), X(0,1), Up(X(1,1))→64] | 192 | 64 |
| X(0,3) | cat[X(0,0), X(0,1), X(0,2), Up(X(1,2))→64] | 256 | 64 |

### ConvBlock (used at every decoder node)

```
Conv2d(in_ch, out_ch, 3, padding=1)
GroupNorm(8, out_ch)
SiLU()
Conv2d(out_ch, out_ch, 3, padding=1)
GroupNorm(8, out_ch)
SiLU()
```

### Upsample Block (used for every skip connection from lower level)

```
F.interpolate(x, size=(H_target, W_target), mode="bilinear", align_corners=False)
Conv2d(in_ch, out_ch, 3, padding=1)
GroupNorm(8, out_ch)
SiLU()
```

---

## Deep Supervision

Heads attached to X(0,1), X(0,2), and final output path from X(0,3).

Each deep supervision head:
```
Conv2d(64, 32, 3, padding=1)
SiLU()
Conv2d(32, 1, 1)
F.interpolate(size=518, bilinear)
```

Training loss weights:
```
L_total = 1.0 * Loss(final) + 0.2 * Loss(DS2) + 0.1 * Loss(DS1)
```

Only the final head is used at inference.

---

## Loss Design

```
Loss = 0.5 * WeightedBCE + 0.5 * DiceLoss
```

### Weighted BCE
```python
pos_weight = clamp(neg_pixels / pos_pixels, 5, 25)
F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
```

pos_weight is computed per-batch.

### Dice Loss
```python
pred = sigmoid(logits)
intersection = sum(pred * target)
dice = 1 - (2 * intersection + 1e-6) / (sum(pred) + sum(target) + 1e-6)
```

### Total Training Loss
```
L_total = 1.0 * (0.5*BCE + 0.5*Dice)_final
        + 0.2 * (0.5*BCE + 0.5*Dice)_ds2
        + 0.1 * (0.5*BCE + 0.5*Dice)_ds1
```

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealing + 5% linear warmup |
| Batch size | 4-8 |
| Gradient clipping | max_norm=1.0 |
| Encoder | Frozen (requires_grad=False, eval mode) |
| Input size | 518x518 |
| Precision | Mixed (fp16 forward, fp32 loss) |
| Trainable parameters | ~9.9M |

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Boundary F1 (BF1) | Primary metric |
| ODS (Optimal Dataset Scale) | F1 at best threshold across dataset |
| OIS (Optimal Image Scale) | F1 at best per-image threshold |
| Dice | Region overlap on edge pixels |
| Thin-edge recall | Recall on edges <= 2px wide |

---

## Key Constants

```
Input:              518 x 518
Patch size:         14
Patch grid:         37 x 37 (1369 patches)
Encoder dim:        2048 (1024 frame + 1024 global, concatenated)
patch_start_idx:    5 (1 camera + 4 register tokens)
Cached layers:      [4, 11, 17, 23]
Decoder channels:   [64, 128, 256, 512]
Decoder spatial:    [148x148, 74x74, 37x37, 19x19]
Normalization:      GroupNorm(num_groups=8)
Activation:         SiLU (decoder only)
Trainable params:   ~9.9M
```
