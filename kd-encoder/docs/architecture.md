# Student Encoder Architecture

**Version:** 1.0  
**Date:** 2026-08-24  
**Status:** Frozen (approved for implementation)

---

## Overview

The student encoder is a compressed version of the VGGT-1B teacher encoder, designed to achieve ≥1.5x latency speedup and ≥2x memory reduction while maintaining feature representation quality.

**Key Design Decisions:**
- Reduce depth: 24 → 18 layers (1.3x fewer)
- Reduce width: 1024 → 768 dim (1.3x smaller)
- Maintain architecture pattern: Alternating frame/global attention
- Maintain token structure: Same 1374 tokens as teacher

---

## Architecture Comparison

### Teacher Encoder (VGGT-1B)

```python
Architecture:
  embed_dim: 1024
  depth: 24 layers
  num_heads: 16
  mlp_ratio: 4.0
  patch_size: 14
  img_size: 518
  
Structure:
  - 24 frame attention blocks
  - 24 global attention blocks
  - Alternating pattern: [frame, global, frame, global, ...]
  
Cached Layers: [4, 11, 17, 23]
Token Structure: 1 camera + 4 register + 1369 patches = 1374 tokens
```

**Performance:**
- Parameters: ~885M
- Latency: 250ms/frame (FP16, A100)
- Memory: 10GB (FP16, batch_size=1)

---

### Student Encoder

```python
Architecture:
  embed_dim: 768
  depth: 18 layers
  num_heads: 12
  mlp_ratio: 4.0
  patch_size: 14
  img_size: 518
  
Structure:
  - 18 frame attention blocks
  - 18 global attention blocks
  - Alternating pattern: [frame, global, frame, global, ...]
  
Cached Layers: [3, 8, 13, 17]
Token Structure: 1 camera + 4 register + 1369 patches = 1374 tokens
```

**Target Performance:**
- Parameters: ~342M (2.6x fewer)
- Latency: ≤167ms/frame (≥1.5x speedup, FP16, A100)
- Memory: ≤5GB (≥2x reduction, FP16, batch_size=1)

---

## Feature Dimensions

### Complete Pipeline

```python
# 1. Input Images
images: [B, S, 3, 518, 518]
# B = batch size
# S = number of frames (1 for Phase 0A, 8 for Phase 1)

# 2. After Patch Embedding
patch_tokens: [B*S, ~1369, 768]
# 518/14 ≈ 37 → 37×37 = 1369 patches

# 3. After Adding Special Tokens
all_tokens: [B*S, 1374, 768]
# 1 camera + 4 register + 1369 patches

# 4. After Frame Block i
frame_features: [B, S, 1374, 768]

# 5. After Global Block i
global_features: [B, S, 1374, 768]

# 6. Cached Layer Features (frame + global concatenated)
cached_features: [B, S, 1374, 1536]
# 1536 = 768 (frame) + 768 (global)

# 7. Final Output (all 4 cached layers)
output: List[Tensor] of length 4
  Each: [B, S, 1374, 1536]
```

**Comparison with Teacher:**
- Teacher cached features: `[B, S, 1374, 2048]` (1024+1024)
- Student cached features: `[B, S, 1374, 1536]` (768+768)

---

## Layer Mapping (Teacher → Student)

### Proportional Mapping Formula

```python
student_layer = round(teacher_layer × (student_depth / teacher_depth))
ratio = 18 / 24 = 0.75
```

### Frozen Mapping

```
Teacher Layer → Student Layer → Purpose
    4         →      3        → Early visual features
   11         →      8        → Mid-level features
   17         →     13        → Late-mid features
   23         →     17        → High-level semantic features
```

**Computation:**
```python
Layer 4  × 0.75 = 3.0  → 3
Layer 11 × 0.75 = 8.25 → 8
Layer 17 × 0.75 = 12.75 → 13
Layer 23 × 0.75 = 17.25 → 17
```

**Properties:**
- Well-spaced distribution across student depth
- Maintains early/mid/late feature hierarchy
- Proportional mapping is standard in ViT distillation

---

## Parameter Breakdown (ESTIMATED)

**IMPORTANT:** These are rough estimates. Actual count must be measured in Phase 0A.

### Per-Block Parameters

**Attention Block:**
- QKV projection: 3 × (768 × 768) = 1,769,472
- Output projection: 768 × 768 = 589,824
- LayerNorm (pre-attention): 2 × 768 = 1,536
- Total attention: ~2.36M

**MLP Block:**
- Linear1: 768 × (768 × 4) = 2,359,296
- Linear2: (768 × 4) × 768 = 2,359,296
- LayerNorm (pre-MLP): 2 × 768 = 1,536
- Total MLP: ~4.72M

**Full Block (Attention + MLP):**
- ~7.08M per block

### Total Estimate

```
Component                      Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patch Embedding                ~0.6M
Special Tokens (1 + 4)         ~0.006M
Frame Blocks (18 × 7.08M)      ~127.4M
Global Blocks (18 × 7.08M)     ~127.4M
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL (estimated)              ~342M
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Target:** ≤ 400M

**Verification command:**
```python
sum(p.numel() for p in model.parameters())
```

---

## Initialization Strategy

### DINOv2 ViT-Base Pretrained Initialization

**Source Model:**
- Name: `dinov2_vitb14_reg`
- Architecture: ViT-Base with register tokens
- Dimension: 768 (matches student!)
- Depth: 12 blocks
- Pretrained on: 142M images (ImageNet + web-scale data)

### Weight Transfer

```python
DINOv2 ViT-Base (12 blocks, 768 dim)
         ↓
Student Encoder (18 blocks, 768 dim)

Transfer:
  Patch Embedding:      DINOv2 → Student ✓
  Frame Blocks 0-11:    DINOv2 blocks 0-11 → Student frame blocks 0-11 ✓
  Frame Blocks 12-17:   Random initialization
  Global Blocks 0-11:   DINOv2 blocks 0-11 → Student global blocks 0-11 ✓
  Global Blocks 12-17:  Random initialization
  Camera Token:         Random (std=1e-6)
  Register Tokens:      Random (std=1e-6)
```

**Rationale:**
1. **Dimension match:** DINOv2 (768) = Student (768) → direct weight copy
2. **Strong features:** DINOv2 pretrained on massive vision datasets
3. **Faster convergence:** 20-30% fewer epochs vs random initialization
4. **Training stability:** Pretrained weights reduce early training instability

**Why NOT initialize from VGGT teacher:**
- Teacher dimension (1024) ≠ Student dimension (768) → can't directly copy
- DINOv2 provides cleaner initialization path

### Branch Specialization

**Initial State:**
- Frame blocks 0-11: DINOv2 weights
- Global blocks 0-11: **Same** DINOv2 weights (identical copy)
- Both branches start with identical features

**Expected During Training:**
- Frame/global branches gradually diverge
- Gradient-driven specialization
- Monitor: Cosine similarity between branches should decrease from ~1.0 to <0.8

---

## Token Structure

### VGGT Token Layout

```
Position  Count  Type
━━━━━━━━━━━━━━━━━━━━━━━━━━
0         1      Camera token (pose information)
1-4       4      Register tokens (global aggregation)
5+        1369   Patch tokens (37×37 grid)
━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL     1374   tokens
```

**Patch Start Index:** 5

**Grid Layout:**
```
Input image: 518×518
Patch size:  14×14
Grid size:   37×37 = 1369 patches
```

---

## Attention Pattern

### Alternating Frame-Global Attention

```
Layer 0:  Frame Attention   (within-frame, across patches)
Layer 1:  Global Attention  (across-frame, within patches)
Layer 2:  Frame Attention
Layer 3:  Global Attention  [CACHED]
Layer 4:  Frame Attention
...
Layer 8:  Frame Attention   [CACHED]
...
Layer 13: Global Attention  [CACHED]
...
Layer 17: Global Attention  [CACHED]
```

**Frame Attention:**
- Operates on: `[B*S, P, D]` (batch-flattened frames)
- Attention across: Patches within each frame
- Does NOT see: Other frames

**Global Attention:**
- Operates on: `[B, S*P, D]` (sequence-flattened)
- Attention across: All patches in all frames
- Cross-frame information flow

---

## Cached Layers

### Purpose

During distillation, we extract intermediate features from both teacher and student at specific layers for loss computation.

### Student Cached Layers

```python
cached_layer_indices = [3, 8, 13, 17]

# Layer 3:  Early features (after 3 frame + 3 global blocks)
# Layer 8:  Mid-early features
# Layer 13: Mid-late features
# Layer 17: High-level features (near output)
```

### Output Format

```python
# At each cached layer, concatenate frame + global features
frame_output = frame_blocks[i](x)      # [B, S, P, 768]
global_output = global_blocks[i](x)    # [B, S, P, 768]
cached = torch.cat([frame_output, global_output], dim=-1)  # [B, S, P, 1536]
```

**Note:** Concatenation happens at **feature dimension** (dim=-1), not sequence dimension.

---

## Architecture Invariants

These properties are **preserved** from teacher:

1. **Token count:** 1374 tokens (same as teacher)
2. **Token structure:** 1 camera + 4 register + 1369 patches
3. **Alternating pattern:** Frame → Global → Frame → Global
4. **Patch size:** 14×14 (same as teacher)
5. **Image size:** 518×518 (same as teacher)
6. **MLP ratio:** 4.0 (same as teacher)

These properties are **changed** from teacher:

1. **Depth:** 24 → 18 layers (1.3x fewer)
2. **Width:** 1024 → 768 dim (1.3x smaller)
3. **Heads:** 16 → 12 (proportional to width)
4. **Cached layers:** [4,11,17,23] → [3,8,13,17]
5. **Output dim:** 2048 → 1536 (concatenated features)

---

## Design Rationale

### Why 18 Layers?

- **Not too shallow:** ≥16 layers to maintain feature hierarchy
- **Not too deep:** ≤20 layers to achieve speedup target
- **Clean ratio:** 18/24 = 0.75 (simple proportional mapping)

### Why 768 Dimension?

- **Matches DINOv2:** Clean pretrained weight transfer
- **Standard size:** ViT-Base standard (proven architecture)
- **Speedup target:** 768/1024 ≈ 0.75 (1.3x parameter reduction per layer)

### Why 12 Heads?

- **Proportional:** 12/16 = 0.75 (same ratio as dimension reduction)
- **Divisibility:** 768 / 12 = 64 (standard head dimension)
- **Standard:** ViT-Base uses 12 heads

---

## Phase 0A Verification

In Phase 0A benchmarking, we will verify:

1. **Architecture builds correctly:**
   - No shape mismatches
   - Forward pass completes
   - Cached layers return correct shapes

2. **Parameters within target:**
   - Actual count ≤ 400M
   - Breakdown matches estimates

3. **Performance within target:**
   - Latency ≥ 1.5x faster than teacher
   - Memory ≥ 2x less than teacher

4. **Initialization succeeds:**
   - DINOv2 weights load without errors
   - Blocks 0-11 initialized from pretrained
   - Blocks 12-17 randomly initialized

If any verification fails → redesign architecture before Phase 1.

---

## References

- VGGT Architecture: Teacher encoder in `../../vggt-unified/`
- DINOv2: https://github.com/facebookresearch/dinov2
- ViT Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- Distillation Plan: `../../../.claude/plans/deep-gathering-plum.md`
