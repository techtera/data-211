# VGGT Knowledge Distillation - Student Encoder

## Project Overview

Compress 909M parameter VGGT encoder → 255M parameter student via layer-wise feature distillation.

**Status:** Training in progress (80 epochs, ~30 hours, epoch 2 currently)

---

## Current Training Configuration

```bash
torchrun --nproc_per_node=2 train_ddp.py \
    --image_dir train_images \
    --epochs 80 \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --warmup_epochs 3 \
    --num_workers 12 \
    --checkpoint_dir checkpoints_full \
    --log_every 5
```

**Key Parameters:**
- **185 steps/epoch** (23,687 images ÷ 128 effective batch)
- **~7-8s/step** (after warmup, first steps slower ~30s)
- **~23 min/epoch** → **30 hours total**
- **GPU usage:** 60-70GB per A100 (2× A100 80GB)

---

## Architecture Deep Dive

### Student Model Structure

```
StudentAggregator (255M parameters)
├── Patch Embedding (412K params)
│   └── Conv2d(3, 768, kernel=14, stride=14)
│       518×518 → 37×37 grid = 1,369 patches
│
├── Special Tokens
│   ├── Camera token (1) - Random init (std=1e-6)
│   └── Register tokens (4) - Random init (std=1e-6)
│   Total: 1,374 tokens per frame
│
├── Position Encoding (RoPE)
│   └── 2D Rotary Position Embedding (frequency=100)
│       Separate for Y and X dimensions
│
├── Frame Blocks (18 layers × 7.08M = 127M params)
│   ├── Blocks 0-11: DINOv2 pretrained ✓
│   └── Blocks 12-17: Random init
│
└── Global Blocks (18 layers × 7.08M = 127M params)
    ├── Blocks 0-11: DINOv2 pretrained ✓ (same as frame)
    └── Blocks 12-17: Random init
```

**Cached Layers:** [3, 8, 13, 17] (for distillation)

---

## Layer-by-Layer Understanding

### 1. DropPath (Stochastic Depth)
- **Purpose:** Regularization - randomly drops entire residual branches
- **Mechanism:** Binary mask per batch sample (not per neuron)
- **Current:** Disabled (drop_path=0.0)

### 2. LayerScale
- **Purpose:** Training stability in deep networks
- **Initial value:** 0.01 (very small contribution)
- **During training:** Learns to increase (0.01 → 0.3-0.5)
- **Effect:** Gradual integration of attention/MLP outputs

### 3. RoPE (Rotary Position Embedding)
- **Type:** 2D position encoding for image patches
- **Mechanism:** Rotate features by position-dependent angle
- **Formula:** R(θ) = [cos θ  -sin θ; sin θ   cos θ]
- **Advantage:** Relative position encoding, no learnable params
- **Implementation:** Split 64-dim head: 32 for Y-axis, 32 for X-axis

### 4. MLP (Feed-Forward Network)
- **Architecture:** Linear(768 → 3072) → GELU → Linear(3072 → 768)
- **Expansion:** 4× (mlp_ratio=4.0)
- **Role:** Token-wise processing (no cross-token mixing)
- **Parameters:** ~4.7M per MLP block

### 5. PatchEmbed
- **Method:** Conv2d with kernel_size=stride=patch_size (non-overlapping)
- **Input:** [64, 3, 518, 518]
- **Output:** [64, 1369, 768]
- **Efficiency:** Single GPU operation vs loop over patches

### 6. Attention (Multi-Head Self-Attention)
- **Heads:** 12
- **Head dim:** 64 (768 / 12)
- **Complexity:** O(N²·d) - Quadratic in sequence length!
- **Formula:** Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V
- **Optimization:** Fused attention (F.scaled_dot_product_attention)

### 7. Block (Transformer Block)
- **Structure:** Pre-LN (LayerNorm before, not after)
- **Formula:**
  ```python
  x = x + LayerScale(Attention(LayerNorm(x), pos))
  x = x + LayerScale(MLP(LayerNorm(x)))
  ```
- **Residual connections:** Critical for gradient flow in 18 layers

### 8. Aggregator (Full Model)
- **Alternating frame/global attention:**
  - Frame: [B*S, P, C] - Within frame
  - Global: [B, S*P, C] - Across frames
- **For S=1 (images):** Frame and global see identical shapes (redundant but useful)
- **For S>1 (video):** Global merges frames for temporal relationships
- **Gradient checkpointing:** Non-cached layers recomputed during backward (saves ~7.7GB)

### 9. Initialization (DINOv2 Transfer Learning)
- **Source:** DINOv2 ViT-Base (768 dim, 12 layers, 86M params)
- **Pretrained on:** 142M images (self-supervised)
- **Transfer strategy:**
  - Patch embedding: DINOv2 → Student ✓
  - Blocks 0-11: DINOv2 → Student (frame and global) ✓
  - Blocks 12-17: Random init (no pretrained available)
  - Special tokens: Random init (std=1e-6)
- **Result:** 66% pretrained, 34% random

---

## Shape Flow Reference

```python
# Complete forward pass shapes:

Input:                    [64, 1, 3, 518, 518]      # B, S, C, H, W
  ↓ Normalize
                         [64, 1, 3, 518, 518]       # ImageNet stats
  ↓ Flatten B×S
                         [64, 3, 518, 518]          # B*S
  ↓ Patch Embedding
                         [64, 1369, 768]            # B*S, P_patches, C
  ↓ Add Special Tokens
                         [64, 1374, 768]            # B*S, P, C (5 + 1369)
  ↓ Position Encoding
                         pos: [64, 1374, 2]         # (y, x) coordinates
  ↓
  ├─ Layer 0-17 (alternating):
  │   ↓ Frame Block
  │               [64, 1374, 768]
  │   ↓ Global Block (reshape to [64, 1374, 768] for S=1)
  │               [64, 1374, 768]
  │
  └─ Cache layers [3, 8, 13, 17]:
                frame:  [64, 1, 1374, 768]
                global: [64, 1, 1374, 768]
                concat: [64, 1, 1374, 1536]        # Frame + Global

Output:                  List of 18 (4 cached, 14 None) + patch_start_idx (5)
```

---

## Key Insights

### 1. Frame vs Global for S=1
**Question:** Why have both when S=1 makes them identical?

**Answer:**
- ✅ **Architecturally redundant** - Process same tensor shapes
- ✅ **Functionally useful** - 2× capacity (36 blocks vs 18)
- ✅ **Diverge during training** - Separate parameters, different gradients
- ✅ **Legacy from video design** - Meaningful for S>1
- ✅ **Implicit ensembling** - Two experts learn different features

### 2. Token Sampling Optimization
```
Before: [64, 1, 1374, 2048] = 181M elements
After:  [64, 1, 133, 2048]  = 17.5M elements
Reduction: 10.3× memory savings!
```

### 3. Gradient Checkpointing
- **Cached layers (4):** Save activations (needed for loss)
- **Non-cached layers (14):** Recompute during backward
- **Savings:** ~7.7GB memory

### 4. Why DINOv2 Transfer Learning Works
- ✅ Dimension match (768)
- ✅ General features from 142M images
- ✅ Faster convergence (starts from good features)
- ✅ Better final performance

### 5. Double Normalization Bug (Fixed!)
- ❌ **Was:** Dataset normalized + Model normalized = corrupted
- ✅ **Now:** Only model normalizes (expects [0,1] input)

---

## Critical Files Fixed

### Issues Resolved (Aug 25, 2026)

1. **Model forward unpacking:**
   - Both teacher and student return `(features, patch_start_idx)` tuple
   - FeaturesOnlyWrapper drops second return value
   - Fixed: Correct unpacking for wrapped vs unwrapped models

2. **Wrapper consistency:**
   - train_ddp.py now uses `DDP(FeaturesOnlyWrapper(model))`
   - Consistent with sanity_check_ddp.py
   - Fixed all checkpoint save/load unwrapping: `student.module.model`

3. **Double normalization:**
   - Removed normalization from dataset transform
   - Models normalize internally

---

## File Structure

```
kd-encoder/
├── student/                  # Student encoder
│   ├── aggregator.py        # Full model (255M params)
│   ├── initialization.py    # DINOv2 transfer learning
│   └── layers/
│       ├── attention.py     # Multi-head self-attention
│       ├── block.py         # Transformer block
│       ├── mlp.py           # Feed-forward network
│       ├── patch_embed.py   # Image → Tokens
│       ├── rope.py          # 2D rotary position encoding
│       ├── layer_scale.py   # Training stability
│       └── drop_path.py     # Stochastic depth
├── training/                 # Training pipeline
│   ├── trainer.py           # Training loops (DDP + single GPU)
│   ├── dataset.py           # Image dataset
│   ├── ddp_utils.py         # Distributed utilities
│   ├── checkpoints.py       # Save/load logic
│   ├── config.py            # Training config
│   ├── optimizer.py         # AdamW setup
│   └── scheduler.py         # LR scheduling
├── distillation/             # Knowledge distillation
│   ├── loss.py              # MSE + Cosine loss
│   ├── projection.py        # Student(1536) → Teacher(2048)
│   └── token_sampling.py    # Memory optimization
├── train_ddp.py             # Main training script ⭐
├── sanity_check_ddp.py      # 3-epoch validation
├── load_real_teacher.py     # Teacher model loader
├── verify_init.py           # Initialization verification
├── README.md                # Project overview
├── TRAINING_GUIDE.md        # Troubleshooting
├── STATUS.md                # Live training status
└── CLAUDE.md                # This file (progress log)
```

---

## Training Progress

**Started:** Aug 25, 2026  
**Current:** Epoch 2/80  
**Expected completion:** ~30 hours from start  

**Monitoring:**
```bash
# Check logs
tail -f nohup.out

# GPU usage
watch -n 1 nvidia-smi
```

**Checkpoints:**
- `checkpoints_full/checkpoint_last.pt` - Latest (every epoch)
- `checkpoints_full/checkpoint_best.pt` - Best loss
- `checkpoints_full/student_final.pt` - Final model (after epoch 80)

---

## Next Steps (After Training)

1. **Evaluate student performance:**
   - Compare with teacher on validation set
   - Measure inference speed (should be 2-3× faster)
   - Check feature quality

2. **Export for deployment:**
   - Convert to TorchScript or ONNX
   - Optimize for Jetson Orin NX (INT8 quantization)
   - Target: <1s latency

3. **Integration:**
   - Replace teacher encoder in vggt-unified
   - Test with edge-mask and obj-mask decoders
   - Validate end-to-end pipeline

---

## References

- **DINOv2:** Meta AI's self-supervised ViT (facebookresearch/dinov2)
- **Teacher:** VGGT encoder (909M params, 24 layers, 1024 dim)
- **Target deployment:** Jetson Orin NX 16GB, <1s latency

---

**Last updated:** Aug 25, 2026  
**Training started:** Aug 25, 2026 ~6:00 UTC  
**Expected completion:** Aug 26, 2026 ~12:00 UTC
