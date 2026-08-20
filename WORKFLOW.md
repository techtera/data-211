# VGGT Truncation Testing Workflow

Complete workflow to test if VGGT block truncation hurts quality.

## Setup

**What you need:**
- 1150 unlabeled RGB images (flat directory)
- `vggt_unified_fp16.pt` checkpoint at `/checkpoints/vggt_unified_fp16.pt`
- Access to the VM with GPU

**What you'll test:**
- Baseline: 24 blocks (FP16)
- Test 1: 20 blocks (FP16) 
- Test 2: 18 blocks (FP16)

---

## Complete Workflow

### Step 1: Baseline (24 blocks)

Run inference on baseline model and save predictions:

```bash
python run_inference_save.py \
    --checkpoint /checkpoints/vggt_unified_fp16.pt \
    --images_dir /path/to/unlabeled_images/ \
    --output_dir predictions/baseline_24blocks \
    --num_warmup 5 \
    --num_profile 50
```

**Output:**
```
predictions/baseline_24blocks/
├── predictions/*.npz       # 1150 prediction files
├── config.json             # Model config + latency
└── latency_stats.json      # Profiling results
```

**Expected time:** ~10-15 minutes (depends on GPU speed)

---

### Step 2: Test 20 Blocks

**2.1 Apply truncation:**
```bash
python apply_truncation.py --depth 20
```

This modifies:
- `encoder/aggregator.py`: depth=20, cached_layers=[4,11,16,19]
- `decoders/obj_mask/segformer_head.py`: intermediate_layer_idx=[4,11,16,19]

**2.2 Run inference:**
```bash
python run_inference_save.py \
    --checkpoint /checkpoints/vggt_unified_fp16.pt \
    --images_dir /path/to/unlabeled_images/ \
    --output_dir predictions/truncated_20blocks \
    --num_warmup 5 \
    --num_profile 50
```

**2.3 Compare with baseline:**
```bash
python compare_visual.py \
    --baseline_dir predictions/baseline_24blocks \
    --test_dir predictions/truncated_20blocks \
    --images_dir /path/to/unlabeled_images/ \
    --output_dir comparisons/24vs20blocks \
    --num_samples 50
```

**Output:**
```
comparisons/24vs20blocks/
├── visual_comparisons/
│   ├── 000_image1_objdiff5.2.png   # Largest differences first
│   ├── 001_image2_objdiff4.1.png
│   └── ...
└── comparison_summary.json         # Speedup + quality metrics
```

---

### Step 3: Test 18 Blocks

**3.1 Apply truncation:**
```bash
python apply_truncation.py --depth 18
```

**3.2 Run inference:**
```bash
python run_inference_save.py \
    --checkpoint /checkpoints/vggt_unified_fp16.pt \
    --images_dir /path/to/unlabeled_images/ \
    --output_dir predictions/truncated_18blocks \
    --num_warmup 5 \
    --num_profile 50
```

**3.3 Compare with baseline:**
```bash
python compare_visual.py \
    --baseline_dir predictions/baseline_24blocks \
    --test_dir predictions/truncated_18blocks \
    --images_dir /path/to/unlabeled_images/ \
    --output_dir comparisons/24vs18blocks \
    --num_samples 50
```

---

### Step 4: Review Results

**4.1 Check latency improvements:**
```bash
cat predictions/baseline_24blocks/latency_stats.json
cat predictions/truncated_20blocks/latency_stats.json
cat predictions/truncated_18blocks/latency_stats.json
```

**4.2 Check quality metrics:**
```bash
cat comparisons/24vs20blocks/comparison_summary.json | grep -A10 quality_comparison
cat comparisons/24vs18blocks/comparison_summary.json | grep -A10 quality_comparison
```

**4.3 Visual inspection:**
```bash
# Review side-by-side images (sorted by difference magnitude)
open comparisons/24vs20blocks/visual_comparisons/
open comparisons/24vs18blocks/visual_comparisons/
```

---

### Step 5: Make Decision

**Review these metrics:**

| Config | Mean Latency | Speedup | Obj Diff % | Edge Diff |
|--------|-------------|---------|------------|-----------|
| 24 blocks (baseline) | X ms | - | - | - |
| 20 blocks | Y ms | +A% | B% | C |
| 18 blocks | Z ms | +D% | E% | F |

**Decision criteria:**
- ✅ **Use 20 blocks if:** Speedup >10%, Obj diff <3%, visually acceptable
- ✅ **Use 18 blocks if:** Speedup >15%, Obj diff <5%, visually acceptable
- ❌ **Stay at 24 blocks if:** Quality degrades too much on visual inspection

---

### Step 6: Finalize

**If you choose truncation (e.g., 20 blocks):**
```bash
# Keep the truncated code
# Delete predictions to save space (optional)
rm -rf predictions/
```

**If quality degrades too much:**
```bash
# Restore original code
python apply_truncation.py --restore
```

---

## Visual Comparison Layout

Each comparison image shows:

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │ Baseline Obj│  Test Obj   │  ← Object masks (green overlay)
│             │             │             │
├─────────────┼─────────────┼─────────────┤
│  Obj Diff   │Baseline Edge│  Test Edge  │  ← Edge masks (white on black)
│ (heatmap)   │             │             │
├─────────────┼─────────────┼─────────────┤
│  Edge Diff  │   Empty     │   Empty     │  ← Edge difference heatmap
│ (heatmap)   │             │             │
└─────────────┴─────────────┴─────────────┘
```

**Heatmap colors:**
- Dark blue/black: No difference
- Yellow/red: Large difference

---

## Expected Results

### 20 Blocks
- **Speedup:** 11-13%
- **Quality loss:** <2% pixel difference
- **Risk:** Low

### 18 Blocks
- **Speedup:** 17-20%
- **Quality loss:** 3-5% pixel difference
- **Risk:** Moderate (needs visual validation)

---

## Troubleshooting

### "No module named 'model'"
```bash
cd /path/to/vggt-unified
python run_inference_save.py ...
```

### "Checkpoint not found"
Check the path:
```bash
ls -lh /checkpoints/vggt_unified_fp16.pt
```

### "CUDA out of memory"
The checkpoint is already FP16, so this shouldn't happen with 1150 images processed one at a time.

### "No common predictions found"
Make sure image names match between directories.

---

## Files Created

After running all steps:

```
vggt-unified/
├── predictions/
│   ├── baseline_24blocks/
│   │   ├── predictions/*.npz (1150 files)
│   │   ├── config.json
│   │   └── latency_stats.json
│   ├── truncated_20blocks/
│   │   └── ... (same structure)
│   └── truncated_18blocks/
│       └── ... (same structure)
│
├── comparisons/
│   ├── 24vs20blocks/
│   │   ├── visual_comparisons/ (50 images)
│   │   └── comparison_summary.json
│   └── 24vs18blocks/
│       └── ... (same structure)
│
└── encoder/aggregator.py.backup  (original file)
```

---

## Summary

**Total testing time:** ~1-2 hours (including review)

**What you get:**
1. Exact latency improvements (ms and %)
2. Pixel-wise quality metrics
3. Visual evidence of changes
4. Data to make informed decision

**No guessing, no overfitting concerns** - pure visual comparison on unseen data.
