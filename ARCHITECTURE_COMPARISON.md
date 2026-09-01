# Architecture Comparison: Teacher vs Student Decoders

## ✅ Edge Decoder Architecture - IDENTICAL

### Components (Same for Both):

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **FeatureProjection** | Conv2d → GroupNorm(8) → SiLU | ✅ IDENTICAL |
| **UNet++ Decoder** | 6 nodes, channels (64,128,256,512) | ✅ IDENTICAL |
| **Edge Refinement** | Residual Conv3x3 block | ✅ IDENTICAL |
| **Final Conv** | Conv2d(64→1) → Bilinear upsample | ✅ IDENTICAL |
| **Loss** | 0.5×BCE + 0.5×Dice, deep supervision | ✅ IDENTICAL |

### Only Differences:

| Feature | Teacher | Student | Impact |
|---------|---------|---------|--------|
| **Encoder output dim** | 2048 | 1536 | Input to first Conv2d |
| **Layer indices** | [4,11,17,23] | [3,8,13,17] | Which encoder layers used |
| **Feature scale** | Normalized (std~2) | Unnormalized (std~27k) | ⚠️ **ROOT CAUSE** |

---

## ✅ Obj Decoder Architecture - IDENTICAL

### Components (Same for Both):

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **LayerNorm** | nn.LayerNorm(dim_in) - 1 shared | ✅ IDENTICAL |
| **Projection** | Conv2d(dim_in→channels) | ✅ IDENTICAL |
| **SegFormer Decoder** | MLP + Linear fuse | ✅ IDENTICAL |
| **Loss** | CrossEntropy | ✅ IDENTICAL |

### Only Differences:

| Feature | Teacher | Student | Impact |
|---------|---------|---------|--------|
| **Encoder output dim** | 2048 | 1536 | Input to LayerNorm |
| **Layer indices** | [4,11,17,23] | [3,8,13,17] | Which encoder layers used |
| **Feature scale** | Normalized (std~2) | Unnormalized (std~27k) | ⚠️ **ROOT CAUSE** |

---

## 🔍 Key Insight: GroupNorm vs Input Scale

### Teacher Pipeline:
```
Encoder → 2048-dim (mean~0, std~2)
    ↓
Conv2d(2048→64) → reasonable weight magnitudes
    ↓
GroupNorm(8, 64) → normalizes output
    ↓
SiLU → UNet++ decoder
```

### Student Pipeline:
```
Encoder → 1536-dim (mean~-1077, std~27k) ⚠️
    ↓
Conv2d(1536→64) → HUGE input magnitudes!
    ↓  Random init weights × huge inputs = saturated outputs
    ↓
GroupNorm(8, 64) → tries to normalize but Conv already saturated
    ↓
SiLU → UNet++ decoder (gets bad features)
```

---

## 🚨 The Problem: Conv2d Input Scale

**GroupNorm can't save us** because the damage happens BEFORE it:

1. Conv2d has randomly initialized weights (mean~0, std~0.01)
2. Student features are 10,000× larger than expected
3. Conv2d output = weights × features = **HUGE values**
4. SiLU activation saturates (values >10 → ~10, values <-10 → ~0)
5. GroupNorm normalizes already-saturated values
6. Information lost, decoder can't learn

**Teacher works** because:
- Conv2d input scale is reasonable (std~2)
- Conv2d output scale is reasonable
- SiLU doesn't saturate
- GroupNorm further stabilizes
- Decoder learns normally

---

## ✅ Architecture is IDENTICAL - Problem is INPUT SCALE

**Conclusion:**
- Decoder architectures are EXACTLY the same ✅
- Difference is ONLY in encoder output scale ⚠️
- This is an ENCODER problem, not DECODER problem
- Can't fix at decoder level without adding normalization BEFORE Conv2d

---

## 📊 Verification: Compare Layer Counts

### Edge Decoder Parameter Counts:

| Component | Teacher | Student | Match? |
|-----------|---------|---------|--------|
| Feature Projections | 4,514,688 | 3,393,024 | Different (input dim) |
| UNet++ Decoder | 5,278,018 | 5,278,018 | ✅ IDENTICAL |
| Edge Refinement | 74,112 | 74,112 | ✅ IDENTICAL |
| Final Conv | 65 | 65 | ✅ IDENTICAL |
| **Total Trainable** | 9,866,883 | 8,745,219 | Different (projections only) |

The projection layer difference is EXPECTED (2048→64 vs 1536→64 requires different number of weights).

---

## 🎯 Final Answer

**Q: Is decoder architecture identical to teacher?**  
**A: YES** - everything except input dimension (which changes projection layer size)

**Q: Why doesn't student decoder train well?**  
**A: Encoder output scale is 10,000× too large**

**Q: Can we fix it at decoder level?**  
**A: NO** - need to normalize BEFORE first Conv2d, which requires:
- Adding LayerNorm before projections (we tried, made it worse)
- OR fixing encoder to output normalized features
- OR using KD projection layers (don't have them)
- OR retraining encoder with output norm (2 weeks)

---

**Recommendation:** Use teacher-based models. Student encoder needs fixing at the source.
