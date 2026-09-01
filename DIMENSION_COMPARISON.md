# Student Encoder Dimension Comparison

## Configuration Options (18 layers)

### **Option A: Current (768-dim)**
```
Depth: 18 layers
Dimension: 768
Heads: 12 (64-dim per head)
Output: 1536-dim (768 frame + 768 global)
```

### **Option B: Upgraded (1024-dim)**
```
Depth: 18 layers
Dimension: 1024
Heads: 16 (64-dim per head)
Output: 2048-dim (1024 frame + 1024 global) ✅ Matches teacher!
```

---

## Parameter Count

### **Option A (768-dim):**

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| Patch Embedding | 413,280 | 3×14×14×768 + 768 |
| Special Tokens | 3,840 | 5×768 |
| Position Embedding | 0 | (RoPE - no params) |
| **Per Block** | **7,087,872** | See breakdown below |
| Frame Blocks (18×) | 127,581,696 | 18×7,087,872 |
| Global Blocks (18×) | 127,581,696 | 18×7,087,872 |
| **TOTAL** | **255,580,512** | **~255M params** |

**Per Block Breakdown (768-dim):**
- QKV projection: 1,769,472 (768×768×3)
- Attention output: 589,824 (768×768)
- MLP layer 1: 2,359,296 (768×3072)
- MLP layer 2: 2,359,296 (3072×768)
- LayerNorm ×2: 3,072 (768×2×2)
- LayerScale ×2: 1,536 (768×2)
- **Total: 7,087,872**

---

### **Option B (1024-dim):**

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| Patch Embedding | 602,112 | 3×14×14×1024 + 1024 |
| Special Tokens | 5,120 | 5×1024 |
| Position Embedding | 0 | (RoPE - no params) |
| **Per Block** | **12,587,008** | See breakdown below |
| Frame Blocks (18×) | 226,566,144 | 18×12,587,008 |
| Global Blocks (18×) | 226,566,144 | 18×12,587,008 |
| **TOTAL** | **453,739,520** | **~454M params** |

**Per Block Breakdown (1024-dim):**
- QKV projection: 3,145,728 (1024×1024×3)
- Attention output: 1,048,576 (1024×1024)
- MLP layer 1: 4,194,304 (1024×4096)
- MLP layer 2: 4,194,304 (4096×1024)
- LayerNorm ×2: 4,096 (1024×2×2)
- LayerScale ×2: 2,048 (1024×2)
- **Total: 12,587,008**

---

## Comparison Summary

| Metric | 768-dim | 1024-dim | Ratio |
|--------|---------|----------|-------|
| **Parameters** | 255M | 454M | **1.78×** |
| **Model Size (FP16)** | 511 MB | 908 MB | 1.78× |
| **Model Size (FP32)** | 1022 MB | 1816 MB | 1.78× |
| **Output Dimension** | 1536 | **2048** ✅ | 1.33× |

---

## Inference Speed Impact

### **Computation (FLOPs):**

**Attention (per token):**
- 768-dim: O(N² × 768) + O(N × 768²)
- 1024-dim: O(N² × 1024) + O(N × 1024²)
- **Ratio: 1.33× more compute**

**MLP (per token):**
- 768-dim: 768 × 3072 × 2 = 4.7M FLOPs
- 1024-dim: 1024 × 4096 × 2 = 8.4M FLOPs
- **Ratio: 1.78× more compute**

**Overall per block:**
- **~1.6× more FLOPs**

### **Memory Bandwidth:**

**Activations (batch=1, sequence=1374):**
- 768-dim: 1374 × 768 × 4 bytes = 4.2 MB per layer
- 1024-dim: 1374 × 1024 × 4 bytes = 5.6 MB per layer
- **Ratio: 1.33× more memory**

### **Estimated Inference Speed:**

| Metric | 768-dim | 1024-dim | Change |
|--------|---------|----------|--------|
| **Forward pass time** | ~50ms | ~75-80ms | **+50-60% slower** |
| **GPU Memory** | ~3-4 GB | ~4-6 GB | +1.5-2 GB |
| **Throughput (imgs/sec)** | ~20 | ~13-15 | -30-35% |

---

## Training Impact

| Metric | 768-dim | 1024-dim | Change |
|--------|---------|----------|--------|
| **Training time/epoch** | ~23 min | ~35-40 min | **+50-70% longer** |
| **GPU Memory** | 60-70 GB/GPU | 75-80 GB/GPU (tight!) | +15-20 GB |
| **Total KD training** | ~30 hours | **~45-50 hours** | +15-20 hours |
| **Batch size** | 64 (128 effective) | 48-56 (may need to reduce) | -15-20% |

---

## Decoder Impact

### **Option A (768-dim → 1536 output):**

❌ **Need to change decoder input dimension**

**Edge decoder changes:**
```python
# Change projections from:
FeatureProjection(2048, 64, ...) 
# To:
FeatureProjection(1536, 64, ...)
```

**Obj decoder changes:**
```python
# Change:
dim_in = 2048
# To:
dim_in = 1536
```

⚠️ **Decoders would need retraining** (~15 hours)

---

### **Option B (1024-dim → 2048 output):**

✅ **NO decoder changes needed!**

**Edge decoder:**
```python
FeatureProjection(2048, 64, ...)  # Already expects 2048 ✅
```

**Obj decoder:**
```python
dim_in = 2048  # Already expects 2048 ✅
```

✅ **Can use existing pretrained decoder weights!**

Or if retraining: exact same architecture as teacher decoders!

---

## Cost-Benefit Analysis

### **Option A: Keep 768-dim**

**Pros:**
- ✅ Smaller model (255M params)
- ✅ Faster inference (~50ms)
- ✅ Less GPU memory
- ✅ Faster training (~30h)

**Cons:**
- ❌ Output 1536-dim (mismatch with teacher 2048)
- ❌ Decoders need dimension changes
- ❌ Decoders need retraining (~15h)
- ❌ Still has scale mismatch issue
- ❌ **Total time: 30h encoder + 15h decoders = 45h**

---

### **Option B: Upgrade to 1024-dim**

**Pros:**
- ✅ Output 2048-dim (**matches teacher!**)
- ✅ Decoders need ZERO changes
- ✅ Can potentially use pretrained decoder weights
- ✅ Higher capacity (better quality)
- ✅ Exact architecture match (easier debugging)
- ✅ **Total time: 45h encoder only**

**Cons:**
- ⚠️ Larger model (454M vs 255M)
- ⚠️ Slower inference (+50-60%)
- ⚠️ More GPU memory (+20GB)
- ⚠️ Longer training (+15h)

---

## Recommendation

### **Go with 1024-dim (Option B)**

**Why:**

1. **Time is similar:**
   - 768-dim: 30h encoder + 15h decoders = **45h total**
   - 1024-dim: 45h encoder + 0h decoders = **45h total**

2. **Architecture alignment:**
   - Output matches teacher (2048-dim)
   - Decoders are exact architectural match
   - No dimension mismatch issues

3. **Quality:**
   - Higher capacity (454M vs 255M)
   - Better feature learning
   - More room for performance

4. **Simplicity:**
   - One training job (encoder only)
   - No decoder modifications
   - Cleaner codebase

5. **GPU memory is OK:**
   - 75-80GB per GPU (you have 2×A100 80GB) ✅
   - Might need to reduce batch size 64→56 (minor)

**Trade-off:** ~50% slower inference (75ms vs 50ms)
- Still acceptable for most use cases
- Can optimize later with quantization

---

## Final Recommendation

**Use 1024-dim (Option B)** with:
- 18 layers
- 1024-dim
- 16 heads
- Output: 2048-dim
- Params: 454M
- Training: ~45-50 hours

**Benefits:**
- ✅ Matches teacher output dimension
- ✅ No decoder changes
- ✅ Better quality
- ✅ Cleaner solution

**Accept the trade-offs:**
- ⚠️ 50% slower inference (still fast enough)
- ⚠️ 454M params vs 255M
- ⚠️ +15h training time
