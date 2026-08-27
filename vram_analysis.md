# VRAM Analysis: VGGT-1B vs Student Encoder

## Architecture Comparison

### VGGT-1B (Teacher)
- **Embed dim**: 1024
- **Transformer blocks**: 24 layers total
- **Parameters**: 909M
- **Output**: 2048-dim (1024 frame + 1024 global concatenated)
- **Tokens per image**: 1374 (5 special + 1369 patches @ 518×518)

### Student
- **Embed dim**: 768
- **Transformer blocks**: 18 frame + 18 global = 36 blocks (alternating)
- **Parameters**: 255M
- **Output**: 1536-dim (768 frame + 768 global concatenated)
- **Tokens per image**: 1374 (same patch structure)

---

## Parameter Memory (FP16 Inference)

### VGGT-1B
```
909M params × 2 bytes (FP16) = 1,818 MB = 1.78 GB
```

### Student
```
255M params × 2 bytes (FP16) = 510 MB = 0.50 GB
```

**Parameter reduction: 1.78 GB → 0.50 GB = 3.6× smaller ✓**

---

## Activation Memory (Forward Pass Only)

Activations depend on:
- Batch size
- Sequence length (1374 tokens)
- Hidden dimension
- Number of layers cached
- Intermediate tensors (attention, MLP)

### Batch Size = 1 (Single Image Inference)

#### VGGT-1B Activations:
```
1. Input tokens:
   1 × 1374 × 2048 × 2 bytes = 5.6 MB

2. Cached layer outputs (4 layers: [4,11,17,23]):
   4 × (1 × 1374 × 2048 × 2) = 22.5 MB

3. Attention intermediate:
   - Q, K, V projections: 1 × 1374 × 1024 × 2 × 3 = 8.5 MB per layer
   - Attention scores: 1 × 12 heads × 1374 × 1374 × 2 = 45.4 MB per layer
   - Peak per layer: ~54 MB
   - For 24 layers (sequential, can reuse): ~54 MB peak

4. MLP intermediate (4× expansion):
   - 1 × 1374 × 4096 × 2 = 11.3 MB per layer
   - For 24 layers (sequential, can reuse): ~11.3 MB peak

5. Gradient checkpointing (if used): 0 MB (inference mode)

Total activations: ~90-120 MB
```

#### Student Activations:
```
1. Input tokens:
   1 × 1374 × 1536 × 2 bytes = 4.2 MB

2. Cached layer outputs (4 layers: [3,8,13,17]):
   4 × (1 × 1374 × 1536 × 2) = 16.9 MB

3. Attention intermediate:
   - Q, K, V projections: 1 × 1374 × 768 × 2 × 3 = 6.3 MB per layer
   - Attention scores: 1 × 12 heads × 1374 × 1374 × 2 = 45.4 MB per layer
   - Peak per layer: ~52 MB
   - For 18 alternating layers (sequential): ~52 MB peak

4. MLP intermediate (4× expansion):
   - 1 × 1374 × 3072 × 2 = 8.4 MB per layer
   - For 18 layers (sequential, can reuse): ~8.4 MB peak

5. Gradient checkpointing: 0 MB (inference mode)

Total activations: ~75-100 MB
```

**Single image inference:**
- VGGT-1B: 1.78 GB (params) + 0.12 GB (activations) = **1.9 GB**
- Student: 0.50 GB (params) + 0.10 GB (activations) = **0.6 GB**
- **Reduction: 1.9 GB → 0.6 GB = 3.2× smaller ✓**

---

### Batch Size = 4 (Typical Decoder Training)

#### VGGT-1B Activations:
```
1. Input tokens:
   4 × 1374 × 2048 × 2 = 22.5 MB

2. Cached layer outputs (4 layers):
   4 × (4 × 1374 × 2048 × 2) = 90 MB

3. Attention intermediate:
   - Q, K, V: 4 × 1374 × 1024 × 2 × 3 = 33.8 MB per layer
   - Attention scores: 4 × 12 × 1374 × 1374 × 2 = 181.5 MB per layer
   - Peak per layer: ~215 MB
   - For 24 layers (sequential): ~215 MB peak

4. MLP intermediate:
   - 4 × 1374 × 4096 × 2 = 45 MB per layer
   - For 24 layers (sequential): ~45 MB peak

Total activations: ~450-500 MB
```

#### Student Activations:
```
1. Input tokens:
   4 × 1374 × 1536 × 2 = 16.9 MB

2. Cached layer outputs (4 layers):
   4 × (4 × 1374 × 1536 × 2) = 67.5 MB

3. Attention intermediate:
   - Q, K, V: 4 × 1374 × 768 × 2 × 3 = 25.3 MB per layer
   - Attention scores: 4 × 12 × 1374 × 1374 × 2 = 181.5 MB per layer
   - Peak per layer: ~207 MB
   - For 18 layers (sequential): ~207 MB peak

4. MLP intermediate:
   - 4 × 1374 × 3072 × 2 = 33.7 MB per layer
   - For 18 layers (sequential): ~33.7 MB peak

Total activations: ~350-400 MB
```

**Batch=4 inference:**
- VGGT-1B: 1.78 GB + 0.50 GB = **2.28 GB**
- Student: 0.50 GB + 0.40 GB = **0.90 GB**
- **Reduction: 2.28 GB → 0.90 GB = 2.5× smaller ✓**

---

## Full Decoder Training VRAM

During decoder training, we also need:
- Decoder parameters + gradients + optimizer states
- Batch data
- Loss computation

### Edge Decoder Training (Batch=4)

#### With VGGT-1B:
```
Encoder (frozen, FP16):
  - Parameters: 1.78 GB
  - Activations: 0.50 GB

Edge Decoder (trainable, FP32):
  - Parameters: 10M × 4 = 40 MB
  - Gradients: 10M × 4 = 40 MB
  - Optimizer (AdamW, 2 states): 10M × 8 = 80 MB
  - Activations: ~500 MB

Batch data:
  - 4 × 3 × 518 × 518 × 4 bytes = 12.9 MB

Total: 1.78 + 0.50 + 0.04 + 0.04 + 0.08 + 0.50 + 0.01 = 2.95 GB
```

#### With Student:
```
Encoder (frozen, FP16):
  - Parameters: 0.50 GB
  - Activations: 0.40 GB

Edge Decoder (trainable, FP32):
  - Parameters: 40 MB
  - Gradients: 40 MB
  - Optimizer: 80 MB
  - Activations: ~500 MB

Batch data: 12.9 MB

Total: 0.50 + 0.40 + 0.04 + 0.04 + 0.08 + 0.50 + 0.01 = 1.57 GB
```

**Edge decoder training reduction: 2.95 GB → 1.57 GB = 1.9× smaller**

---

### Object Decoder Training (Batch=4)

#### With VGGT-1B:
```
Encoder (frozen, FP16):
  - Parameters: 1.78 GB
  - Activations: 0.50 GB

Object Decoder (trainable, FP32):
  - SegFormer decoder: ~15M params
  - Parameters: 15M × 4 = 60 MB
  - Gradients: 60 MB
  - Optimizer: 15M × 8 = 120 MB
  - Activations: ~800 MB

Batch data: 12.9 MB

Total: 1.78 + 0.50 + 0.06 + 0.06 + 0.12 + 0.80 + 0.01 = 3.33 GB
```

#### With Student:
```
Encoder (frozen, FP16):
  - Parameters: 0.50 GB
  - Activations: 0.40 GB

Object Decoder (trainable, FP32):
  - Parameters: 60 MB
  - Gradients: 60 MB
  - Optimizer: 120 MB
  - Activations: ~800 MB

Batch data: 12.9 MB

Total: 0.50 + 0.40 + 0.06 + 0.06 + 0.12 + 0.80 + 0.01 = 1.95 GB
```

**Object decoder training reduction: 3.33 GB → 1.95 GB = 1.7× smaller**

---

## Summary Table

| Scenario | VGGT-1B | Student | Reduction |
|----------|---------|---------|-----------|
| **Parameters only** | 1.78 GB | 0.50 GB | **3.6×** |
| **Single image inference** | 1.90 GB | 0.60 GB | **3.2×** |
| **Batch=4 inference** | 2.28 GB | 0.90 GB | **2.5×** |
| **Edge decoder training** | 2.95 GB | 1.57 GB | **1.9×** |
| **Object decoder training** | 3.33 GB | 1.95 GB | **1.7×** |

---

## Conclusion

**Yes, there IS a significant VRAM reduction:**

1. **Pure inference** (single image): **3.2× reduction** (1.9 GB → 0.6 GB)
2. **Decoder training**: **~2× reduction** (3 GB → 1.6-2 GB)

The reduction is **less during training** because:
- Decoder parameters/gradients/optimizer states are the same size regardless of encoder
- These decoder overheads (0.5-1 GB) become a larger portion of total VRAM
- But encoder savings (1.28 GB) still provide substantial benefit

**For Jetson Orin NX (16 GB):**
- VGGT-1B unified: ~4-5 GB (tight fit)
- Student unified: ~2-3 GB (comfortable)
- **Deployment is definitely feasible with student encoder ✓**

---

## Real-World Verification

To verify actual VRAM usage during decoder training:

```python
import torch

# During training, add this:
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

Expected values:
- Edge decoder training: 1.5-2.0 GB allocated
- Object decoder training: 2.0-2.5 GB allocated
