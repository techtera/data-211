# Decisions

## 2026-08-XX — Student Architecture: 768-dim, 18 Layers

**Decision**: Use 768-dim embedding with 18 transformer layers for student encoder.

**Context**: Need to compress 909M teacher (1024-dim, 24 layers) to fit on Orin NX 16GB.

**Alternatives Considered**:
- 512-dim, 12 layers (~85M params) — too aggressive compression
- 768-dim, 12 layers (~170M params) — reasonable but fewer layers to match teacher
- 1024-dim, 18 layers (~450M params) — too large for deployment target

**Reasoning**: 768-dim with 18 layers gives 255M params, ~3.6x compression. 18 layers allows 4 well-spaced cached layers [3, 8, 13, 17] to align with teacher's [4, 11, 17, 23].

**Tradeoffs**: Slightly larger than minimal viable student, but better feature quality.

**Expected Impact**: Sufficient quality for decoder tasks, fits in Orin NX memory budget.

**Risks**: May still be too large for INT8 TRT on Orin NX after full pipeline assembly.

---

## 2026-08-XX — DINOv2-Large Initialization Over DINOv2-Base

**Decision**: Initialize student from DINOv2 ViT-Large (1024-dim) with truncation projection, not DINOv2-Base (768-dim) with direct copy.

**Context**: Student is 768-dim. DINOv2-Base is 768-dim (exact match). DINOv2-Large is 1024-dim (needs projection).

**Alternatives Considered**:
- DINOv2-Base direct copy — simpler, exact dimension match
- DINOv2-Large with learned projection — more complex, better features
- Random initialization — baseline

**Reasoning**: DINOv2-Large has significantly better features than Base. Truncation (take first 768 of 1024) preserves most information in principal components. Empirically produced better convergence.

**Tradeoffs**: Truncation loses some information from dimensions 769-1024. But DINOv2-Large's first 768 dims carry more information than all 768 dims of DINOv2-Base.

---

## 2026-08-XX — Loss: MSE 70% + Cosine 30%

**Decision**: Use weighted combination of MSE (magnitude matching) and cosine similarity (direction matching) for distillation loss.

**Context**: Need to match both the magnitude and direction of teacher features.

**Alternatives Considered**:
- MSE only — misses directional information
- Cosine only — misses magnitude information
- L1 loss — less sensitive to outliers but slower convergence
- CKA loss — computationally expensive

**Reasoning**: MSE captures magnitude differences, cosine captures angular alignment. 70:30 ratio emphasizes magnitude (more important for downstream tasks) while still enforcing directional consistency.

---

## 2026-08-XX — Token Sampling: 133/1374 Tokens

**Decision**: Sample 5 special + 128 random patch tokens per step instead of using all 1374.

**Context**: Full token distillation requires O(B × S × 1374 × 2048) memory per layer, which OOMs.

**Alternatives Considered**:
- All tokens (no sampling) — OOM on A100
- 64 tokens — may miss spatial patterns
- 256 tokens — still memory-heavy
- Structured sampling (grid) — deterministic but biased

**Reasoning**: 128 patches + 5 special = 133 tokens gives 90% memory reduction. Random sampling ensures all spatial locations are covered over many steps. Special tokens always kept for semantic anchor.

---

## 2026-08-XX — Progressive Layer Weights [1.0, 1.5, 2.0, 2.5]

**Decision**: Weight later cached layers more heavily in the loss.

**Context**: Later layers produce higher-level semantic features more critical for downstream tasks.

**Reasoning**: Layer 17/23 (final cached) contains the most task-relevant features. Weighting it 2.5x vs 1.0x for early layers ensures the student prioritizes getting the final representations right.

---

## 2026-08-XX — DDP Over DataParallel

**Decision**: Use DistributedDataParallel (DDP) via `torchrun` instead of `nn.DataParallel`.

**Context**: Multi-GPU training needed for reasonable training time.

**Alternatives Considered**:
- DataParallel — simpler but less efficient, GIL bottleneck
- DDP — more efficient, better scaling, standard practice

**Reasoning**: DDP is strictly better for multi-GPU: one process per GPU, no GIL contention, gradient reduction is overlapped with backward pass.
