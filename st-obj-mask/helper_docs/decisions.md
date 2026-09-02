# Decisions

## 2026-09-02 — Added Warmup + Cosine LR Scheduler

**Decision**: Add linear warmup (5% of steps) followed by cosine decay to 0, matching the edge-mask training setup.

**Context**: Object mask training used a constant LR=1e-4 for all 100 epochs. Edge-mask already had warmup + cosine and showed better convergence behavior.

**Alternatives Considered**:
- Keep constant LR — simple but prone to plateau stagnation
- StepLR — arbitrary step boundaries
- ReduceOnPlateau — reactive rather than proactive

**Reasoning**: Cosine decay is proven in the edge-mask pipeline. Warmup prevents large early gradients from destabilizing the randomly-initialized decoder. The cosine curve naturally reduces LR in later epochs, enabling fine-grained refinement.

**Tradeoffs**: Slightly more complex training loop. Scheduler state must be saved in checkpoints for resume.

**Expected Impact**: Better convergence, potential for lower final loss and higher mIoU.

**Risks**: Warmup fraction (5%) may need tuning if dataset is very small (fewer total steps).

---

## 2026-09-02 — Added Early Stopping (Patience=15)

**Decision**: Stop training if val mIoU does not improve for 15 consecutive epochs.

**Context**: Object mask trained all 100 epochs regardless of convergence. Edge-mask already had patience=15.

**Alternatives Considered**:
- No early stopping — wastes compute if model converges early
- Patience=5 — too aggressive, may stop during temporary plateaus
- Patience=25 — too conservative, doesn't save much compute

**Reasoning**: 15 epochs is sufficient to distinguish genuine plateau from temporary fluctuation. Saves compute and prevents overfitting when the model has converged.

**Tradeoffs**: May stop training before a late-stage breakthrough (unlikely with cosine schedule).

---

## Architecture — SegFormer Decoder

**Decision**: Use SegFormer-based decoder (DPTHead) for object mask prediction.

**Context**: Need a lightweight decoder that converts multi-scale ViT features into segmentation masks.

**Reasoning**: SegFormer decoder is proven for semantic segmentation. The DPTHead wraps it with a feature extraction pipeline that builds a 4-level pyramid from student cached layers [3, 8, 13, 17]:
- c1: [B*S, 256, 84, 148] (stride 4)
- c2: [B*S, 512, 42, 74] (stride 8)
- c3: [B*S, 1024, 21, 37] (stride 16)
- c4: [B*S, 1024, 11, 19] (stride 32)

Input dimension is 1536 (768 frame + 768 global concatenated from student encoder).

**Tradeoffs**: Larger channel counts (1024) at deeper levels increase compute. Could reduce to 512 if latency is an issue.

---

## Loss — CrossEntropy + Dice (Equal Weight)

**Decision**: Use CrossEntropy + Dice loss with equal weighting (1.0 each).

**Context**: Binary segmentation (background vs object). Need balanced loss that handles class imbalance.

**Reasoning**: CrossEntropy provides pixel-level classification gradient. Dice loss provides region-level overlap gradient, naturally handling class imbalance. Equal weighting is the standard starting point.

**Tradeoffs**: May need to adjust weighting if one dominates during training.
