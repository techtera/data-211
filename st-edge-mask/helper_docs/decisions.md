# Decisions

## 2026-08-XX — UNet++ Decoder with Deep Supervision

**Decision**: Use UNet++ decoder architecture with deep supervision over simpler alternatives.

**Context**: Need a decoder that can reconstruct fine edge details from 4 levels of student encoder features (1536-dim tokens reshaped to spatial maps).

**Alternatives Considered**:
- Simple FPN + conv head — fast but loses fine-grained spatial detail
- U-Net — good but limited skip connection topology
- UNet++ — nested dense skip connections, better multi-scale fusion
- DeepLabV3+ — heavier, ASPP module adds compute

**Reasoning**: Edge detection requires precise spatial localization. UNet++ provides dense nested skip connections that progressively fuse fine and coarse features. Deep supervision at intermediate nodes (x_0_1, x_0_2) provides auxiliary gradient paths that help training converge and prevent vanishing gradients in early layers.

**Tradeoffs**: More parameters than simple FPN, but still small relative to the frozen encoder. Slightly more complex forward pass.

**Expected Impact**: Better edge precision and recall than simpler decoders, especially for thin/fine edges.

---

## 2026-08-XX — Feature Projection Spatial Targets

**Decision**: Project student features to specific spatial sizes: 64@148x148, 128@74x74, 256@37x37, 512@downsampled.

**Context**: Student encoder outputs 1536-dim tokens at all 4 cached layers, all at the same spatial resolution (37x37 patches). Need a multi-scale feature pyramid for UNet++.

**Alternatives Considered**:
- All features at native 37x37 — no multi-scale information
- Fixed power-of-2 sizes — would require more aggressive interpolation
- Learned upsampling — more parameters, slower

**Reasoning**: 148x148 ≈ 4x the patch grid (captures fine detail), 74x74 ≈ 2x (mid-level), 37x37 = native (high-level semantic), downsampled = compressed context. This creates a natural FPN hierarchy from identical-resolution encoder outputs. GroupNorm(8) + SiLU chosen for stability with small batch sizes.

**Tradeoffs**: Interpolation to 148x148 introduces some smoothing. Downsampling the last level discards some spatial info.

---

## 2026-08-XX — Loss: BCE(0.5) + Dice(0.5) with Deep Supervision

**Decision**: Combine BCE and Dice loss equally, with deep supervision auxiliary losses weighted ds1=0.1, ds2=0.2, final=1.0.

**Context**: Edge detection is severely class-imbalanced (typically <5% edge pixels). Need a loss that handles this imbalance.

**Alternatives Considered**:
- BCE only — works but requires careful pos_weight tuning
- Focal loss — handles imbalance but adds hyperparameters (alpha, gamma)
- Dice only — directly optimizes overlap but unstable early in training
- BCE + Dice — complementary: BCE provides stable per-pixel gradients, Dice directly optimizes region overlap

**Reasoning**: BCE with dynamic positive weight (clamped to [5,25]) handles class imbalance at the pixel level. Dice loss directly optimizes the F1-like overlap metric. Equal weighting (0.5 each) found empirically stable. Deep supervision weights increase toward the final output (0.1, 0.2, 1.0) since the final prediction is what matters most.

**Tradeoffs**: Positive weight clamping at [5,25] may under-weight edges in extremely sparse images or over-weight in dense ones.

---

## 2026-08-XX — Early Stopping on F1 (Patience=15)

**Decision**: Use early stopping with patience=15 epochs, monitored on validation F1 score (not loss).

**Context**: Training for 100 epochs is expensive. Need to stop when the model stops improving on the metric that matters.

**Alternatives Considered**:
- No early stopping — wastes compute if model converges early
- Patience=5 — too aggressive, may stop during temporary plateaus
- Patience=20 — too lenient, wastes compute
- Monitor loss instead of F1 — loss can decrease while F1 stagnates

**Reasoning**: F1 score directly measures edge detection quality (harmonic mean of precision and recall). Patience=15 allows the cosine LR schedule to explore multiple regimes before giving up. Monitoring F1 rather than loss avoids the trap where loss improves via better calibration but F1 doesn't.

---

## 2026-08-XX — LR Schedule: 3e-4 with 5% Warmup + Cosine Decay

**Decision**: Use learning rate 3e-4 with linear warmup over 5% of total steps, followed by cosine decay to 0.

**Context**: Decoder is randomly initialized while encoder is frozen and pretrained. Need a schedule that allows aggressive initial learning but smooth convergence.

**Alternatives Considered**:
- Constant LR — no adaptation, risk of overshooting or stagnating
- Step decay — abrupt transitions
- 1e-4 — too conservative for randomly initialized decoder
- 1e-3 — too aggressive, causes instability

**Reasoning**: 3e-4 is higher than the encoder's KD training rate (1e-4) because the decoder starts from scratch and needs faster initial convergence. Warmup prevents gradient explosion from the random initialization. Cosine decay provides smooth convergence in later epochs.
