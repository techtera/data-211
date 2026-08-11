# Development Plan

## Implementation Order

Build inside-out. Validate tensor shapes at each step before connecting to the encoder. The encoder integration is the riskiest coupling point — test everything else first with synthetic inputs.

---

## Step 1: losses.py

### What to Build
- `EdgeLoss` class with WeightedBCE + Dice
- `compute_total_loss` function combining final + deep supervision outputs

### Implementation Details
- `pos_weight = clamp(neg / pos, 5, 25)` computed per-batch
- Dice uses sigmoid(logits), not raw logits
- epsilon = 1e-6 in Dice denominator
- Loss function takes raw logits (not sigmoid-applied)

### Test Criteria
- Feed synthetic logits and binary targets
- Verify gradient flows (non-zero .grad on logits)
- Verify pos_weight clamps correctly at boundaries
- Verify loss is non-negative
- Verify all-zero target doesn't produce NaN
- Verify all-one target doesn't produce NaN

---

## Step 2: feature_extractor.py

### What to Build
- `FeatureProjection` class (1x1 proj + spatial resize)
- `VGGTFeatureExtractor` class (extracts from aggregator + projects)

### Implementation Details
- 4 projection levels with configs:
  - Level 0: 2048→64, bilinear to 148x148, Conv3x3 smooth
  - Level 1: 2048→128, bilinear to 74x74, Conv3x3 smooth
  - Level 2: 2048→256, identity
  - Level 3: 2048→512, Conv3x3 stride=2
- GroupNorm(8) after every conv
- SiLU activation after every GroupNorm
- Feature extraction uses `torch.no_grad()` for encoder forward
- `.detach()` on extracted features before projection

### Test Criteria
- Feed random `[B*S, 2048, 37, 37]` through each projection
- Verify output shapes:
  - Level 0: [B*S, 64, 148, 148]
  - Level 1: [B*S, 128, 74, 74]
  - Level 2: [B*S, 256, 37, 37]
  - Level 3: [B*S, 512, 19, 19]
- Verify all outputs have requires_grad=True (decoder is trainable)
- Verify projections have correct parameter count

---

## Step 3: decoder.py — ConvBlock and Upsample

### What to Build
- `ConvBlock` class
- `Upsample` class
- `DeepSupervisionHead` class

### Implementation Details
- ConvBlock: Conv3x3 + GN8 + SiLU + Conv3x3 + GN8 + SiLU
- Upsample: F.interpolate(size=target) + Conv3x3 + GN8 + SiLU
- Upsample takes `target_size` as forward argument (not fixed)
- DeepSupervisionHead: Conv3x3(in→32) + SiLU + Conv1x1(32→1) + bilinear(518)

### Test Criteria
- ConvBlock: verify [B, 384, 74, 74] → [B, 128, 74, 74]
- Upsample: verify [B, 512, 19, 19] → [B, 256, 37, 37] with target_size=(37,37)
- Upsample: verify [B, 128, 74, 74] → [B, 64, 148, 148] with target_size=(148,148)
- DeepSupervisionHead: verify [B, 64, 148, 148] → [B, 1, 518, 518]

---

## Step 4: decoder.py — Full UNet++ Grid

### What to Build
- `UNetPPDecoder` class with all 6 upsample blocks, 6 ConvBlocks, 2 DS heads

### Implementation Details
- 6 Upsample blocks:
  - up_3_0: 512→256 (target 37x37)
  - up_2_0: 256→128 (target 74x74)
  - up_2_1: 256→128 (target 74x74)
  - up_1_0: 128→64 (target 148x148)
  - up_1_1: 128→64 (target 148x148)
  - up_1_2: 128→64 (target 148x148)
- 6 ConvBlocks:
  - conv_2_1: 512→256
  - conv_1_1: 256→128
  - conv_1_2: 384→128
  - conv_0_1: 128→64
  - conv_0_2: 192→64
  - conv_0_3: 256→64
- Dense skip concatenation at every node

### Test Criteria
- Feed 4 synthetic feature levels:
  - [B, 64, 148, 148]
  - [B, 128, 74, 74]
  - [B, 256, 37, 37]
  - [B, 512, 19, 19]
- Verify output x_0_3 shape = [B, 64, 148, 148]
- Verify ds1 shape = [B, 1, 518, 518]
- Verify ds2 shape = [B, 1, 518, 518]
- Verify no shape mismatches during concatenation
- Verify backward pass completes without error

---

## Step 5: refinement.py

### What to Build
- `EdgeRefinement` class (residual block)

### Implementation Details
- Input and output: same shape [B*S, 64, 148, 148]
- Residual: x + refine(x)
- refine = Conv3x3 + GN8 + SiLU + Conv3x3 + GN8 + SiLU

### Test Criteria
- Verify output.shape == input.shape
- Verify output != input (refine path contributes)
- Verify gradient flows through both branches

---

## Step 6: model.py — Full Pipeline Assembly

### What to Build
- `VGGTEdgeMask` class combining all modules

### Implementation Details
- Instantiate: feature_extractor, decoder, refinement, final_conv
- Forward: extract → decode → refine → conv1x1 → upsample → reshape
- Training mode returns (logits, ds1_logits, ds2_logits)
- Eval mode returns sigmoid(logits)
- Final reshape from [B*S, 1, 518, 518] to [B, S, 1, 518, 518]

### Test Criteria (without real VGGT)
- Mock feature extractor to return synthetic features
- Verify full forward pass produces [B, S, 1, 518, 518]
- Verify training mode returns 3 tensors
- Verify eval mode returns 1 tensor with values in [0, 1]

### Test Criteria (with real VGGT)
- Load VGGT model (random weights)
- Forward one image [1, 2, 3, 518, 518]
- Verify output shape [1, 2, 1, 518, 518]
- Verify encoder params have no grad
- Verify decoder params have grad after backward
- Verify loss.backward() completes

---

## Step 7: train.py

### What to Build
- Training loop with:
  - AdamW optimizer (lr=3e-4, weight_decay=0.01)
  - CosineAnnealing scheduler with 5% linear warmup
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping (max_norm=1.0)
  - Logging (loss, learning rate, edge pixel ratio)
  - Checkpointing (save best model by validation BF1)
  - Early stopping (patience on validation BF1)

### Implementation Details
- Encoder in eval mode always (model.feature_extractor.aggregator.eval())
- Call model.train() but encoder stays frozen
- Gradient clipping after scaler.unscale_()
- Log per-epoch: train_loss, val_loss, val_bf1, lr

### Test Criteria
- Overfit on 1 image: loss should approach 0
- Verify encoder weights unchanged after training
- Verify decoder weights changed after training
- Verify gradient clipping activates (log norm)

---

## Step 8: evaluate.py

### What to Build
- Boundary F1 at various thresholds
- ODS (best threshold across dataset)
- OIS (best threshold per image, averaged)
- Dice score
- Thin-edge recall (edges <= 2px width)

### Test Criteria
- Perfect prediction gives BF1 = 1.0
- All-zero prediction gives BF1 = 0.0
- Metrics are in expected ranges on synthetic data

---

## Validation Checkpoints

After each step, run the test criteria before proceeding. Do not move to the next step until the current step's tests pass.

| Step | Gate |
|------|------|
| 1 → 2 | Loss gradients verified, no NaN |
| 2 → 3 | All 4 projection output shapes correct |
| 3 → 4 | ConvBlock and Upsample shapes verified |
| 4 → 5 | Full UNet++ grid forward + backward pass clean |
| 5 → 6 | Refinement residual verified |
| 6 → 7 | End-to-end forward produces correct shape |
| 7 → 8 | Overfitting on 1 image confirmed |
