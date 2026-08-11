# Risk Assessment

## Risks Already Identified and Addressed

### Risk 1: Upsample Shape Mismatch (19x19 → 37x37)

**Status**: Fixed in design.

**Problem**: Using `F.interpolate(scale_factor=2)` on a 19x19 tensor produces 38x38, not 37x37. Concatenation with Level 2 (37x37) would crash.

**Root Cause**: 37 is odd. `Conv3x3(stride=2, padding=1)` on 37x37 → `floor((37+2-3)/2)+1 = 19`. Then 19×2 = 38 ≠ 37.

**Solution**: All Upsample modules use `F.interpolate(x, size=(H_target, W_target))` with explicit target dimensions. Target size is taken from the corresponding level's spatial dimensions at runtime.

**Verified**: test_feature_extraction.py confirms all concatenation sizes align.

---

### Risk 2: Autograd Graph on Frozen Encoder

**Status**: Fixed in design.

**Problem**: Without `torch.no_grad()` around the encoder forward pass, PyTorch builds an autograd graph for encoder activations even if `requires_grad=False`. This wastes significant memory.

**Solution**: 
- Wrap encoder forward in `torch.no_grad()`
- Call `.detach()` on feature tensors before passing to projections
- Set encoder to `.eval()` mode permanently

---

### Risk 3: S-Dimension Handling

**Status**: Addressed in design.

**Problem**: VGGT processes multi-view inputs `[B, S, 3, 518, 518]`. Features come out as `[B, S, 1369, 2048]`. The decoder operates on `[B*S, C, H, W]`.

**Solution**: 
- Flatten to `B*S` before decoder processing
- Reshape back to `[B, S, ...]` only at the final output
- All intermediate operations use batch dimension `B*S`

---

### Risk 4: Deep Supervision Over-Constraining Intermediates

**Status**: Addressed via weight selection.

**Problem**: Original weights (0.25/0.25/0.50) gave 50% of gradient signal to auxiliary heads. X(0,1) has only seen 2 encoder levels and cannot produce high-quality edge maps. Forcing it constrains intermediate features to be "edge-map-like" prematurely.

**Solution**: Weights set to (0.1/0.2/1.0). Final head receives 77% of gradient signal. Auxiliary heads provide sufficient gradient flow without constraining intermediate representations.

---

### Risk 5: pos_weight Too Aggressive with Dice

**Status**: Addressed via clamping range.

**Problem**: pos_weight=50 combined with Dice loss creates double correction for class imbalance, producing thick/over-predicted edges and halo artifacts.

**Solution**: Clamp range (5, 25). Combined with 50% Dice loss, effective edge weighting stays in the 12-15x range — sufficient to prevent collapse without causing over-prediction.

---

## Known Assumptions

### Assumption 1: Synthetic Pyramid Validity

**What we assume**: Assigning early transformer layers to fine spatial levels and deep layers to coarse levels is valid, even though all features start at 37x37.

**Evidence supporting**: DPT (Ranftl et al., 2021) uses this exact approach. VGGT's own DPT head uses this in its codebase for depth estimation.

**What could go wrong**: Level 0 features (from layer 4, upsampled to 148x148) may be less informative than expected. The decoder might learn to mostly ignore them.

**Monitoring**: During training, check gradient magnitudes at Level 0 projection. If consistently near zero, the level is not contributing.

---

### Assumption 2: Edge Quality from Coarse Encoder

**What we assume**: 37x37 patch features contain enough information to reconstruct 518x518 edge maps with thin, continuous boundaries.

**Evidence supporting**: DPT produces pixel-level depth from the same features. The decoder's job is reconstruction, not the encoder's.

**What could go wrong**: Edges thinner than 1 patch (14px) may not be well-represented in encoder features at any layer.

**Monitoring**: Check thin-edge recall metric. If significantly worse than thick-edge recall, encoder features may be limiting.

---

### Assumption 3: GroupNorm Stability

**What we assume**: GroupNorm(8) works well with small batch sizes and all channel counts used.

**Evidence supporting**: All channel counts (64, 128, 256, 512, 32) are divisible by 8.

**What could go wrong**: With very small effective batch sizes (B*S=1), GroupNorm may still have variance issues in early training.

**Monitoring**: Watch for loss spikes in first 100 iterations.

---

## Validation Checkpoints

| Checkpoint | What to Verify | When |
|-----------|---------------|------|
| Feature shapes | All 4 levels produce expected spatial dimensions | After Step 2 |
| Concatenation | No runtime shape errors in UNet++ grid | After Step 4 |
| Gradient flow | Encoder: zero grad, Decoder: non-zero grad | After Step 6 |
| Loss sanity | Loss decreases on 1-image overfit | After Step 7 |
| No collapse | Predictions are not all-zero after 100 iterations | Early training |
| Edge quality | Predictions show visible edge structure by epoch 10 | Early training |
| Convergence | Validation BF1 improves over first 50 epochs | Mid training |

---

## Failure Modes

### Mode 1: All-Zero Collapse

**Symptoms**: Model predicts all zeros (background). Loss plateaus at a fixed value equal to the BCE of predicting 0 for all pixels.

**Cause**: Insufficient positive-class weighting, or learning rate too low for early gradient signal.

**Recovery**: 
- Verify pos_weight is being computed correctly (should be 5-25)
- Check that Dice loss gradient is non-zero
- Try increasing lr for first 1000 steps

---

### Mode 2: Thick/Blurry Edges

**Symptoms**: Model predicts edges wider than ground truth (2-3px instead of 1px). High recall, low precision.

**Cause**: pos_weight too high, or insufficient spatial resolution in decoder.

**Recovery**:
- Lower pos_weight upper clamp
- Verify that bilinear upsample to 518x518 isn't blurring
- Check edge refinement block is contributing (compare with/without)

---

### Mode 3: Slow Convergence

**Symptoms**: Loss decreases very slowly. No visible edge structure in predictions until epoch 50+.

**Cause**: ViT features lack local inductive bias that CNN features provide. Conv3x3 blocks need more iterations to adapt.

**Recovery**:
- Expected behavior — give it 100+ epochs
- Can increase lr slightly (5e-4)
- Can increase batch size for more stable gradients

---

### Mode 4: Training Instability (Loss Spikes)

**Symptoms**: Sudden large increases in loss, followed by recovery or divergence.

**Cause**: Large gradient norms from sparse edge signal, or Dice loss instability on near-empty masks.

**Recovery**:
- Verify gradient clipping is active (max_norm=1.0)
- Check epsilon in Dice denominator (must be 1e-6, not 0)
- Reduce learning rate

---

### Mode 5: Level 0 Not Contributing

**Symptoms**: Removing Level 0 from the decoder makes no difference in output quality.

**Cause**: Layer 4 features upsampled to 148x148 are too smooth/interpolated to provide useful local detail.

**Recovery**:
- Acceptable behavior for v1 — the other 3 levels carry the information
- Consider collapsing to 3-level decoder in v2 (not in scope)

---

## Debugging Checklist

Use this ordered checklist when encountering issues during training:

```
1. [ ] Check shapes: print all tensor shapes at each module boundary
2. [ ] Check gradients: print .grad for decoder parameters after backward
3. [ ] Check encoder frozen: verify encoder param values unchanged after step
4. [ ] Check loss components: print BCE and Dice separately
5. [ ] Check pos_weight: print computed pos_weight per batch
6. [ ] Check predictions: visualize predicted edge maps every N steps
7. [ ] Check targets: visualize ground truth to confirm data loading correct
8. [ ] Check data range: verify inputs are [0, 1], targets are binary {0, 1}
9. [ ] Check no NaN/Inf: torch.isnan(loss).any(), torch.isinf(loss).any()
10. [ ] Check memory: print torch.cuda.memory_allocated() at key points
```
