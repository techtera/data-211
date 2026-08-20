# VGGT + UNet++ Edge Masking
  
## Project Overview. 

Edge mask prediction model using a frozen VGGT encoder (ViT-L) and a custom UNet++ decoder. Predicts binary edge masks from 518x518 images. Not semantic segmentation — the goal is sharp, continuous boundary reconstruction under severe class imbalance (1-10% edge pixels).

## Current State

Architecture and training pipeline fully implemented and tested. Evaluation metrics done. Next step: run overfit test with real data + VGGT checkpoint, then full training.

### Completed

- `edge_mask/losses.py` — WeightedBCE + Dice (8 tests pass)
- `edge_mask/feature_extractor.py` — VGGT feature extraction + 4-level projection (7 tests pass)
- `edge_mask/decoder.py` — UNet++ with dense skip connections + deep supervision (7 tests pass)
- `edge_mask/refinement.py` — Residual edge refinement block (5 tests pass)
- `edge_mask/model.py` — Full pipeline assembly (6 tests pass)
- `fine_tuning/config.py` — All training constants (9 tests pass)
- `fine_tuning/dataset.py` — EdgeMaskDataset: rgb/abc.png → masks/abc_mask.png (6 tests pass)
- `fine_tuning/dataloader.py` — 90/10 train/val split (2 tests pass)
- `fine_tuning/losses.py` — EdgeLoss + deep supervision weights (5 tests pass)
- `fine_tuning/model_builder.py` — VGGT.from_pretrained + VGGTEdgeMask assembly
- `fine_tuning/optimizer.py` — AdamW, trainable params only (4 tests pass)
- `fine_tuning/scheduler.py` — Cosine + 5% linear warmup (3 tests pass)
- `fine_tuning/checkpoints.py` — Save best + latest checkpoint (2 tests pass)
- `fine_tuning/validate.py` — Validation loop with collapse detection (2 tests pass)
- `fine_tuning/trainer.py` — Full training loop with early stopping (5 tests pass)
- `fine_tuning/evaluate.py` — Dice, BF1, ODS, Confusion Matrix (12 tests pass)
- `fine_tune.py` — Top-level training entry point
- `overfit_single_batch.py` — Single batch overfit sanity check

### Not Started

- Real training run (need GPU + data)
- Inference script

## Architecture (do not modify)

```
Input [B, S, 3, 518, 518]
  → VGGT Encoder (frozen, layers [4,11,17,23] → [B,S,1374,2048])
  → Slice patch tokens [5:] → reshape to [B*S, 2048, 37, 37]
  → Feature Projections:
      Level 0: 2048→64, bilinear(148x148), Conv3x3  
      Level 1: 2048→128, bilinear(74x74), Conv3x3
      Level 2: 2048→256, identity
      Level 3: 2048→512, Conv3x3 stride=2
  → UNet++ Decoder (6 nodes, channels [64,128,256,512])
  → Deep Supervision (DS1 weight=0.1, DS2 weight=0.2, final weight=1.0)
  → Edge Refinement (residual Conv3x3 block)
  → Conv1x1(64→1) → bilinear(518x518)
  → Output [B, S, 1, 518, 518]
```

## Key Constants

- Input: 518x518, patch_size=14, grid=37x37
- Encoder dim: 2048 (frame+global concat), patch_start_idx=5
- Decoder: GroupNorm(8), SiLU, bilinear upsample with explicit target size
- Loss: 0.5*BCE + 0.5*Dice, pos_weight clamp (5, 25)
- Trainable params: 9,866,883

## Training Config

- AdamW lr=3e-4, weight_decay=0.01
- CosineAnnealing + 5% linear warmup
- Gradient clipping max_norm=1.0
- Mixed precision (fp16 forward, fp32 loss)
- Batch size 4
- Early stopping: patience=15
- Checkpoints: best + latest only (~3.6 GB each)

## Dataset

- Structure: `data/rgb/abc.png` → `data/masks/abc_mask.png`
- S=1, pre-augmented, resized to 518x518
- 90/10 train/val split (seed=42)

## Evaluation Metrics

- **Dice**: 2*intersection / (pred + gt)
- **BF1**: Boundary F1 with tolerance=2px
- **ODS**: Best F1 across thresholds (0.05 to 0.95)
- **Confusion Matrix**: TP, FP, FN, TN

## Important Notes

- Encoder is ALWAYS frozen (no_grad + detach + eval mode)
- Upsample uses `F.interpolate(size=target)` NOT `scale_factor=2` (19*2=38≠37)
- All tests in `tests/` — run with `python tests/test_<module>.py`
- Full docs in `docs/` — architecture_v1.md, implementation_spec.md, development_plan.md, risk_assessment.md, unetpp_study_guide.md
- `inference_smoke_test.py` verifies end-to-end forward pass
- Device transfers use `non_blocking=True` to avoid device mismatch

## Commands

```bash
# Run architecture tests
python tests/test_losses.py
python tests/test_feature_extractor.py
python tests/test_decoder.py
python tests/test_refinement.py
python tests/test_model.py

# Run fine-tuning tests
python tests/test_ft_config.py
python tests/test_ft_dataset.py
python tests/test_ft_dataloader.py
python tests/test_ft_losses.py
python tests/test_ft_optimizer.py
python tests/test_ft_scheduler.py
python tests/test_ft_checkpoints.py
python tests/test_ft_validate.py
python tests/test_ft_trainer.py
python tests/test_ft_evaluate.py

# Smoke test (full forward pass)
python inference_smoke_test.py

# Training
python fine_tune.py

# Single batch overfit test
python overfit_single_batch.py
```
