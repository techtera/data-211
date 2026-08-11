# Project Status: VGGT + UNet++ Edge Masking

## Completed Modules

| Module | File | Status | Tests |
|--------|------|--------|-------|
| Loss function | `edge_mask/losses.py` | Done | 8/8 passed |
| Feature extractor | `edge_mask/feature_extractor.py` | Done | 7/7 passed |
| UNet++ decoder | `edge_mask/decoder.py` | Done | 7/7 passed |
| Edge refinement | `edge_mask/refinement.py` | Done | 5/5 passed |
| Full model | `edge_mask/model.py` | Done | 6/6 passed |

## Remaining Work

| Module | File | Status |
|--------|------|--------|
| Training loop | `edge_mask/train.py` | Not started |
| Evaluation metrics | `edge_mask/evaluate.py` | Not started |
| Dataset / dataloader | — | Not started |
| Config file | `edge_mask/config.yaml` | Not started |

## Parameter Counts

| Component | Parameters |
|-----------|-----------|
| VGGT Encoder (frozen) | 909,112,320 |
| Feature projections | 4,514,688 |
| UNet++ decoder | 5,278,018 |
| Edge refinement | 74,112 |
| Final conv | 65 |
| **Total trainable** | **9,866,883** |

## Verified Tensor Shapes

```
Input:              [B, S, 3, 518, 518]

Encoder output:     [B, S, 1374, 2048] at layers [4, 11, 17, 23]
After slice:        [B, S, 1369, 2048]
Spatial reshape:    [B*S, 2048, 37, 37]

Feature projections:
  Level 0:          [B*S, 64, 148, 148]
  Level 1:          [B*S, 128, 74, 74]
  Level 2:          [B*S, 256, 37, 37]
  Level 3:          [B*S, 512, 19, 19]

Decoder output:
  x_0_3:            [B*S, 64, 148, 148]
  DS1:              [B*S, 1, 518, 518]
  DS2:              [B*S, 1, 518, 518]

Final output:       [B, S, 1, 518, 518]
```

## Architecture Summary

```
VGGT Encoder (frozen ViT-L, 24 layers)
    → Extract layers [4, 11, 17, 23]
    → 1x1 Projection + Spatial Resize → 4-level pyramid
    → UNet++ Decoder (6 nodes, dense skip connections)
    → Deep Supervision (2 auxiliary heads, weights 0.1/0.2/1.0)
    → Edge Refinement (residual Conv3x3 block)
    → Conv1x1 → Bilinear upsample → Edge logits
```

## Loss Configuration

```
Loss = 0.5 * WeightedBCE + 0.5 * DiceLoss
pos_weight = clamp(neg/pos, 5, 25), computed per-batch
Total = 1.0 * final + 0.2 * DS2 + 0.1 * DS1
```

## Training Configuration (planned)

```
Optimizer:     AdamW (lr=3e-4, weight_decay=0.01)
Scheduler:     CosineAnnealing + 5% linear warmup
Batch size:    4-8
Grad clip:     max_norm=1.0
Precision:     Mixed (fp16 forward, fp32 loss)
```

## File Structure

```
vggt-edge-mask/
├── edge_mask/
│   ├── __init__.py
│   ├── losses.py
│   ├── feature_extractor.py
│   ├── decoder.py
│   ├── refinement.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   ├── test_losses.py
│   ├── test_feature_extractor.py
│   ├── test_decoder.py
│   ├── test_refinement.py
│   └── test_model.py
├── docs/
│   ├── architecture_v1.md
│   ├── implementation_spec.md
│   ├── development_plan.md
│   ├── risk_assessment.md
│   └── unetpp_study_guide.md
├── vggt/                    (upstream VGGT source)
├── inference_smoke_test.py
└── STATUS.md
```
