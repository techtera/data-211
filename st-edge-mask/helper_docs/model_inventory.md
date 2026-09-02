# Model Inventory

## StudentEdgeMask

### Model Architecture

End-to-end edge mask prediction model combining a frozen distilled student encoder with a UNet++ decoder.

### Encoder

- **Type**: StudentAggregator (distilled from VGGT teacher)
- **Parameters**: 255M (all frozen)
- **Embed dim**: 768 per branch, 1536 concatenated (frame + global)
- **Depth**: 18 layers
- **Cached layers**: [3, 8, 13, 17]
- **Token structure**: [1 camera, 4 registers, 1369 patches] = 1374 tokens
- **patch_start_idx**: 5
- **Initialization**: DINOv2 ViT-Large with truncation projection

### Feature Extractor

- **Type**: StudentFeatureExtractor
- **4 FeatureProjection modules**:
  - Layer 3 → Conv2d(1536, 64, 1) + GroupNorm(8) + SiLU + interpolate to 148x148 + Conv2d(64, 64, 3)
  - Layer 8 → Conv2d(1536, 128, 1) + GroupNorm(8) + SiLU + interpolate to 74x74 + Conv2d(128, 128, 3)
  - Layer 13 → Conv2d(1536, 256, 1) + GroupNorm(8) + SiLU (native 37x37)
  - Layer 17 → Conv2d(1536, 512, 1) + GroupNorm(8) + SiLU + Conv2d(512, 512, 3, stride=2) (downsample)

### Decoder

- **Type**: UNetPPDecoder
- **Channels**: (64, 128, 256, 512)
- **Upsample blocks**: 6 (up_3_0, up_2_0, up_2_1, up_1_0, up_1_1, up_1_2)
- **ConvBlocks**: 6 (conv_2_1, conv_1_1, conv_1_2, conv_0_1, conv_0_2, conv_0_3)
- **Deep Supervision Heads**: 2 (ds1 at x_0_1, ds2 at x_0_2)
  - Each: Conv2d(64, 32, 3) → SiLU → Conv2d(32, 1, 1) → interpolate to 518x518

### Refinement

- **Type**: EdgeRefinement
- **Channels**: 64

### Final Output

- **Layer**: Conv2d(64, 1, 1) → interpolate to 518x518
- **Training**: returns (logits, ds1_logits, ds2_logits)
- **Eval**: returns sigmoid(logits)

### Loss Functions

- **EdgeLoss**: BCE(weight=0.5) + Dice(weight=0.5)
- **Deep supervision weights**: ds1=0.1, ds2=0.2, final=1.0
- **Positive weight**: Dynamic, clamped to [5, 25] based on edge/background ratio

### Datasets

- **Type**: EdgeMaskDataset
- **Structure**: `data/rgb/` (images) + `data/masks/` (binary edge masks with `_mask` suffix)
- **Size**: 518x518
- **Binarization**: threshold 0.5
- **Validation split**: 10% (random_split with seed=42)

### Augmentations

- None (current pipeline)

### Metrics

- **Per-batch**: Precision, Recall, F1, IoU (at threshold 0.5)
- **Final evaluation**: BF1 (Boundary F1), ODS (Optimal Dataset Scale F1), Dice, Confusion Matrix

### Training Configuration

- **Framework**: DDP via torchrun
- **Batch size**: 4 per GPU
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: Linear warmup (5% of steps) + cosine decay to 0
- **Gradient clipping**: max_norm=1.0
- **Epochs**: 100 (max)
- **Early stopping**: patience=15 on F1
- **GPUs**: 2× (typical setup)

### Inference Configuration

- **Script**: `infer_standalone.py`
- **Input**: Single image or directory
- **Output**: Edge probability map (sigmoid) or binary mask (thresholded)
- **Mode**: model.eval() — returns sigmoid(logits) only

### Checkpoint Locations

- **Latest**: `checkpoints/checkpoint_last.pt`
- **Best F1**: `checkpoints/checkpoint_best.pt`
- **Best Loss**: `checkpoints/checkpoint_best_loss.pt`
- **Student encoder**: `../kd-encoder/checkpoints_v2/student_final.pt`

### Known Limitations

- No data augmentation — may limit generalization
- Single-frame training (S=1) — no temporal edge consistency
- Validation uses model.train() mode (BatchNorm statistics differ from eval)
- Feature quality depends entirely on student encoder KD training quality
- No multi-threshold evaluation during training (only threshold=0.5)
