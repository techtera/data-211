# Model Inventory

## Teacher: VGGT Encoder

- **Architecture**: ViT-based aggregator with alternating frame/global attention
- **Parameters**: 909M
- **Layers**: 24 transformer blocks (frame + global)
- **Dimension**: 1024 per branch, 2048 concatenated (frame+global)
- **Heads**: 16 attention heads
- **Cached Layers**: [4, 11, 17, 23]
- **Input**: [B, S, 3, 518, 518] images in [0,1]
- **Output**: List of 24 tensors (4 non-None cached) [B, S, 1374, 2048]
- **Checkpoint**: `../../vggt-unified/checkpoints/vggt_unified_fp16.pt`

## Student: StudentAggregator

- **Architecture**: ViT-based aggregator with alternating frame/global attention
- **Parameters**: 255M
- **Layers**: 18 transformer blocks (frame + global)
- **Dimension**: 768 per branch, 1536 concatenated (frame+global)
- **Heads**: 12 attention heads
- **MLP Ratio**: 4.0
- **Cached Layers**: [3, 8, 13, 17]
- **Input**: [B, S, 3, 518, 518] images in [0,1]
- **Output**: List of 18 tensors (4-5 non-None cached) [B, S, 1374, 1536]
- **Initialization**: DINOv2 ViT-Large projected 1024→768
- **Special Tokens**: 1 camera + 4 register = 5 (patch_start_idx=5)
- **Position Encoding**: RoPE 2D (frequency=100)
- **Output Norm**: LayerNorm(1536) on concatenated features

## Projection Heads (Training-Only)

- **Architecture**: 4× (LayerNorm(1536) + Linear(1536→2048))
- **Parameters**: ~12.6M total (~3.15M per head)
- **Purpose**: Align student features to teacher dimension during distillation
- **Discarded**: After training completes

## DINOv2 ViT-Large (Initialization Source)

- **Architecture**: ViT-Large with register tokens
- **Parameters**: ~300M
- **Layers**: 24
- **Dimension**: 1024
- **Source**: `torch.hub` → `facebookresearch/dinov2` → `dinov2_vitl14_reg`
- **Usage**: Weight initialization only (projected 1024→768 by truncation)

## Loss Functions

- **Primary**: `DistillationLoss` — MSE (0.7) + Cosine (0.3) with projection
- **Layer Weights**: [1.0, 1.5, 2.0, 2.5] (later layers weighted more)
- **Simplified**: `SimplifiedDistillationLoss` — same loss without projection (testing)

## Datasets

- **Format**: Directory of JPG/PNG images
- **Preprocessing**: Resize to 518×518, ToTensor ([0,1])
- **Normalization**: Internal (ImageNet mean/std applied in model forward)
- **Training set**: 23,687 images
- **Sequence**: `num_frames=1` (single frame, replicated for S dimension)

## Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
- **Schedule**: Cosine annealing with linear warmup
- **Warmup**: 5 epochs (or 3 for sanity checks)
- **Total Epochs**: 80
- **Batch Size**: 64 per GPU (effective = batch × GPUs × accumulation)
- **Token Sampling**: 128 patches + 5 special = 133 tokens (from 1374)
- **Hardware**: 2× A100 GPUs, ~30 hours for 80 epochs

## Evaluation Metrics

- **Cross-Correlation**: Student-teacher feature correlation per layer
- **Feature Variance**: Discriminative power (student/teacher ratio)
- **Activation Sparsity**: Information density (near-zero fraction)
- **Feature Statistics**: Mean, std, range comparison
- **Quality Thresholds**: Correlation >0.75 = EXCELLENT, >0.65 = GOOD

## Checkpoint Locations

- `checkpoints/checkpoint_last.pt` — Latest epoch (resume training)
- `checkpoints/checkpoint_best.pt` — Lowest validation loss
- `checkpoints/student_final.pt` — Final model (state_dict only, no optimizer)
- `checkpoints_full/checkpoint_epoch_36.pt` — Known good checkpoint (score 1.49)

## Known Limitations

- Single-frame training may miss temporal feature learning
- DINOv2 initialization uses truncation, not learned projection
- Token sampling is random (not spatially strategic)
- No validation set split — best checkpoint is based on training loss
