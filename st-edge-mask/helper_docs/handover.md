# Handover

## Date

2026-09-02

## Current Objective

Train the edge mask decoder using the distilled student encoder (255M params). The decoder uses a UNet++ architecture with deep supervision to predict binary edge masks from student encoder features.

## Completed

- StudentEdgeMask model architecture (frozen StudentAggregator + StudentFeatureExtractor + UNet++ decoder + EdgeRefinement + final_conv)
- StudentFeatureExtractor: 4 FeatureProjection modules mapping 1536-dim student features to multi-scale spatial maps (64@148x148, 128@74x74, 256@37x37, 512@downsampled)
- UNetPPDecoder with nested skip connections and deep supervision at nodes x_0_1, x_0_2
- EdgeRefinement module (ch=64)
- EdgeLoss: BCE(0.5) + Dice(0.5) with deep supervision weights (ds1=0.1, ds2=0.2, final=1.0), dynamic positive weight clamped [5,25]
- DDP training infrastructure (train_ddp.py with torchrun)
- EdgeMaskDataset pipeline (rgb/ + masks/ with _mask suffix, binarized at 0.5, 518x518)
- Warmup (5%) + cosine decay LR scheduler
- Early stopping with patience=15 on F1 score
- Validation with Precision, Recall, F1, IoU metrics
- Final evaluation with BF1 and ODS metrics
- Standalone inference script (infer_standalone.py)
- Single-batch overfit test (single_batch_overfit.py)
- Fixed checkpoint path from checkpoints_full → checkpoints

## In Progress

- Waiting for student encoder checkpoint to finish training (currently epoch 17 of 80 in kd-encoder)
- Loss at ~0.1137, LR at 9.9e-5 (cosine schedule, still in upper portion)

## Pending

- Run decoder training once student_final.pt is available
- Evaluate F1/Precision/Recall/IoU/BF1/ODS on validation set
- Test inference pipeline end-to-end
- Verify Orin NX memory budget with full student encoder + edge decoder
- TensorRT conversion planning

## Known Issues

- Checkpoint path was wrong (referenced checkpoints_full/ which doesn't exist) — fixed to ../kd-encoder/checkpoints_v2/student_final.pt
- No data augmentation in current pipeline (may limit generalization)
- Validation keeps model in training mode to get 3 outputs for deep supervision loss computation (torch.no_grad() prevents gradient computation but BatchNorm/Dropout behavior differs)
- Single-frame training only (S=1)

## Risks

- Student encoder feature quality is unknown until evaluated after KD training completes
- Orin NX 16GB memory budget may not accommodate student encoder + edge decoder at full resolution
- Edge detection is highly class-imbalanced (few edge pixels vs background) — positive weight clamping is critical
- No validation split reproducibility across runs if dataset changes

## Important Decisions

- UNet++ over simpler decoders (better multi-scale feature fusion for fine edge details)
- Deep supervision at x_0_1, x_0_2 for auxiliary gradient paths
- BCE+Dice combination for handling edge pixel imbalance
- Feature projection targets: 148x148, 74x74, 37x37, downsampled (matching a 4-level FPN hierarchy)
- GroupNorm(8) + SiLU activation throughout projections
- Early stopping on F1 (not loss) — F1 is the metric that matters for edge quality

## Recommended Next Step

1. Wait for kd-encoder training to complete (epoch 80) and produce student_final.pt
2. Run evaluate_features.py on the final checkpoint to verify feature quality (correlation > 0.75)
3. Prepare data/ directory with rgb/ and masks/ subdirectories
4. Start decoder training: `torchrun --nproc_per_node=2 train_ddp.py --epochs 100`
