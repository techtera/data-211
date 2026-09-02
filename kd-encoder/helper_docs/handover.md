# Handover

## Date

2026-09-02

## Current Objective

Knowledge distillation of VGGT encoder: compress 909M teacher to 255M student via layer-wise feature distillation.

## Completed

- Student encoder architecture (StudentAggregator: 768-dim, 18 layers, frame+global attention)
- DINOv2 ViT-Large initialization (project 1024→768)
- Distillation pipeline (MSE+Cosine loss, token sampling, projection heads)
- DDP training infrastructure (torchrun, DistributedSampler, gradient accumulation)
- Training to epoch 36 with excellent results (score 1.49)
- Feature evaluation tool (evaluate_features.py)
- Training ongoing to epoch 80
- Removed unused one-time test scripts (test_init.py, test_training_setup.py, verify_init.py, sanity_check_ddp.py)

## In Progress

- Continued training to epoch 80 (currently at or past epoch 36)
- Awaiting final checkpoint for decoder training

## Pending

- Complete training to epoch 80
- Final feature evaluation on epoch 80 checkpoint
- Decoder training with student encoder features
- TensorRT conversion for Orin NX deployment
- INT8 quantization for inference

## Known Issues

- No validation set split (best checkpoint based on training loss only)
- Single-frame training (num_frames=1) — temporal features not learned
- Token sampling is random per step (not deterministic)

## Risks

- Student features may degrade after epoch 36 (overfitting)
- Decoder performance unknown until decoder training begins
- Orin NX memory budget (16GB) constrains model size + input resolution

## Important Decisions

- DINOv2-Large over DINOv2-Base for initialization (better feature quality)
- 768-dim student (not 1024) for deployment size constraints
- MSE:Cosine ratio of 0.7:0.3 (empirically chosen)
- Progressive layer weights [1.0, 1.5, 2.0, 2.5] (later layers more important)

## Recommended Next Step

1. Check training progress (is epoch 80 complete?)
2. Evaluate final checkpoint with `evaluate_features.py`
3. If quality is sufficient (correlation >0.75), proceed to decoder training
