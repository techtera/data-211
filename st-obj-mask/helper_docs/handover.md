# Handover

## Date

2026-09-02

## Current Objective

Train the object mask decoder using the distilled student encoder. The decoder (SegFormer-based) takes frozen student features and produces 2-class segmentation masks (background + object).

## Completed

- StudentObjMask model architecture (frozen StudentAggregator + ObjMaskDecoder/DPTHead with SegFormer)
- DDP training script (train_ddp.py) with DistributedDataParallel
- SegmentationDataset pipeline (YOLO polygon annotations → binary masks, 518x518)
- Validation loop with mIoU, Dice, Pixel Accuracy metrics
- Loss function: CrossEntropy + Dice (equal weight)
- Inference scripts (inference.py, infer_standalone.py)
- Single-batch overfit test script (single_batch_overfit.py)
- Fixed checkpoint path from `checkpoints_full/` to `checkpoints/`
- Added warmup + cosine LR scheduler (was constant LR)
- Added early stopping with patience=15 (was training all 100 epochs blindly)
- Added GRAD_CLIP_MAX_NORM and WARMUP_FRACTION to config

## In Progress

- Waiting for student encoder checkpoint to finish training (epoch 17 of 80 in kd-encoder)
- Encoder loss currently at ~0.1137, cosine LR schedule will continue decaying

## Pending

- Run decoder training once student_final.pt is available
- Evaluate mIoU/Dice on validation set
- Test inference pipeline end-to-end
- Benchmark latency for Orin NX deployment planning
- TensorRT conversion of full pipeline (encoder + decoder)

## Known Issues

- Checkpoint path was pointing to non-existent `checkpoints_full/` directory (fixed)
- obj-mask had no LR scheduler while edge-mask did (fixed: warmup+cosine added)
- obj-mask had no early stopping (fixed: patience=15 on val mIoU)
- No data augmentation in dataset pipeline
- Single-frame training only (num_frames=1)

## Risks

- Student encoder feature quality is unknown until evaluated with evaluate_features.py
- Decoder performance depends entirely on encoder quality — poor encoder = poor masks
- Orin NX 16GB memory budget constrains the full pipeline (encoder + decoder)
- No validation during encoder training (best checkpoint based on training loss only)

## Important Decisions

- SegFormer decoder chosen over DPT refinement network (lighter, proven architecture)
- 4-level feature pyramid from student cached layers [3, 8, 13, 17]
- CrossEntropy + Dice loss (equal weight) for balanced segmentation
- Warmup + cosine LR schedule to match edge-mask training strategy
- Early stopping on val mIoU with patience=15

## Recommended Next Step

1. Wait for encoder training to complete (epoch 80) or evaluate early checkpoint
2. Run `evaluate_features.py` on the student checkpoint (correlation > 0.75 threshold)
3. If quality is sufficient, start decoder training: `torchrun --nproc_per_node=2 train_ddp.py`
4. Monitor val mIoU — target > 0.7 for acceptable segmentation quality
