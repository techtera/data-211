# Handover

## Date

2026-09-04

## Current Objective

Knowledge distillation of VGGT encoder: compress 909M teacher to 255M student via layer-wise feature distillation. Breaking the loss plateau at ~0.128.

## Completed

- Student encoder architecture (StudentAggregator: 768-dim, 18 layers, frame+global attention)
- DINOv2 ViT-Large initialization (project 1024→768)
- Distillation pipeline (MSE+Cosine loss, token sampling, projection heads)
- DDP training infrastructure (torchrun, DistributedSampler, gradient accumulation)
- Previous run (checkpoints_full/): trained to epoch 36 with excellent results (score 1.49)
- Feature evaluation tool (evaluate_features.py)
- Decoder training pipelines prepared (st-obj-mask, st-edge-mask)
- Training monitor tool (monitor.py) for live loss tracking
- **Fixed cosine similarity: per-token (dim=-1) instead of per-sample flatten**
- **Added gradient clipping (max_norm=1.0) to stabilize training**
- **Removed per-step torch.cuda.empty_cache() (kept periodic every 100 steps)**

## In Progress

- checkpoints_v2/ run: reached epoch 41, loss plateaued at ~0.128-0.133 (best 0.111 at ep18)
- OOM crash at step 1195/1480 on epoch 41 due to GPU contention (PID 1747393 using 59GB)
- Need to resume training with fixed code from checkpoint_best.pt or checkpoint_last.pt

## Pending

- Kill competing GPU process (PID 1747393) on VM
- Resume training with fixed code + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- Monitor whether loss breaks below 0.111 with fixed cosine loss
- Final feature evaluation
- Decoder training with student encoder features
- TensorRT conversion for Orin NX deployment

## Known Issues

- No validation set split (best checkpoint based on training loss only)
- Single-frame training (num_frames=1) — temporal features not learned
- Token sampling is random per step (not deterministic)
- No mixed precision (AMP) — FP32 training, speedup possible for future runs

## Risks

- Changing cosine computation mid-training alters loss landscape — loss values not directly comparable to pre-fix epochs
- Should resume from checkpoint_best.pt (epoch 18) for cleanest comparison
- Decoder performance unknown until decoder training begins
- Orin NX memory budget (16GB) constrains model size + input resolution

## Important Decisions

- DINOv2-Large over DINOv2-Base for initialization (better feature quality)
- 768-dim student (not 1024) for deployment size constraints
- MSE:Cosine ratio of 0.7:0.3 (empirically chosen)
- Progressive layer weights [1.0, 1.5, 2.0, 2.5] (later layers more important)
- **2026-09-04: Fixed cosine sim from flatten-per-sample to per-token — root cause of plateau**
- **2026-09-04: Added grad clipping 1.0 to prevent oscillations from token sampling noise**

## Recommended Next Step

1. On VM: kill PID 1747393, verify GPU is free with `nvidia-smi`
2. Resume from checkpoint_best.pt (epoch 18) with fixed code:
   ```
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 train_ddp.py \
     --image_dir <images> --resume_from checkpoints_v2/checkpoint_best.pt \
     --epochs 80 --checkpoint_dir checkpoints_v3
   ```
3. Monitor loss — should break below 0.111 within a few epochs
4. Evaluate with `evaluate_features.py` once loss stabilizes
