# Rollback

## 2026-09-02 — Added LR Scheduler + Early Stopping to Object Mask Training

**Change**: Added warmup+cosine LR scheduler, early stopping (patience=15), and config constants (GRAD_CLIP_MAX_NORM, WARMUP_FRACTION, PATIENCE) to object mask decoder training.

**Affected Files**:
- `fine_tuning/config.py` — added GRAD_CLIP_MAX_NORM, WARMUP_FRACTION, PATIENCE
- `fine_tuning/scheduler.py` — new file
- `train_ddp.py` — integrated scheduler and early stopping

**Rollback Procedure**:
1. Revert `fine_tuning/config.py` to remove GRAD_CLIP_MAX_NORM, WARMUP_FRACTION, PATIENCE
2. Delete `fine_tuning/scheduler.py`
3. Revert `train_ddp.py` to remove scheduler import, scheduler.step(), patience logic
4. Restore hardcoded grad_clip=1.0 and remove scheduler param from train_epoch_ddp

**Verification Steps**:
- Run `python -c "import ast; ast.parse(open('train_ddp.py').read())"` to verify syntax
- Run single-batch overfit to verify training still works

**Known Risks**: None — these are additive changes to training infrastructure, not model architecture.

**Recovery Time Estimate**: < 5 minutes via git revert.

---

## 2026-09-02 — Fixed Checkpoint Path

**Change**: Changed STUDENT_CHECKPOINT from `../kd-encoder/checkpoints_full/student_final.pt` to `../kd-encoder/checkpoints/student_final.pt` in config.py and inference scripts.

**Affected Files**:
- `fine_tuning/config.py`
- `infer_standalone.py`
- `README.md`

**Rollback Procedure**: Revert the path strings (only needed if encoder saves to a different directory).

**Known Risks**: None — the old path was incorrect (directory didn't exist).
