# Rollback

## Current Rollback Points

### Training Checkpoint Rollback

**Change**: Training from epoch N to epoch N+K

**Affected Files**: `checkpoints/checkpoint_last.pt`, `checkpoints/checkpoint_best.pt`

**Rollback Procedure**:
1. Stop training (`Ctrl+C` or kill torchrun process)
2. Resume from earlier checkpoint: `--resume_from checkpoints/checkpoint_best.pt`

**Verification Steps**:
1. Load checkpoint and check epoch number
2. Run `evaluate_features.py` on the checkpoint
3. Verify loss is reasonable

**Known Risks**: Optimizer state may be stale if resuming from much earlier checkpoint.

**Recovery Time Estimate**: Immediate (checkpoint loading is fast)

---

### Architecture Change Rollback

**Change**: Any modification to `student/aggregator.py`

**Affected Files**: `student/aggregator.py`, `student/layers/*.py`

**Rollback Procedure**:
1. `git checkout HEAD -- student/aggregator.py`
2. Verify with `python test_init.py`

**Verification Steps**:
1. Import StudentAggregator successfully
2. Forward pass produces expected output shape
3. Checkpoint loading still works

**Known Risks**: Checkpoint incompatibility if architecture changed.

**Recovery Time Estimate**: <1 minute

---

### Loss Function Rollback

**Change**: Modifications to `distillation/loss.py` or `distillation/projection.py`

**Affected Files**: `distillation/loss.py`, `distillation/projection.py`

**Rollback Procedure**:
1. `git checkout HEAD -- distillation/loss.py distillation/projection.py`
2. Verify projection checkpoint compatibility

**Known Risks**: Projection state_dict keys must match.

**Recovery Time Estimate**: <1 minute
