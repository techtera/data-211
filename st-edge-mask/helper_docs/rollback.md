# Rollback

## 2026-09-02 — Checkpoint Path Fix

**Change**: Updated all references from `checkpoints_full/student_final.pt` to `checkpoints/student_final.pt`.

**Affected Files**:
- `fine_tuning/config.py`
- `infer_standalone.py`
- `README.md`

**Rollback Procedure**:
```bash
# Revert to old path if needed
sed -i 's|../kd-encoder/checkpoints/student_final.pt|../kd-encoder/checkpoints_full/student_final.pt|g' \
  fine_tuning/config.py infer_standalone.py README.md
```

**Verification Steps**:
1. `grep -r "checkpoints/" fine_tuning/config.py` — should show the expected path
2. Attempt to load the checkpoint: `python -c "import torch; torch.load('<path>', map_location='cpu')"`

**Known Risks**: None — this was a bug fix, not a behavior change.

**Recovery Time Estimate**: < 1 minute.
