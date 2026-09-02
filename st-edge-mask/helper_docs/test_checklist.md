# Test Checklist

## Pending Tests

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| Student checkpoint load | Verify student_final.pt loads into StudentAggregator | `python -c "import torch; from student import StudentAggregator; m = StudentAggregator(); m.load_state_dict(torch.load('../kd-encoder/checkpoints_v2/student_final.pt', map_location='cpu')['student_state_dict']); print('OK')"` | Prints "OK" | — | PENDING | Requires student_final.pt to exist |
| Model forward pass | Verify StudentEdgeMask produces correct output shapes | `python single_batch_overfit.py` (first iteration) | logits shape [B, 1, 518, 518], ds1 and ds2 same shape | — | PENDING | — |
| Single batch overfit | Verify decoder can memorize a single sample | `python single_batch_overfit.py` | Loss converges toward 0 over ~100 steps | — | PENDING | — |
| DDP training launch | Verify torchrun + DDP setup works | `torchrun --nproc_per_node=2 train_ddp.py --epochs 1` | Completes 1 epoch without errors | — | PENDING | Needs data/ directory populated |
| Validation metrics | Verify Precision/Recall/F1/IoU are computed correctly | Run 1 epoch, check logged metrics | Non-zero, plausible metric values | — | PENDING | — |
| Early stopping | Verify training stops after patience epochs with no improvement | Run with patience=2 on small dataset | Stops within 2+N epochs | — | PENDING | — |
| Checkpoint save/load | Verify checkpoints can be saved and loaded for resume | Save checkpoint, then load and verify state matches | model_state_dict, optimizer_state_dict, scheduler_state_dict all present | — | PENDING | — |
| Inference pipeline | Verify infer_standalone.py produces edge masks | `python infer_standalone.py --input test.jpg` | Output edge mask PNG with correct dimensions | — | PENDING | — |
