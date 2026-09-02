# Test Checklist

## Decoder Training Tests

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| Single-batch overfit | Verify model can learn | `python single_batch_overfit.py` | Loss → ~0, mIoU → ~1.0 | — | PENDING | Run after encoder checkpoint available |
| DDP training launch | Verify DDP setup | `torchrun --nproc_per_node=2 train_ddp.py --epochs 1` | Completes 1 epoch without error | — | PENDING | |
| Validation metrics | Verify metrics computation | Run 1 epoch with val set | mIoU, Dice, Pixel Acc reported | — | PENDING | |
| Early stopping | Verify patience logic | Train with patience=3 (test) | Stops after 3 no-improvement epochs | — | PENDING | |
| Checkpoint save/load | Verify checkpoint round-trip | Save then load checkpoint | Model produces same output | — | PENDING | |
| Inference standalone | Verify end-to-end inference | `python infer_standalone.py --image test.jpg` | Outputs segmentation mask | — | PENDING | |

## Encoder Checkpoint Compatibility

| Test | Purpose | Command | Expected Result | Actual Result | Status | Notes |
|------|---------|---------|-----------------|---------------|--------|-------|
| Load student checkpoint | Verify checkpoint format | Load student_final.pt into StudentAggregator | No errors, 255M params | — | PENDING | |
| Feature dimensions | Verify encoder output shape | Forward pass with dummy input | [B, S, 1374, 1536] at cached layers | — | PENDING | |
| Frozen encoder | Verify no gradients | Check requires_grad after forward | All encoder params have grad=None | — | PENDING | |
