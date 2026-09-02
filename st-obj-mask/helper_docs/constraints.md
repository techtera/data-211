# Constraints

## Hard Constraints

1. Do not modify the StudentAggregator or its checkpoint format.
2. Do not change the student's cached layer indices [3, 8, 13, 17].
3. Do not change the student's output dimension (1536 = 768 frame + 768 global concatenated).
4. Do not modify ImageNet normalization constants (handled inside the encoder).
5. Do not change the number of output classes (2: background + object) without updating loss and metrics.
6. The student encoder must remain frozen (requires_grad=False) during all decoder training.
7. Do not add new dependencies without approval.
8. Do not change the checkpoint format (model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, loss).

## Soft Constraints

1. Target deployment on Jetson Orin NX 16GB — decoder size must be small relative to encoder.
2. Decoder training should complete in reasonable time on 2x GPU setup.
3. Validation mIoU should exceed 0.7 before considering the decoder production-ready.
4. Batch size of 2 per GPU is the tested configuration — increasing may cause OOM.

## Data Constraints

1. Images must be 518x518 (VGGT standard input size).
2. Dataset uses YOLO polygon annotation format (class_id x1 y1 x2 y2 ... xn yn).
3. Masks are binary: 0 = background, 1 = object.
4. Dataset normalization happens inside the encoder — do not normalize in the dataloader.
