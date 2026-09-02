# Constraints

## Hard Constraints

1. Do not modify the StudentAggregator architecture or its checkpoint format.
2. Do not change the cached layer indices [3, 8, 13, 17] without updating both the feature extractor and the student encoder.
3. Do not change the input feature dimension (1536 = 768 frame + 768 global concatenated) without updating the encoder.
4. Do not modify ImageNet normalization constants — normalization is handled inside the student encoder.
5. The student encoder must remain frozen (requires_grad=False) during all decoder training.
6. Do not add new dependencies without approval.
7. Do not change the deep supervision output structure (logits, ds1_logits, ds2_logits) without updating the loss function, trainer, and validation code.
8. Positive weight clamping [5, 25] in EdgeLoss is critical for handling edge/background imbalance — do not remove or widen without testing.
9. Do not change the checkpoint format (model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, loss).

## Soft Constraints

1. Target deployment on Jetson Orin NX 16GB — decoder size decisions must consider this budget (shared with object mask decoder + student encoder).
2. Feature evaluation correlation from student encoder should be >0.75 before starting decoder training.
3. Edge detection F1 should be >0.5 before considering the model viable.
4. Batch size should stay at 4 per GPU to avoid OOM with the frozen encoder in memory.

## Data Constraints

1. Images must be 518x518 (VGGT standard input size).
2. Edge masks must be binary (binarized at threshold 0.5).
3. Mask files must follow the naming convention: `{stem}_mask.{ext}` in the `masks/` directory.
4. RGB images go in `rgb/`, masks go in `masks/`.
5. Do not modify the dataset directory structure without updating EdgeMaskDataset.
