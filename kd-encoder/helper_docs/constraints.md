# Constraints

## Hard Constraints

1. Do not modify the teacher encoder or its checkpoint format.
2. Do not change the student's output token structure: [1 camera, 4 registers, 1369 patches] = 1374 tokens.
3. Do not change the student's output dimension (1536 = 768 frame + 768 global concatenated).
4. Do not change cached layer indices without updating both student ([3, 8, 13, 17]) and teacher ([4, 11, 17, 23]) mappings.
5. Do not modify ImageNet normalization constants or move normalization outside the model.
6. Do not add new dependencies without approval.
7. Do not change the checkpoint format (student_state_dict, optimizer_state_dict, etc.).

## Soft Constraints

1. Target deployment on Jetson Orin NX 16GB — model size decisions must consider this budget.
2. Training should complete in <36 hours on 2× A100.
3. Feature evaluation correlation should be >0.75 before proceeding to decoder training.
4. Token sampling should keep all 5 special tokens (camera + registers).

## Data Constraints

1. Images must be 518×518 (VGGT standard input size).
2. Dataset normalization happens inside the model — do not normalize in the dataloader.
3. Do not modify the training dataset structure or image format requirements.
