"""
Configuration file for SegFormer fine-tuning.
"""

import torch

# ============================================================
# Pretrained Model
# ============================================================

PRETRAINED_MODEL = "facebook/VGGT-1B"

# ============================================================
# Dataset
# ============================================================

IMAGE_SIZE = 518

NUM_CLASSES = 2

BATCH_SIZE = 2

NUM_WORKERS = 4

# ============================================================
# Training
# ============================================================

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

# Print training loss every N batches
LOG_EVERY = 10

# ============================================================
# TensorBoard
# ============================================================

LOG_DIR = "runs/segformer_finetuning"

# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = "checkpoints"

SAVE_EVERY = 5

SAVE_LATEST = True

SAVE_BEST = True

SAVE_FINAL = True

# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)