"""
Configuration for VGGT + UNet++ Edge Mask fine-tuning.
"""

import torch


# ============================================================
# Pretrained Model
# ============================================================

PRETRAINED_MODEL = "facebook/VGGT-1B"


# ============================================================
# Image
# ============================================================

IMAGE_SIZE = 518


# ============================================================
# Dataset
# ============================================================

DATASET_ROOT = "data"

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# ============================================================
# DataLoader
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 8


# ============================================================
# Training
# ============================================================

NUM_EPOCHS = 100

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

GRAD_CLIP_MAX_NORM = 1.0

# Warmup is 5% of total steps
WARMUP_FRACTION = 0.05

# Print training loss every N batches
LOG_EVERY = 10


# ============================================================
# Loss
# ============================================================

BCE_WEIGHT = 0.5

DICE_WEIGHT = 0.5

POS_WEIGHT_CLAMP = (5, 25)

# Deep supervision weights
DS1_WEIGHT = 0.1

DS2_WEIGHT = 0.2

FINAL_WEIGHT = 1.0


# ============================================================
# Checkpoints
# ============================================================

CHECKPOINT_DIR = "checkpoints"

SAVE_EVERY = 0

SAVE_LATEST = True

SAVE_BEST = True


# ============================================================
# Early Stopping
# ============================================================

PATIENCE = 15


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
