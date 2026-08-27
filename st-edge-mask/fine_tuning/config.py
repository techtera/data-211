"""Configuration for Student Encoder + UNet++ Edge Mask fine-tuning."""

import torch

STUDENT_CHECKPOINT = "../kd-encoder/checkpoints_full/student_final.pt"

IMAGE_SIZE = 518
DATASET_ROOT = "data"
VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42

BATCH_SIZE = 4
NUM_WORKERS = 8

NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_FRACTION = 0.05
LOG_EVERY = 10

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
POS_WEIGHT_CLAMP = (5, 25)
DS1_WEIGHT = 0.1
DS2_WEIGHT = 0.2
FINAL_WEIGHT = 1.0

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 0
SAVE_LATEST = True
SAVE_BEST = True
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
