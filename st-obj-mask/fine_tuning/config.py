"""Configuration for Student Encoder + Object Mask fine-tuning."""

import torch

STUDENT_CHECKPOINT = "../kd-encoder/checkpoints_full/student_final.pt"

IMAGE_SIZE = 518
NUM_CLASSES = 2
BATCH_SIZE = 4
NUM_WORKERS = 4

DATASET_ROOT = "data"
VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42

NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LOG_EVERY = 10

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 0
SAVE_LATEST = True
SAVE_BEST = True
SAVE_FINAL = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
