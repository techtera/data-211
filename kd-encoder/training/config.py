"""
Training configuration for Phase 1 distillation.

All hyperparameters in one place for easy tuning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Phase 1 distillation training configuration."""

    # Training duration
    num_epochs: int = 50

    # Batch settings
    batch_size: int = 4  # Per GPU
    num_workers: int = 4  # DataLoader workers

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss weights
    mse_weight: float = 0.7
    cosine_weight: float = 0.3
    layer_weights: list = None  # [1.0, 1.5, 2.0, 2.5] default

    # Token sampling
    num_patch_samples: int = 128  # 5 special + 128 patches = 133 total

    # Checkpointing
    save_every: int = 5  # Save every N epochs
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every: int = 10  # Log every N steps

    # Data
    image_dir: str = None  # Required: path to images
    image_size: int = 518  # VGGT input size
    num_frames: int = 8  # Sequence length

    # Device
    device: str = "cuda"  # cuda or cpu
    use_multi_gpu: bool = True  # Use all available GPUs
    gpu_ids: Optional[list] = None  # Specific GPU IDs (None = use all)

    # Resume training
    resume_from: Optional[str] = None  # Checkpoint path

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.layer_weights is None:
            self.layer_weights = [1.0, 1.5, 2.0, 2.5]

    @classmethod
    def sanity_check_config(cls):
        """Quick config for sanity check (3-5 epochs)."""
        return cls(
            num_epochs=5,
            batch_size=2,
            warmup_epochs=1,
            save_every=1,
            log_every=5,
        )

    @classmethod
    def full_training_config(cls):
        """Full 40-50 epoch training config."""
        return cls(
            num_epochs=50,
            batch_size=4,
            warmup_epochs=5,
            save_every=5,
            log_every=10,
        )
