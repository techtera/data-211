#!/usr/bin/env python3
"""
End-to-end dry run test for training pipeline.

Tests complete flow:
    1. Create mock dataset
    2. Initialize teacher and student
    3. Setup optimizer, scheduler, loss
    4. Run training for 2 epochs
    5. Verify checkpointing and resuming
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tempfile
import os
import shutil

from student import StudentAggregator, initialize_student_from_dinov2
from training import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    DistillationTrainer
)
from distillation import DistillationLoss


class MockTeacher(nn.Module):
    """Mock teacher that returns correct output format."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, images):
        # images: [B, S, C, H, W]
        B, S, C, H, W = images.shape

        # Mock forward pass
        _ = self.conv(images.reshape(B * S, C, H, W))

        # Return cached features: 4 layers × [B, S, 1374, 2048]
        features = [
            torch.randn(B, S, 1374, 2048, device=images.device)
            for _ in range(4)
        ]

        patch_start_idx = 5
        return features, patch_start_idx


class MockDataset(Dataset):
    """Mock dataset that generates random images."""

    def __init__(self, num_samples=20, num_frames=8, image_size=518):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return [S, C, H, W]
        return torch.randn(self.num_frames, 3, self.image_size, self.image_size)


def test_training_pipeline():
    """Test complete training pipeline."""
    print("="*60)
    print("End-to-End Training Pipeline Test")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {temp_dir}")

    try:
        # Step 1: Create mock dataset
        print("\n[1] Creating mock dataset...")
        dataset = MockDataset(num_samples=20, num_frames=4, image_size=518)  # VGGT size
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Batches per epoch: {len(dataloader)}")

        # Step 2: Initialize models
        print("\n[2] Initializing models...")
        teacher = MockTeacher().to(device).eval()
        student = StudentAggregator().to(device)

        # Use DINOv2 initialization (optional, can skip for speed)
        print("  Initializing student with DINOv2...")
        try:
            initialize_student_from_dinov2(student, verbose=False)
            print("  ✓ DINOv2 initialized")
        except Exception as e:
            print(f"  ⚠ DINOv2 init failed (OK for dry run): {e}")

        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        student_params = sum(p.numel() for p in student.parameters()) / 1e6
        print(f"  Teacher params: {teacher_params:.1f}M")
        print(f"  Student params: {student_params:.1f}M")

        # Step 3: Create training components
        print("\n[3] Setting up training components...")

        # Config
        config = TrainingConfig(
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            warmup_epochs=1,
            save_every=1,
            log_every=2,
            checkpoint_dir=temp_dir,
            use_multi_gpu=False  # Force single GPU for dry run
        )

        # Loss function
        loss_fn = DistillationLoss(
            student_dim=1536,
            teacher_dim=2048,
            num_layers=4,
            layer_weights=config.layer_weights,
            mse_weight=config.mse_weight,
            cosine_weight=config.cosine_weight
        ).to(device)

        # Optimizer (student + projection heads)
        params_to_optimize = list(student.parameters()) + list(loss_fn.parameters())
        optimizer = create_optimizer(
            params_to_optimize,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        scheduler = create_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
            steps_per_epoch=len(dataloader),
            min_lr=config.min_lr
        )

        print(f"  ✓ Loss function created")
        print(f"  ✓ Optimizer created")
        print(f"  ✓ Scheduler created")

        # Step 4: Create trainer and run training
        print("\n[4] Running training...")
        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device
        )

        # Train for 2 epochs
        trainer.train(dataloader, start_epoch=0)

        # Step 5: Verify checkpoints
        print("\n[5] Verifying checkpoints...")
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        print(f"  Checkpoint files: {checkpoint_files}")

        assert len(checkpoint_files) >= 2, f"Expected at least 2 checkpoints, got {len(checkpoint_files)}"
        print(f"  ✓ Checkpoints created: {len(checkpoint_files)} files")

        # Check student_final.pt exists
        final_path = os.path.join(temp_dir, "student_final.pt")
        assert os.path.exists(final_path), "student_final.pt not found"
        print(f"  ✓ Final student model saved")

        # Step 6: Test checkpoint loading
        print("\n[6] Testing checkpoint loading...")
        from training.checkpoints import load_checkpoint

        latest_checkpoint = os.path.join(temp_dir, "checkpoint_epoch_2.pt")
        checkpoint_data = load_checkpoint(
            checkpoint_path=latest_checkpoint,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            projection=loss_fn.projection,
            device=device
        )

        assert checkpoint_data['epoch'] == 2, f"Expected epoch 2, got {checkpoint_data['epoch']}"
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"  ✓ Epoch: {checkpoint_data['epoch']}")
        print(f"  ✓ Loss: {checkpoint_data['loss']:.6f}")

        # Step 7: Test resuming training
        print("\n[7] Testing training resume...")

        # Create fresh optimizer and scheduler for resume
        optimizer_resume = create_optimizer(
            list(student.parameters()) + list(loss_fn.parameters()),
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler_resume = create_scheduler(
            optimizer_resume,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs + 1,  # One more epoch
            steps_per_epoch=len(dataloader),
            min_lr=config.min_lr
        )

        # Load checkpoint
        load_checkpoint(
            checkpoint_path=latest_checkpoint,
            student=student,
            optimizer=optimizer_resume,
            scheduler=scheduler_resume,
            projection=loss_fn.projection,
            device=device
        )

        # Create new trainer
        config_resume = TrainingConfig(
            num_epochs=3,  # One more epoch
            batch_size=2,
            learning_rate=1e-4,
            warmup_epochs=1,
            save_every=1,
            log_every=2,
            checkpoint_dir=temp_dir,
            use_multi_gpu=False
        )

        trainer_resume = DistillationTrainer(
            teacher=teacher,
            student=student,
            loss_fn=loss_fn,
            optimizer=optimizer_resume,
            scheduler=scheduler_resume,
            config=config_resume,
            device=device
        )

        print("  Running 1 more epoch...")
        trainer_resume.train(dataloader, start_epoch=2)

        # Verify epoch 3 checkpoint
        epoch3_checkpoint = os.path.join(temp_dir, "checkpoint_epoch_3.pt")
        assert os.path.exists(epoch3_checkpoint), "Epoch 3 checkpoint not found"
        print(f"  ✓ Resume training successful")

        # Final verification
        print("\n" + "="*60)
        print("✓ ALL PIPELINE TESTS PASSED")
        print("="*60)
        print("\nVerified components:")
        print("  ✓ Mock dataset creation")
        print("  ✓ Teacher and student initialization")
        print("  ✓ Loss function setup")
        print("  ✓ Optimizer and scheduler")
        print("  ✓ Training loop (2 epochs)")
        print("  ✓ Checkpoint saving")
        print("  ✓ Checkpoint loading")
        print("  ✓ Training resume")
        print("\nPipeline ready for production training!")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\nCleaning up temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"  ✓ Cleanup complete")


if __name__ == "__main__":
    import sys
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
