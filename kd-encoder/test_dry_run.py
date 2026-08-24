#!/usr/bin/env python3
"""
Complete dry run with dummy data - test locally before VM deployment.
"""

import torch
import tempfile
import shutil
import os

from student import StudentAggregator, initialize_student_from_dinov2
from training import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    DistillationTrainer
)
from distillation import DistillationLoss
from load_real_teacher import load_real_teacher


class DummyDataset(torch.utils.data.Dataset):
    """Generate dummy images on the fly."""
    def __init__(self, num_samples=50, num_frames=8, image_size=518):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return [S, C, H, W]
        return torch.randn(self.num_frames, 3, self.image_size, self.image_size)


def test_dry_run():
    """Full dry run with 2 epochs."""
    print("="*60)
    print("Dry Run: Local Test with Dummy Data")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {temp_dir}")

    try:
        # Step 1: Create dummy dataloader
        print("\n[1] Creating dummy dataset...")
        dataset = DummyDataset(num_samples=20, num_frames=4, image_size=518)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  Batches: {len(dataloader)}")

        # Step 2: Load teacher
        print("\n[2] Loading teacher...")
        teacher = load_real_teacher(device=device)

        # Step 3: Initialize student
        print("\n[3] Initializing student...")
        student = StudentAggregator().to(device)
        print("  Initializing with DINOv2...")
        try:
            initialize_student_from_dinov2(student, verbose=False)
            print("  ✓ DINOv2 initialized")
        except Exception as e:
            print(f"  ⚠ DINOv2 init failed (OK for dry run): {e}")

        # Step 4: Setup training
        print("\n[4] Setting up training...")
        config = TrainingConfig(
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            warmup_epochs=1,
            save_every=0,
            save_last=True,
            save_best=True,
            log_every=2,
            checkpoint_dir=temp_dir,
            device=device,
            use_multi_gpu=False  # Force single GPU for dry run
        )

        loss_fn = DistillationLoss(
            student_dim=1536,
            teacher_dim=2048,
            num_layers=4
        ).to(device)

        params = list(student.parameters()) + list(loss_fn.parameters())
        optimizer = create_optimizer(params, learning_rate=config.learning_rate)

        scheduler = create_scheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
            steps_per_epoch=len(dataloader)
        )

        print("  ✓ Components ready")

        # Step 5: Create trainer
        print("\n[5] Running training (2 epochs)...")
        trainer = DistillationTrainer(
            teacher=teacher,
            student=student,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device
        )

        # Train
        trainer.train(dataloader, start_epoch=0)

        # Step 6: Verify checkpoints
        print("\n[6] Verifying outputs...")
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
        print(f"  Checkpoint files: {checkpoint_files}")

        assert 'checkpoint_last.pt' in checkpoint_files
        assert 'checkpoint_best.pt' in checkpoint_files
        assert 'student_final.pt' in checkpoint_files

        print("\n" + "="*60)
        print("✓ DRY RUN SUCCESSFUL")
        print("="*60)
        print("\nAll components working:")
        print("  ✓ Dummy data loading")
        print("  ✓ Teacher/student forward pass")
        print("  ✓ Loss computation")
        print("  ✓ Training loop (2 epochs)")
        print("  ✓ Checkpoint saving")
        print("\nReady for VM deployment!")

        return True

    except Exception as e:
        print(f"\n✗ Dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print(f"\nCleaning up...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    import sys
    success = test_dry_run()
    sys.exit(0 if success else 1)
