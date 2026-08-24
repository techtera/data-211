#!/usr/bin/env python3
"""
Sanity Check: Quick 3-5 epoch test to verify training pipeline.

Run this BEFORE full training to catch issues early.

Usage:
    python sanity_check.py --image_dir /path/to/images
    python sanity_check.py --image_dir /path/to/images --device cuda --epochs 3
"""

import argparse
import torch
import sys
import os

from student import StudentAggregator, initialize_student_from_dinov2
from training import (
    TrainingConfig,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    DistillationTrainer
)
from distillation import DistillationLoss
from load_real_teacher import load_real_teacher


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: Quick training test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to image directory'
    )
    parser.add_argument(
        '--teacher_checkpoint',
        type=str,
        default='../../vggt-unified/checkpoints/vggt_unified_fp16.pt',
        help='Path to teacher checkpoint'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs for sanity check'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda or cpu)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints_sanity',
        help='Checkpoint directory'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase 1 Distillation - Sanity Check")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # Verify image directory exists
    if not os.path.isdir(args.image_dir):
        print(f"\n✗ Error: Image directory not found: {args.image_dir}")
        return 1

    # Step 1: Load teacher
    print(f"\n{'='*60}")
    print("Step 1: Loading Teacher Encoder")
    print("="*60)

    try:
        teacher = load_real_teacher(args.teacher_checkpoint, args.device)
        print(f"✓ Teacher loaded")
    except Exception as e:
        print(f"✗ Failed to load teacher: {e}")
        return 1

    # Step 2: Initialize student
    print(f"\n{'='*60}")
    print("Step 2: Initializing Student Encoder")
    print("="*60)

    try:
        student = StudentAggregator()
        print("Initializing with DINOv2 pretrained weights...")
        initialize_student_from_dinov2(student, verbose=True)
        student = student.to(args.device)
        print(f"✓ Student initialized")
    except Exception as e:
        print(f"✗ Failed to initialize student: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Create dataloader
    print(f"\n{'='*60}")
    print("Step 3: Creating Dataloader")
    print("="*60)

    try:
        dataloader = create_dataloader(
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            num_workers=4,
            num_frames=8,
            image_size=518,
            shuffle=True,
            drop_last=True
        )
        print(f"✓ Dataloader created")
        print(f"  Dataset size: {len(dataloader.dataset)}")
        print(f"  Batches per epoch: {len(dataloader)}")
    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Setup training
    print(f"\n{'='*60}")
    print("Step 4: Setting Up Training Components")
    print("="*60)

    # Config
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        warmup_epochs=1,
        save_every=0,  # No periodic saves
        save_last=True,
        save_best=True,
        log_every=5,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_multi_gpu=False  # Disable multi-GPU for sanity check (avoids device mismatch issues)
    )

    # Loss function
    loss_fn = DistillationLoss(
        student_dim=1536,
        teacher_dim=2048,
        num_layers=4
    ).to(args.device)

    # Optimizer
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

    print(f"✓ Training components ready")

    # Step 5: Create trainer and run
    print(f"\n{'='*60}")
    print("Step 5: Running Sanity Check Training")
    print("="*60)

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device
    )

    # Train
    try:
        trainer.train(dataloader, start_epoch=0)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*60)
    print("✓ SANITY CHECK COMPLETE")
    print("="*60)
    print("\nCheckpoints saved:")
    print(f"  - {args.checkpoint_dir}/checkpoint_last.pt")
    print(f"  - {args.checkpoint_dir}/checkpoint_best.pt")
    print(f"  - {args.checkpoint_dir}/student_final.pt")
    print("\nNext steps:")
    print("  1. Verify loss decreased over epochs")
    print("  2. Check checkpoint files exist")
    print("  3. If all good, run full training with train.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
