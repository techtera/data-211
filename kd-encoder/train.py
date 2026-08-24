#!/usr/bin/env python3
"""
Full Training: 40-50 epoch distillation training.

Run this AFTER sanity check passes.

Usage:
    python train.py --image_dir /path/to/images
    python train.py --image_dir /path/to/images --epochs 50 --batch_size 4
    python train.py --resume_from checkpoints/checkpoint_last.pt  # Resume training
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
    DistillationTrainer,
    load_checkpoint
)
from distillation import DistillationLoss
from load_real_teacher import load_real_teacher


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Full distillation training",
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
        default=50,
        help='Total number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=5,
        help='Warmup epochs'
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
        default='checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader workers'
    )
    parser.add_argument(
        '--log_every',
        type=int,
        default=10,
        help='Log every N steps'
    )
    parser.add_argument(
        '--use_multi_gpu',
        action='store_true',
        default=True,
        help='Use multiple GPUs if available'
    )

    args = parser.parse_args()

    print("="*60)
    print("Phase 1: Full Distillation Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Multi-GPU: {args.use_multi_gpu}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    if args.resume_from:
        print(f"  Resume from: {args.resume_from}")

    # Verify image directory exists
    if not os.path.isdir(args.image_dir):
        print(f"\n✗ Error: Image directory not found: {args.image_dir}")
        return 1

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Step 1: Load teacher
    print(f"\n{'='*60}")
    print("Step 1: Loading Teacher Encoder")
    print("="*60)

    try:
        teacher = load_real_teacher(args.teacher_checkpoint, args.device)
        teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
        print(f"✓ Teacher loaded ({teacher_params:.1f}M params)")
    except Exception as e:
        print(f"✗ Failed to load teacher: {e}")
        return 1

    # Step 2: Initialize student
    print(f"\n{'='*60}")
    print("Step 2: Initializing Student Encoder")
    print("="*60)

    try:
        student = StudentAggregator()

        if args.resume_from:
            print(f"Skipping DINOv2 init (will load from checkpoint)")
        else:
            print("Initializing with DINOv2 pretrained weights...")
            initialize_student_from_dinov2(student, verbose=True)

        student = student.to(args.device)
        student_params = sum(p.numel() for p in student.parameters()) / 1e6
        print(f"✓ Student initialized ({student_params:.1f}M params)")
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
            num_workers=args.num_workers,
            num_frames=8,
            image_size=518,
            shuffle=True,
            drop_last=True
        )
        print(f"✓ Dataloader created")
        print(f"  Dataset size: {len(dataloader.dataset)}")
        print(f"  Batches per epoch: {len(dataloader)}")
        print(f"  Total steps: {len(dataloader) * args.epochs}")
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
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        save_every=0,  # Only last/best
        save_last=True,
        save_best=True,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_multi_gpu=args.use_multi_gpu
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

    # Step 5: Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"\n{'='*60}")
        print("Step 5: Loading Checkpoint")
        print("="*60)

        try:
            checkpoint_data = load_checkpoint(
                checkpoint_path=args.resume_from,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                projection=loss_fn.projection,
                device=args.device
            )
            start_epoch = checkpoint_data['epoch']
            print(f"✓ Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return 1

    # Step 6: Create trainer and run
    print(f"\n{'='*60}")
    print(f"Step {6 if args.resume_from else 5}: Starting Training")
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
        trainer.train(dataloader, start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print(f"Last checkpoint saved: {args.checkpoint_dir}/checkpoint_last.pt")
        print(f"Resume with: python train.py --resume_from {args.checkpoint_dir}/checkpoint_last.pt")
        return 1
    except Exception as e:
        print(f"\n\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print("\nCheckpoints saved:")
    print(f"  - {args.checkpoint_dir}/checkpoint_last.pt (final epoch)")
    print(f"  - {args.checkpoint_dir}/checkpoint_best.pt (best loss)")
    print(f"  - {args.checkpoint_dir}/student_final.pt (student weights only)")
    print("\nNext steps:")
    print("  1. Use checkpoint_best.pt for best performance")
    print("  2. Extract student encoder: student_final.pt")
    print("  3. Train decoder heads (edge-mask, obj-mask)")
    print("  4. Compare with original VGGT encoder")

    return 0


if __name__ == '__main__':
    sys.exit(main())
