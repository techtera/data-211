#!/usr/bin/env python3
"""
DDP Training: Multi-GPU training with DistributedDataParallel.

Better than DataParallel: more efficient, better scaling, lower memory per GPU.

Usage:
    # 2 GPUs
    torchrun --nproc_per_node=2 train_ddp.py --image_dir train_images --epochs 50

    # Or with python -m
    python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py --image_dir train_images
"""

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
import os

from student import StudentAggregator, initialize_student_from_dinov2_large
from training import (
    TrainingConfig,
    create_dataloader,
    create_optimizer,
    create_scheduler,
    load_checkpoint
)
from training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, get_rank
from distillation import DistillationLoss
from load_real_teacher import load_real_teacher


def train_ddp(rank, world_size, args):
    """
    DDP training function (runs on each GPU).

    Args:
        rank: Current GPU rank
        world_size: Total number of GPUs
        args: Training arguments
    """
    # Setup DDP
    setup_ddp(rank, world_size)
    device = f'cuda:{rank}'

    if is_main_process():
        effective_batch = args.batch_size * world_size * args.gradient_accumulation_steps
        print("="*60)
        print(f"DDP Training: {world_size} GPUs")
        print("="*60)
        print(f"Image directory: {args.image_dir}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch} ({args.batch_size}×{world_size}×{args.gradient_accumulation_steps})")
        print(f"Learning rate: {args.learning_rate}")

    try:
        # Load teacher
        if is_main_process():
            print("\n[1] Loading teacher...")
        teacher = load_real_teacher(args.teacher_checkpoint, device)

        # Wrap with FeaturesOnlyWrapper (NO DDP - teacher has no trainable params)
        from training.trainer import FeaturesOnlyWrapper
        teacher = FeaturesOnlyWrapper(teacher)
        # Don't wrap with DDP - teacher is frozen!

        # Initialize student
        if is_main_process():
            print("\n[2] Initializing student...")
        student = StudentAggregator(embed_dim=768, depth=18).to(device)
        if not args.resume_from:
            initialize_student_from_dinov2_large(student, verbose=is_main_process())

        # Wrap student with FeaturesOnlyWrapper then DDP (consistent with sanity_check_ddp.py)
        from training.trainer import FeaturesOnlyWrapper
        student = FeaturesOnlyWrapper(student)
        student = DDP(student, device_ids=[rank], find_unused_parameters=False)

        # Create dataloader with DistributedSampler
        if is_main_process():
            print("\n[3] Creating dataloader...")

        from training.dataset import ImageDataset
        dataset = ImageDataset(
            image_dir=args.image_dir,
            num_frames=1,  # Single frame per sample (images are unrelated)
            image_size=518
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4  # Preload 4 batches per worker
        )

        if is_main_process():
            print(f"  Dataset: {len(dataset)} images")
            print(f"  Per GPU: {len(dataset) // world_size} images")
            print(f"  Batches per epoch: {len(dataloader)}")

        # Setup training
        if is_main_process():
            print("\n[4] Setting up training...")

        config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_epochs=args.warmup_epochs,
            save_every=0,
            save_last=True,
            save_best=True,
            log_every=args.log_every,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            use_multi_gpu=False  # DDP handles this
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

        # Resume if specified
        start_epoch = 0
        if args.resume_from and is_main_process():
            print(f"\n[5] Loading checkpoint...")
            checkpoint_data = load_checkpoint(
                checkpoint_path=args.resume_from,
                student=student.module.model,  # Unwrap DDP -> FeaturesOnlyWrapper -> model
                optimizer=optimizer,
                scheduler=scheduler,
                projection=loss_fn.projection,
                device=device
            )
            start_epoch = checkpoint_data['epoch']

        # Training loop
        if is_main_process():
            print(f"\n[{6 if args.resume_from else 5}] Starting training...")
            print("="*60)

        from training.trainer import train_epoch_ddp

        best_loss = float('inf')

        for epoch in range(start_epoch, config.num_epochs):
            sampler.set_epoch(epoch)  # Shuffle differently each epoch

            if is_main_process():
                print(f"\nEpoch {epoch+1}/{config.num_epochs}")
                print("-"*60)

            # Train epoch
            epoch_loss = train_epoch_ddp(
                teacher=teacher,
                student=student,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=dataloader,
                device=device,
                epoch=epoch,
                config=config
            )

            # Save checkpoints (main process only)
            if is_main_process():
                print(f"\n  Epoch {epoch+1} Loss: {epoch_loss:.6f}")

                from training.checkpoints import save_checkpoint, save_student_only

                # Save last
                if config.save_last:
                    save_checkpoint(
                        student=student.module.model,  # Unwrap DDP -> FeaturesOnlyWrapper -> model
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        save_path=os.path.join(config.checkpoint_dir, "checkpoint_last.pt"),
                        projection=loss_fn.projection
                    )
                    print(f"  ✓ Checkpoint saved (last)")

                # Save best
                if config.save_best and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(
                        student=student.module.model,  # Unwrap DDP -> FeaturesOnlyWrapper -> model
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        save_path=os.path.join(config.checkpoint_dir, "checkpoint_best.pt"),
                        projection=loss_fn.projection
                    )
                    print(f"  ✓ New best checkpoint! Loss: {best_loss:.6f}")

        # Save final
        if is_main_process():
            print("\n" + "="*60)
            print("✓ Training Complete!")
            print("="*60)
            save_student_only(
                student.module.model,  # Unwrap DDP -> FeaturesOnlyWrapper -> model
                os.path.join(config.checkpoint_dir, "student_final.pt")
            )

    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str,
                       default='../../vggt-unified/checkpoints/vggt_unified_fp16.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size PER GPU (with num_frames=1, can use much larger batches)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps (effective_batch = batch_size × num_gpus × accum_steps)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--log_every', type=int, default=100)

    args = parser.parse_args()

    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    rank = int(os.environ.get('RANK', 0))

    train_ddp(rank, world_size, args)


if __name__ == '__main__':
    main()
