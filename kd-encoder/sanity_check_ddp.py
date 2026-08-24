#!/usr/bin/env python3
"""
Quick DDP sanity check: 3 epochs to verify everything works.

Usage:
    torchrun --nproc_per_node=2 sanity_check_ddp.py --image_dir train_images
"""

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
import os

from student import StudentAggregator, initialize_student_from_dinov2
from training import (
    TrainingConfig,
    create_optimizer,
    create_scheduler,
)
from training.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, get_rank
from training.dataset import ImageDataset
from distillation import DistillationLoss
from load_real_teacher import load_real_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str,
                       default='../../vggt-unified/checkpoints/vggt_unified_fp16.pt')
    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N steps (default: 10 for faster feedback)')

    args = parser.parse_args()

    # Get world size from environment
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    rank = int(os.environ.get('RANK', 0))

    # Setup DDP
    setup_ddp(rank, world_size)
    device = f'cuda:{rank}'

    if is_main_process():
        effective_batch = args.batch_size * world_size * args.gradient_accumulation_steps
        print("="*60, flush=True)
        print("DDP Sanity Check: 3 Epochs", flush=True)
        print("="*60, flush=True)
        print(f"GPUs: {world_size}", flush=True)
        print(f"Batch size per GPU: {args.batch_size}", flush=True)
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}", flush=True)
        print(f"Effective batch: {effective_batch}", flush=True)
        print("="*60, flush=True)

    try:
        # Load teacher
        if is_main_process():
            print("\n[1] Loading teacher...", flush=True)
        teacher = load_real_teacher(args.teacher_checkpoint, device)

        # Wrap teacher with FeaturesOnlyWrapper (NO DDP - teacher has no trainable params)
        from training.trainer import FeaturesOnlyWrapper
        teacher = FeaturesOnlyWrapper(teacher)
        # Don't wrap with DDP - teacher is frozen!

        # Initialize student
        if is_main_process():
            print("\n[2] Initializing student...", flush=True)
        student = StudentAggregator().to(device)
        initialize_student_from_dinov2(student, verbose=is_main_process())

        # Wrap student with FeaturesOnlyWrapper for DDP
        student = FeaturesOnlyWrapper(student)
        student = DDP(student, device_ids=[rank], find_unused_parameters=False)

        # Create dataloader
        if is_main_process():
            print("\n[3] Creating dataloader...", flush=True)

        dataset = ImageDataset(
            image_dir=args.image_dir,
            num_frames=8,
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
            num_workers=12,  # Increased for better I/O prefetching
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4  # Preload 4 batches per worker (default: 2)
        )

        if is_main_process():
            print(f"  Dataset: {len(dataset)} images", flush=True)
            print(f"  Batches per epoch: {len(dataloader)}", flush=True)

        # Setup training
        if is_main_process():
            print("\n[4] Setting up training...", flush=True)

        config = TrainingConfig(
            num_epochs=3,  # Only 3 epochs for sanity check
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            warmup_epochs=1,
            save_every=0,
            save_last=True,
            save_best=True,
            log_every=args.log_every,
            checkpoint_dir='checkpoints_sanity_ddp',
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

        # Training loop
        if is_main_process():
            print("\n[5] Running sanity check (3 epochs)...", flush=True)
            print("="*60, flush=True)

        from training.trainer import train_epoch_ddp

        best_loss = float('inf')

        for epoch in range(3):
            sampler.set_epoch(epoch)

            if is_main_process():
                print(f"\nEpoch {epoch+1}/3", flush=True)
                print("-"*60, flush=True)

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
                print(f"\n  Epoch {epoch+1} Loss: {epoch_loss:.6f}", flush=True)

                from training.checkpoints import save_checkpoint

                # Save last
                save_checkpoint(
                    student=student.module.model,  # Unwrap DDP and FeaturesOnlyWrapper
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_path=os.path.join(config.checkpoint_dir, "checkpoint_last.pt"),
                    projection=loss_fn.projection
                )

                # Save best
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(
                        student=student.module.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        save_path=os.path.join(config.checkpoint_dir, "checkpoint_best.pt"),
                        projection=loss_fn.projection
                    )
                    print(f"  ✓ New best! Loss: {best_loss:.6f}", flush=True)

        if is_main_process():
            print("\n" + "="*60, flush=True)
            print("✓ SANITY CHECK COMPLETE", flush=True)
            print("="*60, flush=True)
            print(f"\nFinal loss: {epoch_loss:.6f}", flush=True)
            print(f"Best loss: {best_loss:.6f}", flush=True)
            print(f"\nCheckpoints saved to: {config.checkpoint_dir}/", flush=True)
            print("\nIf loss decreased and no errors occurred:", flush=True)
            print("  ✅ Ready for full training!", flush=True)
            print("\nRun full training with:", flush=True)
            print("  torchrun --nproc_per_node=2 train_ddp.py \\", flush=True)
            print("    --image_dir train_images \\", flush=True)
            print("    --epochs 50 \\", flush=True)
            print(f"    --batch_size {args.batch_size} \\", flush=True)
            print(f"    --gradient_accumulation_steps {args.gradient_accumulation_steps}", flush=True)

    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
