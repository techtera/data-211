#!/usr/bin/env python3
"""
DDP Training for Edge Decoder with Student Encoder.

Multi-GPU training with DistributedDataParallel.

Usage:
    # 2 GPUs
    torchrun --nproc_per_node=2 train_ddp.py --epochs 100

    # Single GPU (fallback)
    python fine_tune.py
"""

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
import os

sys.path.insert(0, "../kd-encoder")

from student import StudentAggregator
from edge_mask.model import StudentEdgeMask
from fine_tuning.config import *
from fine_tuning.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, get_rank
from fine_tuning.dataset import EdgeMaskDataset
from fine_tuning.losses import EdgeLoss
from fine_tuning.optimizer import build_optimizer
from fine_tuning.scheduler import build_scheduler
from fine_tuning.checkpoints import save_checkpoint


def train_epoch_ddp(model, criterion, optimizer, scheduler, dataloader, device, epoch, config):
    """Train one epoch with DDP."""
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward
        logits, ds1_logits, ds2_logits = model(images)

        # Loss
        loss = criterion(logits, ds1_logits, ds2_logits, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        # Logging (main process only)
        if is_main_process() and (batch_idx + 1) % LOG_EVERY == 0:
            lr = optimizer.param_groups[0]['lr']
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"  [{batch_idx+1}/{len(dataloader)}] Loss: {avg_loss:.4f}, LR: {lr:.6f}")

    return epoch_loss / len(dataloader)


def validate_ddp(model, criterion, dataloader, device):
    """Validate with DDP."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits, ds1_logits, ds2_logits = model(images)
            loss = criterion(logits, ds1_logits, ds2_logits, masks)
            val_loss += loss.item()

    return val_loss / len(dataloader)


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
        effective_batch = BATCH_SIZE * world_size
        print("="*60)
        print(f"Edge Decoder DDP Training: {world_size} GPUs")
        print("="*60)
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Learning rate: {LEARNING_RATE}")

    try:
        # Load student encoder
        if is_main_process():
            print("\n[1] Loading student encoder...")

        checkpoint = torch.load(STUDENT_CHECKPOINT, map_location='cpu')
        state_dict = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict', checkpoint))

        student_aggregator = StudentAggregator()
        student_aggregator.load_state_dict(state_dict)
        student_aggregator.eval()
        student_aggregator.requires_grad_(False)

        if is_main_process():
            student_params = sum(p.numel() for p in student_aggregator.parameters())
            print(f"  ✓ Student encoder loaded: {student_params:,} parameters")

        # Build model
        if is_main_process():
            print("\n[2] Building edge decoder...")

        model = StudentEdgeMask(student_aggregator).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        if is_main_process():
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  ✓ Trainable parameters: {trainable:,}")

        # Create dataset
        if is_main_process():
            print("\n[3] Creating dataloaders...")

        from fine_tuning.dataloader import build_dataloaders

        # Get datasets (without DataLoader wrapper)
        from torch.utils.data import random_split
        from pathlib import Path

        dataset = EdgeMaskDataset(Path(DATASET_ROOT))
        train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
        val_size = len(dataset) - train_size

        torch.manual_seed(RANDOM_SEED)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DDP samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        if is_main_process():
            print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
            print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")

        # Setup training
        if is_main_process():
            print("\n[4] Setting up training...")

        from fine_tuning.losses import build_loss
        criterion = build_loss()

        optimizer = build_optimizer(model)
        total_steps = NUM_EPOCHS * len(train_loader)
        scheduler, warmup_steps = build_scheduler(optimizer, total_steps)

        # Training loop
        if is_main_process():
            print(f"\n[5] Starting training...")
            print("="*60)

        best_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)  # Shuffle differently each epoch

            if is_main_process():
                print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
                print("-"*60)

            # Train
            train_loss = train_epoch_ddp(
                model, criterion, optimizer, scheduler,
                train_loader, device, epoch, None
            )

            # Validate
            val_loss = validate_ddp(model, criterion, val_loader, device)

            # Save checkpoints (main process only)
            if is_main_process():
                print(f"\n  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save last
                if SAVE_LATEST:
                    save_checkpoint(
                        model=model.module,  # Unwrap DDP
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        loss=val_loss,
                        save_path=os.path.join(CHECKPOINT_DIR, "checkpoint_last.pt")
                    )

                # Save best
                if SAVE_BEST and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model=model.module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        loss=val_loss,
                        save_path=os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
                    )
                    print(f"  ✓ New best checkpoint! Val Loss: {best_val_loss:.4f}")

        if is_main_process():
            print("\n" + "="*60)
            print("✓ Training Complete!")
            print("="*60)

    finally:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
    rank = int(os.environ.get('RANK', 0))

    train_ddp(rank, world_size, args)


if __name__ == '__main__':
    main()
