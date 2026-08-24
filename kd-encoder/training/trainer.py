"""
Main distillation training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation import (
    sample_tokens,
    sample_tokens_with_indices,
    DistillationLoss
)


def setup_multi_gpu(model: nn.Module, config, device: str) -> tuple:
    """
    Setup multi-GPU training if available.

    Args:
        model: Model to wrap
        config: TrainingConfig
        device: Primary device

    Returns:
        (wrapped_model, num_gpus, gpu_ids)
    """
    if not torch.cuda.is_available() or not config.use_multi_gpu:
        return model, 1, [0]

    # Get available GPUs
    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        return model, 1, [0]

    # Use specified GPUs or all available
    if config.gpu_ids is not None:
        gpu_ids = config.gpu_ids
        num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(num_gpus))

    # Wrap with DataParallel
    model = nn.DataParallel(model, device_ids=gpu_ids)

    return model, num_gpus, gpu_ids


class DistillationTrainer:
    """
    Handles training loop for knowledge distillation.

    Flow per step:
        1. Load batch of images
        2. Forward through teacher (no grad)
        3. Forward through student (with grad)
        4. Sample tokens (shared indices)
        5. Compute loss (projection + MSE + Cosine)
        6. Backward + optimizer step
        7. Log metrics
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        loss_fn: DistillationLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config,
        device: str = 'cuda'
    ):
        """
        Args:
            teacher: Teacher encoder (frozen)
            student: Student encoder (trainable)
            loss_fn: Distillation loss with projection
            optimizer: Optimizer
            scheduler: LR scheduler
            config: TrainingConfig
            device: Device
        """
        self.config = config
        self.device = device

        # Setup multi-GPU
        teacher = teacher.to(device).eval()
        student = student.to(device).train()
        loss_fn = loss_fn.to(device).train()

        # Wrap with DataParallel if multi-GPU
        self.teacher, self.num_gpus_teacher, self.gpu_ids = setup_multi_gpu(teacher, config, device)
        self.student, self.num_gpus_student, _ = setup_multi_gpu(student, config, device)
        self.loss_fn, self.num_gpus_loss, _ = setup_multi_gpu(loss_fn, config, device)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Multi-GPU info
        self.is_multi_gpu = self.num_gpus_student > 1

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """
        Train for one epoch.

        Returns:
            Epoch metrics dict
        """
        self.student.train()
        self.loss_fn.train()

        epoch_loss = 0.0
        epoch_metrics = {
            'mse': 0.0,
            'cosine_sim': 0.0,
        }

        start_time = time.time()

        for step, images in enumerate(dataloader):
            images = images.to(self.device)  # [B, S, C, H, W]

            # Forward teacher (no grad)
            with torch.no_grad():
                teacher_features, _ = self.teacher(images)  # List of [B, S, 1374, 2048]

            # Forward student (with grad)
            student_features_all, _ = self.student(images)  # List with None for uncached layers

            # Filter out None (only keep cached layers)
            student_features = [f for f in student_features_all if f is not None]

            # Verify we have matching number of features
            assert len(student_features) == len(teacher_features), \
                f"Mismatch: student has {len(student_features)} cached, teacher has {len(teacher_features)}"

            # Sample tokens with shared indices
            teacher_sampled = []
            student_sampled = []

            for i in range(len(teacher_features)):
                # Teacher: get features + indices
                t_sampled, indices = sample_tokens(teacher_features[i])
                teacher_sampled.append(t_sampled)

                # Student: use same indices
                s_sampled = sample_tokens_with_indices(student_features[i], indices)
                student_sampled.append(s_sampled)

            # Compute loss
            loss, metrics = self.loss_fn(student_sampled, teacher_sampled)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_metrics['mse'] += sum(metrics[f'layer_{i}_mse'] for i in range(4)) / 4
            epoch_metrics['cosine_sim'] += sum(metrics[f'layer_{i}_cosine_sim'] for i in range(4)) / 4

            # Log
            self.global_step += 1
            if (step + 1) % self.config.log_every == 0:
                lr = self.scheduler.get_lr()
                print(f"  Step {step+1}/{len(dataloader)}: "
                      f"Loss={loss.item():.4f}, LR={lr:.6f}")

        # Epoch averages
        num_steps = len(dataloader)
        epoch_loss /= num_steps
        for key in epoch_metrics:
            epoch_metrics[key] /= num_steps

        epoch_time = time.time() - start_time

        return {
            'loss': epoch_loss,
            'time': epoch_time,
            **epoch_metrics
        }

    def train(
        self,
        dataloader: DataLoader,
        start_epoch: int = 0
    ):
        """
        Full training loop.

        Args:
            dataloader: Training dataloader
            start_epoch: Starting epoch (for resuming)
        """
        print("="*60)
        print("Starting Phase 1 Distillation Training")
        print("="*60)
        print(f"Device: {self.device}")
        if self.is_multi_gpu:
            print(f"Multi-GPU: {self.num_gpus_student} GPUs (IDs: {self.gpu_ids})")
            print(f"Effective batch size: {self.config.batch_size * self.num_gpus_student}")
        else:
            print(f"Single GPU mode")
        print(f"Batch size per GPU: {self.config.batch_size}")
        print(f"Epochs: {start_epoch} → {self.config.num_epochs}")
        print(f"Steps per epoch: {len(dataloader)}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")
        print("="*60)

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("-"*60)

            # Train epoch
            epoch_metrics = self.train_epoch(dataloader)

            # Print summary
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Loss: {epoch_metrics['loss']:.6f}")
            print(f"    MSE: {epoch_metrics['mse']:.6f}")
            print(f"    Cosine Sim: {epoch_metrics['cosine_sim']:.6f}")
            print(f"    Time: {epoch_metrics['time']:.1f}s")
            print(f"    LR: {self.scheduler.get_lr():.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                from .checkpoints import save_checkpoint
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch+1}.pt"
                )

                # Unwrap DataParallel if needed
                student_to_save = self.student.module if self.is_multi_gpu else self.student
                projection_to_save = self.loss_fn.module.projection if self.is_multi_gpu else self.loss_fn.projection

                save_checkpoint(
                    student=student_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=epoch_metrics['loss'],
                    save_path=save_path,
                    projection=projection_to_save
                )

        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)

        # Save final model (unwrap DataParallel if needed)
        from .checkpoints import save_student_only
        final_path = os.path.join(self.config.checkpoint_dir, "student_final.pt")
        student_to_save = self.student.module if self.is_multi_gpu else self.student
        save_student_only(student_to_save, final_path)
