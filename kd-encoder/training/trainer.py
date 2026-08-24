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

try:
    from .ddp_utils import reduce_tensor, is_main_process
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False


class FeaturesOnlyWrapper(nn.Module):
    """
    Wrapper that only returns features (not patch_start_idx) for DataParallel compatibility.

    DataParallel can't gather integers across GPUs, so we drop the patch_start_idx return value.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        features, _ = self.model(images)  # Drop patch_start_idx
        return features


def setup_multi_gpu(model: nn.Module, config, device: str) -> tuple:
    """
    Setup multi-GPU training if available.

    Always wraps with FeaturesOnlyWrapper to provide consistent interface.

    Args:
        model: Model to wrap
        config: TrainingConfig
        device: Primary device

    Returns:
        (wrapped_model, num_gpus, gpu_ids)
    """
    # Always wrap with FeaturesOnlyWrapper for consistent return values
    # (returns only features, drops patch_start_idx)
    model = FeaturesOnlyWrapper(model)

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

    # Wrap with DataParallel (already wrapped with FeaturesOnlyWrapper above)
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

        # Wrap encoders with DataParallel if multi-GPU
        # NOTE: loss_fn should NOT be wrapped - it operates on features, not images
        self.teacher, self.num_gpus_teacher, self.gpu_ids = setup_multi_gpu(teacher, config, device)
        self.student, self.num_gpus_student, _ = setup_multi_gpu(student, config, device)
        self.loss_fn = loss_fn  # Keep on single device

        self.optimizer = optimizer
        self.scheduler = scheduler

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')  # Track best loss for best checkpoint

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

            # Forward teacher (no grad) - sample tokens immediately to reduce memory
            with torch.no_grad():
                teacher_features_all = self.teacher(images)  # List with None for uncached layers
                teacher_features = [f for f in teacher_features_all if f is not None]

                # Sample teacher tokens immediately (reduces memory 10x: 1374→133 tokens)
                teacher_sampled = []
                teacher_indices = []
                for t_feat in teacher_features:
                    t_sampled, indices = sample_tokens(t_feat)
                    teacher_sampled.append(t_sampled.detach())  # Detach to free computation graph
                    teacher_indices.append(indices)

                # Clear teacher features immediately to free memory
                del teacher_features_all, teacher_features
                torch.cuda.empty_cache()

            # Forward student (with grad)
            student_features_all = self.student(images)  # List with None for uncached layers
            student_features = [f for f in student_features_all if f is not None]
            del student_features_all  # Free memory

            # Sample student tokens with same indices as teacher
            student_sampled = []
            for i, s_feat in enumerate(student_features):
                s_sampled = sample_tokens_with_indices(s_feat, teacher_indices[i])
                student_sampled.append(s_sampled)

            # Clear student features to free memory before loss computation
            del student_features

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

            # Clear sampled features and free memory
            del teacher_sampled, student_sampled, loss
            if (step + 1) % 100 == 0:  # Periodic cache clearing
                torch.cuda.empty_cache()

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

            # Unwrap DataParallel and FeaturesOnlyWrapper if needed (for all checkpoint types)
            if self.is_multi_gpu:
                # Unwrap: DataParallel -> FeaturesOnlyWrapper -> actual model
                student_to_save = self.student.module.model
            else:
                # Single GPU: Unwrap FeaturesOnlyWrapper -> actual model
                student_to_save = self.student.model

            # Loss function is never wrapped with DataParallel
            projection_to_save = self.loss_fn.projection

            from .checkpoints import save_checkpoint

            # 1. Save periodic checkpoint (every N epochs)
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                save_checkpoint(
                    student=student_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=epoch_metrics['loss'],
                    save_path=save_path,
                    projection=projection_to_save
                )
                print(f"  ✓ Periodic checkpoint saved (epoch {epoch+1})")

            # 2. Save last checkpoint (always, overwrites previous)
            if self.config.save_last:
                last_path = os.path.join(self.config.checkpoint_dir, "checkpoint_last.pt")
                save_checkpoint(
                    student=student_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=epoch_metrics['loss'],
                    save_path=last_path,
                    projection=projection_to_save
                )

            # 3. Save best checkpoint (when loss improves)
            if self.config.save_best and epoch_metrics['loss'] < self.best_loss:
                self.best_loss = epoch_metrics['loss']
                best_path = os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt")
                save_checkpoint(
                    student=student_to_save,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    loss=epoch_metrics['loss'],
                    save_path=best_path,
                    projection=projection_to_save
                )
                print(f"  ✓ New best checkpoint! Loss: {self.best_loss:.6f}")

        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)

        # Save final model (unwrap DataParallel and FeaturesOnlyWrapper if needed)
        from .checkpoints import save_student_only
        final_path = os.path.join(self.config.checkpoint_dir, "student_final.pt")
        if self.is_multi_gpu:
            student_to_save = self.student.module.model  # Unwrap DataParallel -> FeaturesOnlyWrapper -> model
        else:
            student_to_save = self.student.model  # Unwrap FeaturesOnlyWrapper -> model
        save_student_only(student_to_save, final_path)


def train_epoch_ddp(teacher, student, loss_fn, optimizer, scheduler, dataloader, device, epoch, config):
    """
    Train one epoch with DDP.

    Args:
        teacher: DDP-wrapped teacher
        student: DDP-wrapped student
        loss_fn: Loss function (not wrapped)
        optimizer: Optimizer
        scheduler: LR scheduler
        dataloader: DataLoader with DistributedSampler
        device: Current device
        epoch: Current epoch
        config: TrainingConfig

    Returns:
        Average loss for epoch
    """
    student.train()
    loss_fn.train()

    epoch_loss = 0.0
    num_steps = 0

    for step, images in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        # Forward teacher (no grad)
        with torch.no_grad():
            teacher_features_all = teacher(images)

        # Forward student
        student_features_all = student(images)

        # Filter None
        teacher_features = [f for f in teacher_features_all if f is not None]
        student_features = [f for f in student_features_all if f is not None]

        # Sample tokens
        teacher_sampled = []
        student_sampled = []
        for i in range(len(teacher_features)):
            t_sampled, indices = sample_tokens(teacher_features[i])
            teacher_sampled.append(t_sampled)
            s_sampled = sample_tokens_with_indices(student_features[i], indices)
            student_sampled.append(s_sampled)

        # Compute loss
        loss, metrics = loss_fn(student_sampled, teacher_sampled)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate
        epoch_loss += loss.item()
        num_steps += 1

        # Log
        if is_main_process() and (step + 1) % config.log_every == 0:
            lr = scheduler.get_lr()
            print(f"  Step {step+1}/{len(dataloader)}: Loss={loss.item():.4f}, LR={lr:.6f}")

    # Average across all processes
    epoch_loss /= num_steps

    return epoch_loss
