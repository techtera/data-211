#!/usr/bin/env python3
"""
Profile memory usage during training to identify bottlenecks.
"""

import torch
import gc

from student import StudentAggregator
from load_real_teacher import load_real_teacher
from distillation import DistillationLoss


def print_memory(stage):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage:30s}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


def profile_memory():
    """Profile memory at each stage."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not torch.cuda.is_available():
        print("CUDA not available - skipping memory profiling")
        return

    print("="*60)
    print("Memory Profiling")
    print("="*60)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    print_memory("Initial")

    # Load teacher
    print("\n[1] Loading teacher...")
    teacher = load_real_teacher(device=device)
    print_memory("After teacher load")

    # Load student
    print("\n[2] Loading student...")
    student = StudentAggregator().to(device)
    print_memory("After student load")

    # Loss function
    print("\n[3] Loading loss function...")
    loss_fn = DistillationLoss().to(device)
    print_memory("After loss function")

    # Create dummy batch
    print("\n[4] Creating dummy batch...")
    batch_size = 4
    num_frames = 8
    images = torch.randn(batch_size, num_frames, 3, 518, 518, device=device)
    print_memory("After creating images")

    # Forward teacher
    print("\n[5] Forward teacher...")
    with torch.no_grad():
        teacher_features, _ = teacher(images)
    teacher_features = [f for f in teacher_features if f is not None]
    print(f"  Teacher features: {len(teacher_features)} layers")
    print(f"  Shape per layer: {teacher_features[0].shape}")
    print_memory("After teacher forward")

    # Forward student
    print("\n[6] Forward student...")
    student_features, _ = student(images)
    student_features = [f for f in student_features if f is not None]
    print(f"  Student features: {len(student_features)} layers")
    print(f"  Shape per layer: {student_features[0].shape}")
    print_memory("After student forward")

    # Sample tokens
    print("\n[7] Sampling tokens...")
    from distillation import sample_tokens, sample_tokens_with_indices
    teacher_sampled = []
    student_sampled = []
    for i in range(len(teacher_features)):
        t_s, indices = sample_tokens(teacher_features[i])
        teacher_sampled.append(t_s)
        s_s = sample_tokens_with_indices(student_features[i], indices)
        student_sampled.append(s_s)
    print(f"  Sampled shape: {teacher_sampled[0].shape}")
    print_memory("After token sampling")

    # Compute loss
    print("\n[8] Computing loss...")
    loss, metrics = loss_fn(student_sampled, teacher_sampled)
    print(f"  Loss: {loss.item():.4f}")
    print_memory("After loss computation")

    # Backward
    print("\n[9] Backward pass...")
    loss.backward()
    print_memory("After backward")

    # Peak memory
    print("\n" + "="*60)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory: {peak:.2f}GB")
    print("="*60)

    # Memory breakdown
    print("\nMemory breakdown estimates:")
    print(f"  Teacher params (909M × 2 bytes FP16): ~1.8GB")
    print(f"  Student params (255M × 4 bytes FP32): ~1.0GB")
    print(f"  Projection (12.6M × 4 bytes): ~0.05GB")
    print(f"  Images batch (4×8×3×518×518 × 4): ~0.10GB")
    print(f"  Teacher features (4×4×8×1374×2048 × 2): ~1.4GB")
    print(f"  Student features (4×4×8×1374×1536 × 4): ~1.0GB")
    print(f"  After sampling (4×4×8×133×2048 × 2): ~0.14GB")
    print(f"  Gradients (student + projection): ~1.0GB")
    print(f"  Optimizer states (2× params): ~2.0GB")
    print(f"  ---")
    print(f"  Expected total: ~8-10GB")
    print(f"  Actual peak: {peak:.2f}GB")

    if peak > 20:
        print("\n⚠️  WARNING: Memory usage is much higher than expected!")
        print("Possible causes:")
        print("  1. Keeping intermediate activations from all layers")
        print("  2. Not releasing teacher features before student forward")
        print("  3. Accumulating gradients across steps")
        print("\nSolutions:")
        print("  1. Use gradient checkpointing")
        print("  2. Clear cache after each step")
        print("  3. Reduce batch size")


if __name__ == '__main__':
    profile_memory()
