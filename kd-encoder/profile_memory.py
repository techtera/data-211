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

    # Forward teacher (memory-efficient: sample immediately)
    print("\n[5] Forward teacher and sample...")
    from distillation import sample_tokens, sample_tokens_with_indices

    with torch.no_grad():
        teacher_features, _ = teacher(images)
        teacher_features = [f for f in teacher_features if f is not None]
        print(f"  Teacher features: {len(teacher_features)} layers")
        print(f"  Shape per layer: {teacher_features[0].shape}")
        print_memory("After teacher forward")

        # Sample immediately to reduce memory
        teacher_sampled = []
        teacher_indices = []
        for t_feat in teacher_features:
            t_s, indices = sample_tokens(t_feat)
            teacher_sampled.append(t_s.detach())
            teacher_indices.append(indices)

        # Clear teacher features
        del teacher_features
        torch.cuda.empty_cache()
        print_memory("After teacher sampling + clear")

    # Forward student
    print("\n[6] Forward student...")
    student_features, _ = student(images)
    student_features = [f for f in student_features if f is not None]
    print(f"  Student features: {len(student_features)} layers")
    print(f"  Shape per layer: {student_features[0].shape}")
    print_memory("After student forward")

    # Sample student tokens
    print("\n[7] Sampling student tokens...")
    student_sampled = []
    for i, s_feat in enumerate(student_features):
        s_s = sample_tokens_with_indices(s_feat, teacher_indices[i])
        student_sampled.append(s_s)

    # Clear student features
    del student_features
    print(f"  Sampled shape: {teacher_sampled[0].shape}")
    print_memory("After student sampling")

    # Compute loss
    print("\n[8] Computing loss...")
    loss, metrics = loss_fn(student_sampled, teacher_sampled)
    print(f"  Loss: {loss.item():.4f}")
    print_memory("After loss computation")

    # Backward
    print("\n[9] Backward pass...")
    loss.backward()
    print_memory("After backward")

    # Clear
    del teacher_sampled, student_sampled, loss
    torch.cuda.empty_cache()
    print_memory("After clearing + cache")

    # Peak memory
    print("\n" + "="*60)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory: {peak:.2f}GB")
    print("="*60)

    # Memory breakdown
    print("\nMemory analysis:")
    print(f"  Batch size: 4")
    print(f"  Memory per sample: {peak/4:.2f}GB")
    print(f"  ---")
    print(f"  For 2 GPUs with batch_size=8 per GPU:")
    print(f"  Expected per GPU: ~{peak/4*8:.1f}GB")

    if peak < 20:
        print(f"\n✅ GOOD: Memory usage is acceptable!")
        print(f"   You can use batch_size=8-12 per GPU with 80GB A100")
    elif peak < 35:
        print(f"\n⚠️  MODERATE: Memory usage is higher than ideal but workable")
        print(f"   Recommended: batch_size=6-8 per GPU")
        print(f"   With gradient checkpointing: Memory will be more efficient during training")
    else:
        print(f"\n❌ HIGH: Memory usage is too high")
        print(f"   Reduce batch_size or investigate further")


if __name__ == '__main__':
    profile_memory()
