#!/usr/bin/env python3
"""
Phase 0A: Student Encoder Benchmarking

Measures student encoder performance before training.
Generates GO/NO-GO decision for Phase 1.

Usage:
    python benchmark_student.py --device cuda
    python benchmark_student.py --device cpu --quick  # Fast test with fewer iterations
"""

import argparse
import sys
import torch
import torch.nn as nn

from student import StudentAggregator, initialize_student_from_dinov2
from benchmarking import benchmark_model, compare_models, generate_report


class MockTeacher(nn.Module):
    """
    Mock teacher encoder for testing when real checkpoint unavailable.
    Simulates teacher architecture: 1024 dim, 24 layers.
    """
    def __init__(self):
        super().__init__()
        # Simplified mock - just enough to count parameters
        self.patch_embed = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
        self.frame_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True)
            for _ in range(24)
        ])
        self.global_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True)
            for _ in range(24)
        ])
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, 1024))
        self.register_token = nn.Parameter(torch.randn(1, 2, 4, 1024))

    def forward(self, images):
        # Simplified forward - just return dummy output
        B, S, C, H, W = images.shape
        # Return list of cached outputs (mimics real teacher)
        dummy_output = torch.zeros(B, S, 1374, 2048, device=images.device)
        return [dummy_output] * 4, 5  # 4 cached layers, patch_start_idx=5


def load_teacher(checkpoint_path: str = None, device: str = 'cuda', verbose: bool = True):
    """
    Load teacher encoder.

    Args:
        checkpoint_path: Path to teacher checkpoint (optional)
        device: Device to load model on
        verbose: Print messages

    Returns:
        Teacher model
    """
    if checkpoint_path and checkpoint_path != 'mock':
        if verbose:
            print(f"\nLoading teacher from checkpoint: {checkpoint_path}")
        try:
            # Try to load real teacher checkpoint
            # This requires the full VGGT model code
            raise NotImplementedError("Real teacher loading not implemented yet")
        except Exception as e:
            if verbose:
                print(f"⚠ Could not load teacher checkpoint: {e}")
                print(f"⚠ Falling back to mock teacher")
            checkpoint_path = 'mock'

    if verbose:
        print(f"\nUsing mock teacher for benchmarking")
        print(f"Note: Mock teacher has correct architecture for parameter counting")
        print(f"      but latency/memory may differ from real teacher")

    teacher = MockTeacher()
    teacher = teacher.to(device).eval()

    if verbose:
        print(f"✓ Mock teacher loaded")

    return teacher


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0A: Benchmark student encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device for benchmarking'
    )
    parser.add_argument(
        '--teacher_checkpoint',
        type=str,
        default='mock',
        help='Path to teacher checkpoint (use "mock" for testing)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8,
        help='Number of frames per sample'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (fewer iterations)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/benchmark_report.md',
        help='Output path for report'
    )

    args = parser.parse_args()

    # Set iteration counts
    if args.quick:
        warmup_iters = 5
        measurement_iters = 10
        print("\n⚡ Quick mode: Using fewer iterations for faster testing")
    else:
        warmup_iters = 20
        measurement_iters = 100

    print("="*60)
    print("Phase 0A: Student Encoder Benchmarking")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Device: {args.device}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Frames per sample: {args.num_frames}")
    print(f"  - Warmup iterations: {warmup_iters}")
    print(f"  - Measurement iterations: {measurement_iters}")
    print(f"  - Output: {args.output}")

    # 1. Load teacher encoder
    print(f"\n{'='*60}")
    print(f"Step 1: Loading Teacher Encoder")
    print(f"{'='*60}")

    try:
        teacher = load_teacher(args.teacher_checkpoint, args.device, verbose=True)
    except Exception as e:
        print(f"✗ Failed to load teacher: {e}")
        return 1

    # 2. Create and initialize student encoder
    print(f"\n{'='*60}")
    print(f"Step 2: Initializing Student Encoder")
    print(f"{'='*60}")

    try:
        print(f"\nCreating student encoder...")
        student = StudentAggregator()

        print(f"\nInitializing with DINOv2 pretrained weights...")
        initialize_student_from_dinov2(student, verbose=True)

        student = student.to(args.device).eval()
        print(f"\n✓ Student encoder ready")
    except Exception as e:
        print(f"✗ Failed to initialize student: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 3. Benchmark teacher
    print(f"\n{'='*60}")
    print(f"Step 3: Benchmarking Models")
    print(f"{'='*60}")

    try:
        teacher_results = benchmark_model(
            teacher,
            model_name="Teacher",
            device=args.device,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            warmup_iters=warmup_iters,
            measurement_iters=measurement_iters,
            verbose=True
        )
    except Exception as e:
        print(f"✗ Teacher benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. Benchmark student
    try:
        student_results = benchmark_model(
            student,
            model_name="Student",
            device=args.device,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            warmup_iters=warmup_iters,
            measurement_iters=measurement_iters,
            verbose=True
        )
    except Exception as e:
        print(f"✗ Student benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 5. Compare and evaluate
    print(f"\n{'='*60}")
    print(f"Step 4: Comparison and Evaluation")
    print(f"{'='*60}")

    try:
        comparison = compare_models(student_results, teacher_results, verbose=True)
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 6. Generate report
    print(f"\n{'='*60}")
    print(f"Step 5: Generating Report")
    print(f"{'='*60}")

    try:
        generate_report(student_results, teacher_results, comparison, output_path=args.output)
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 7. Final decision
    print(f"\n{'='*60}")
    print(f"PHASE 0A COMPLETE")
    print(f"{'='*60}")

    if comparison['meets_targets']:
        print(f"\n✓ GO - All targets met!")
        print(f"\nNext steps:")
        print(f"  1. Review report: {args.output}")
        print(f"  2. Proceed to Phase 1 (distillation training)")
        return 0
    else:
        print(f"\n✗ NO-GO - Targets not met")
        print(f"\nNext steps:")
        print(f"  1. Review report: {args.output}")
        print(f"  2. Redesign architecture")
        print(f"  3. Re-run Phase 0A")
        return 1


if __name__ == '__main__':
    sys.exit(main())
