#!/usr/bin/env python3
"""
Test benchmarking metrics module.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from student import StudentAggregator, initialize_student_from_dinov2
from benchmarking.metrics import (
    count_parameters,
    measure_latency,
    measure_memory,
    calculate_throughput,
    format_number
)


def test_metrics():
    """Test all metrics functions."""
    print("="*60)
    print("Testing Benchmarking Metrics")
    print("="*60)

    # Create and initialize student
    print("\n[Setup] Creating and initializing student encoder...")
    student = StudentAggregator()
    initialize_student_from_dinov2(student, verbose=False)
    print("✓ Student encoder ready")

    # Test 1: Parameter counting
    print("\n[Test 1] Testing parameter counting...")
    try:
        params = count_parameters(student)

        print(f"  Total parameters: {format_number(params['total'])}")
        print(f"  Breakdown:")
        print(f"    - Patch embed: {format_number(params['patch_embed'])}")
        print(f"    - Frame blocks: {format_number(params['frame_blocks'])}")
        print(f"    - Global blocks: {format_number(params['global_blocks'])}")
        print(f"    - Special tokens: {format_number(params['special_tokens'])}")

        # Verify total matches
        calculated_total = sum(v for k, v in params.items() if k != 'total')
        if abs(calculated_total - params['total']) > 100:  # Allow small diff
            print(f"  ⚠ Warning: Breakdown sum ({calculated_total:,}) doesn't match total ({params['total']:,})")
        else:
            print(f"  ✓ Parameter counting verified")
    except Exception as e:
        print(f"✗ Parameter counting failed: {e}")
        return False

    # Test 2: Latency measurement
    print("\n[Test 2] Testing latency measurement...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    try:
        # Small input for quick test
        B, S = 1, 2
        input_tensor = torch.rand(B, S, 3, 518, 518)

        # Quick measurement (few iterations)
        latency_stats = measure_latency(
            student,
            input_tensor,
            device=device,
            warmup=5,
            iters=10
        )

        print(f"  Latency statistics (10 iterations):")
        print(f"    - Mean: {latency_stats['mean_ms']:.2f} ms")
        print(f"    - Median: {latency_stats['median_ms']:.2f} ms")
        print(f"    - Std: {latency_stats['std_ms']:.2f} ms")
        print(f"    - Min: {latency_stats['min_ms']:.2f} ms")
        print(f"    - Max: {latency_stats['max_ms']:.2f} ms")
        print(f"    - P95: {latency_stats['p95_ms']:.2f} ms")
        print(f"    - P99: {latency_stats['p99_ms']:.2f} ms")

        # Verify reasonable values
        if latency_stats['mean_ms'] <= 0:
            print(f"  ✗ Invalid latency: {latency_stats['mean_ms']}")
            return False

        print(f"  ✓ Latency measurement successful")
    except Exception as e:
        print(f"✗ Latency measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Memory measurement
    print("\n[Test 3] Testing memory measurement...")
    try:
        memory_gb = measure_memory(student, input_tensor, device=device)

        if device == 'cuda':
            print(f"  Peak memory: {memory_gb:.3f} GB")

            if memory_gb <= 0:
                print(f"  ✗ Invalid memory measurement: {memory_gb}")
                return False

            print(f"  ✓ Memory measurement successful")
        else:
            print(f"  ⚠ Memory measurement skipped (CPU only)")
    except Exception as e:
        print(f"✗ Memory measurement failed: {e}")
        return False

    # Test 4: Throughput calculation
    print("\n[Test 4] Testing throughput calculation...")
    try:
        throughput = calculate_throughput(latency_stats['mean_ms'])
        print(f"  Throughput: {throughput:.2f} FPS")

        if throughput <= 0:
            print(f"  ✗ Invalid throughput: {throughput}")
            return False

        print(f"  ✓ Throughput calculation successful")
    except Exception as e:
        print(f"✗ Throughput calculation failed: {e}")
        return False

    # Test 5: Number formatting
    print("\n[Test 5] Testing number formatting...")
    try:
        test_cases = [
            (1_500_000_000, "1.50B"),
            (255_687_936, "255.69M"),
            (1_500, "1.50K"),
            (42.5, "42.50"),
        ]

        all_passed = True
        for num, expected_prefix in test_cases:
            formatted = format_number(num)
            if not formatted.startswith(expected_prefix):
                print(f"  ✗ format_number({num}) = {formatted}, expected prefix {expected_prefix}")
                all_passed = False

        if all_passed:
            print(f"  ✓ Number formatting correct")
        else:
            return False
    except Exception as e:
        print(f"✗ Number formatting failed: {e}")
        return False

    # All tests passed!
    print("\n" + "="*60)
    print("✓ ALL METRICS TESTS PASSED")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)
