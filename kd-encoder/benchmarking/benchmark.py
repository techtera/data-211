# Orchestrate benchmarking for Phase 0A
# Runs all measurements and compares student vs teacher

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .metrics import (
    count_parameters,
    measure_latency,
    measure_memory,
    calculate_throughput,
    format_number
)


def benchmark_model(
    model: nn.Module,
    model_name: str,
    device: str = 'cuda',
    batch_size: int = 1,
    num_frames: int = 8,
    warmup_iters: int = 20,
    measurement_iters: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Run complete benchmark suite on a model.

    Args:
        model: Model to benchmark
        model_name: Name for display (e.g., "Teacher" or "Student")
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        num_frames: Number of frames per sample
        warmup_iters: Warmup iterations for latency
        measurement_iters: Measurement iterations for latency
        verbose: Print progress messages

    Returns:
        Dictionary with all benchmark results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_name}")
        print(f"{'='*60}")

    results = {
        'model_name': model_name,
        'device': device,
        'batch_size': batch_size,
        'num_frames': num_frames,
    }

    # 1. Parameter counting
    if verbose:
        print(f"\n[1/3] Counting parameters...")

    params = count_parameters(model)
    results['parameters'] = params

    if verbose:
        print(f"  Total: {format_number(params['total'])}")
        print(f"  Breakdown:")
        for key, value in params.items():
            if key != 'total':
                print(f"    - {key}: {format_number(value)}")

    # 2. Latency measurement
    if verbose:
        print(f"\n[2/3] Measuring inference latency...")
        print(f"  Configuration:")
        print(f"    - Device: {device}")
        print(f"    - Input shape: [{batch_size}, {num_frames}, 3, 518, 518]")
        print(f"    - Warmup iterations: {warmup_iters}")
        print(f"    - Measurement iterations: {measurement_iters}")

    # Create input tensor
    input_tensor = torch.rand(batch_size, num_frames, 3, 518, 518)

    latency_stats = measure_latency(
        model,
        input_tensor,
        device=device,
        warmup=warmup_iters,
        iters=measurement_iters
    )
    results['latency'] = latency_stats

    if verbose:
        print(f"\n  Results:")
        print(f"    - Mean: {latency_stats['mean_ms']:.2f} ms")
        print(f"    - Median: {latency_stats['median_ms']:.2f} ms")
        print(f"    - Std: {latency_stats['std_ms']:.2f} ms")
        print(f"    - P95: {latency_stats['p95_ms']:.2f} ms")
        print(f"    - P99: {latency_stats['p99_ms']:.2f} ms")

    # 3. Throughput calculation
    throughput = calculate_throughput(latency_stats['mean_ms'])
    results['throughput_fps'] = throughput

    if verbose:
        print(f"    - Throughput: {throughput:.2f} FPS")

    # 4. Memory measurement
    if verbose:
        print(f"\n[3/3] Measuring memory usage...")

    memory_gb = measure_memory(model, input_tensor, device=device)
    results['memory_gb'] = memory_gb

    if verbose:
        if device == 'cuda':
            print(f"  Peak memory: {memory_gb:.3f} GB")
        else:
            print(f"  ⚠ Memory measurement skipped (CPU only)")

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ {model_name} Benchmark Complete")
        print(f"{'='*60}")

    return results


def compare_models(
    student_results: Dict,
    teacher_results: Dict,
    verbose: bool = True
) -> Dict:
    """
    Compare student and teacher benchmark results.

    Args:
        student_results: Student benchmark results
        teacher_results: Teacher benchmark results
        verbose: Print comparison

    Returns:
        Dictionary with comparison metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model Comparison")
        print(f"{'='*60}")

    comparison = {}

    # 1. Parameter comparison
    student_params = student_results['parameters']['total']
    teacher_params = teacher_results['parameters']['total']
    param_reduction = teacher_params / student_params

    comparison['parameter_reduction'] = param_reduction
    comparison['student_params'] = student_params
    comparison['teacher_params'] = teacher_params

    if verbose:
        print(f"\n[1/3] Parameters:")
        print(f"  Teacher: {format_number(teacher_params)}")
        print(f"  Student: {format_number(student_params)}")
        print(f"  Reduction: {param_reduction:.2f}x")

    # 2. Latency comparison
    student_latency = student_results['latency']['mean_ms']
    teacher_latency = teacher_results['latency']['mean_ms']
    latency_speedup = teacher_latency / student_latency

    comparison['latency_speedup'] = latency_speedup
    comparison['student_latency_ms'] = student_latency
    comparison['teacher_latency_ms'] = teacher_latency

    if verbose:
        print(f"\n[2/3] Latency:")
        print(f"  Teacher: {teacher_latency:.2f} ms")
        print(f"  Student: {student_latency:.2f} ms")
        print(f"  Speedup: {latency_speedup:.2f}x")

    # 3. Memory comparison
    student_memory = student_results['memory_gb']
    teacher_memory = teacher_results['memory_gb']

    if student_memory > 0 and teacher_memory > 0:
        memory_reduction = teacher_memory / student_memory
        comparison['memory_reduction'] = memory_reduction
        comparison['student_memory_gb'] = student_memory
        comparison['teacher_memory_gb'] = teacher_memory

        if verbose:
            print(f"\n[3/3] Memory:")
            print(f"  Teacher: {teacher_memory:.3f} GB")
            print(f"  Student: {student_memory:.3f} GB")
            print(f"  Reduction: {memory_reduction:.2f}x")
    else:
        comparison['memory_reduction'] = None
        if verbose:
            print(f"\n[3/3] Memory:")
            print(f"  ⚠ Memory comparison not available (CPU mode)")

    # 4. Check against targets
    comparison['meets_targets'] = check_targets(comparison, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")

    return comparison


def check_targets(comparison: Dict, verbose: bool = True) -> bool:
    """
    Check if student meets Phase 0A targets.

    Targets:
        - Parameters ≤ 400M
        - Latency speedup ≥ 1.5x
        - Memory reduction ≥ 2.0x

    Args:
        comparison: Comparison dictionary
        verbose: Print results

    Returns:
        True if all targets met, False otherwise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Target Evaluation")
        print(f"{'='*60}")

    targets = {
        'parameters': {
            'target': 400_000_000,
            'actual': comparison['student_params'],
            'condition': lambda x: x <= 400_000_000,
            'label': 'Parameters ≤ 400M'
        },
        'latency_speedup': {
            'target': 1.5,
            'actual': comparison['latency_speedup'],
            'condition': lambda x: x >= 1.5,
            'label': 'Latency speedup ≥ 1.5x'
        },
    }

    # Add memory target only if available
    if comparison.get('memory_reduction') is not None:
        targets['memory_reduction'] = {
            'target': 2.0,
            'actual': comparison['memory_reduction'],
            'condition': lambda x: x >= 2.0,
            'label': 'Memory reduction ≥ 2.0x'
        }

    all_passed = True

    for key, target_info in targets.items():
        passed = target_info['condition'](target_info['actual'])
        all_passed = all_passed and passed

        if verbose:
            status = "✓" if passed else "✗"
            if key == 'parameters':
                actual_str = format_number(target_info['actual'])
                target_str = format_number(target_info['target'])
            else:
                actual_str = f"{target_info['actual']:.2f}x"
                target_str = f"{target_info['target']:.1f}x"

            print(f"  {status} {target_info['label']}")
            print(f"      Target: {target_str}")
            print(f"      Actual: {actual_str}")

    if verbose:
        print(f"\n{'='*60}")
        if all_passed:
            print(f"✓ ALL TARGETS MET - GO FOR PHASE 1")
        else:
            print(f"✗ TARGETS NOT MET - REDESIGN REQUIRED")
        print(f"{'='*60}")

    return all_passed
