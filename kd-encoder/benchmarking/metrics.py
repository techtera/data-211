# Benchmarking metrics for Phase 0A
# Measures parameters, latency, memory usage

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters with breakdown.

    Args:
        model: StudentAggregator instance

    Returns:
        Dictionary with parameter counts:
            - patch_embed: Patch embedding parameters
            - frame_blocks: Frame attention blocks
            - global_blocks: Global attention blocks
            - special_tokens: Camera and register tokens
            - total: Total parameters
    """
    breakdown = {}

    # Patch embedding
    if hasattr(model, 'patch_embed'):
        breakdown['patch_embed'] = sum(p.numel() for p in model.patch_embed.parameters())
    else:
        breakdown['patch_embed'] = 0

    # Frame blocks
    if hasattr(model, 'frame_blocks'):
        breakdown['frame_blocks'] = sum(p.numel() for p in model.frame_blocks.parameters())
    else:
        breakdown['frame_blocks'] = 0

    # Global blocks
    if hasattr(model, 'global_blocks'):
        breakdown['global_blocks'] = sum(p.numel() for p in model.global_blocks.parameters())
    else:
        breakdown['global_blocks'] = 0

    # Special tokens
    special_tokens = 0
    if hasattr(model, 'camera_token'):
        special_tokens += model.camera_token.numel()
    if hasattr(model, 'register_token'):
        special_tokens += model.register_token.numel()
    breakdown['special_tokens'] = special_tokens

    # Total
    breakdown['total'] = sum(p.numel() for p in model.parameters())

    return breakdown


def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda',
    warmup: int = 20,
    iters: int = 100
) -> Dict[str, float]:
    """
    Measure inference latency with proper GPU synchronization.

    Args:
        model: Model to benchmark
        input_tensor: Input tensor [B, S, 3, H, W]
        device: 'cuda' or 'cpu'
        warmup: Number of warmup iterations
        iters: Number of measurement iterations

    Returns:
        Dictionary with latency statistics (in milliseconds):
            - mean_ms: Mean latency
            - std_ms: Standard deviation
            - median_ms: Median latency
            - p95_ms: 95th percentile
            - p99_ms: 99th percentile
            - min_ms: Minimum latency
            - max_ms: Maximum latency
    """
    model = model.eval()

    # Move to device
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    print(f"  Running {warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    # Measure
    print(f"  Running {iters} measurement iterations...")
    latencies = []

    with torch.no_grad():
        for i in range(iters):
            if device == 'cuda':
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
            else:
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{iters} iterations completed")

    latencies = np.array(latencies)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
    }


def measure_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Measure peak memory allocation during inference.

    Args:
        model: Model to benchmark
        input_tensor: Input tensor [B, S, 3, H, W]
        device: 'cuda' or 'cpu'

    Returns:
        Peak memory allocated in GB

    Note:
        Only works on CUDA. Returns 0.0 for CPU.
    """
    if device != 'cuda':
        print("  Warning: Memory measurement only available on CUDA")
        return 0.0

    model = model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    torch.cuda.synchronize()

    # Get peak memory
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)

    return peak_memory_gb


def calculate_throughput(latency_ms: float) -> float:
    """
    Calculate throughput in FPS from latency.

    Args:
        latency_ms: Latency in milliseconds

    Returns:
        Throughput in frames per second
    """
    if latency_ms <= 0:
        return 0.0
    return 1000.0 / latency_ms


def format_number(num: float, unit: str = '') -> str:
    """Format large numbers with commas and units."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B{unit}"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M{unit}"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K{unit}"
    else:
        return f"{num:.2f}{unit}"
