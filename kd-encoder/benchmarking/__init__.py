# Benchmarking module for Phase 0A

from .metrics import (
    count_parameters,
    measure_latency,
    measure_memory,
    calculate_throughput,
    format_number
)
from .benchmark import (
    benchmark_model,
    compare_models,
    check_targets
)
from .report import generate_report

__all__ = [
    # Metrics
    'count_parameters',
    'measure_latency',
    'measure_memory',
    'calculate_throughput',
    'format_number',
    # Benchmark orchestration
    'benchmark_model',
    'compare_models',
    'check_targets',
    # Report generation
    'generate_report',
]
