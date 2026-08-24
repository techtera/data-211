# Generate benchmark report in markdown format

from typing import Dict
from datetime import datetime
from .metrics import format_number


def generate_report(
    student_results: Dict,
    teacher_results: Dict,
    comparison: Dict,
    output_path: str = 'docs/benchmark_report.md'
) -> None:
    """
    Generate comprehensive benchmark report in markdown format.

    Args:
        student_results: Student benchmark results
        teacher_results: Teacher benchmark results
        comparison: Comparison metrics
        output_path: Path to save report
    """
    report_lines = []

    # Header
    report_lines.append("# Phase 0A Benchmark Report")
    report_lines.append("")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Device:** {student_results['device']}")
    report_lines.append(f"**Precision:** FP16")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")

    decision = "**GO**" if comparison['meets_targets'] else "**NO-GO**"
    decision_icon = "✓" if comparison['meets_targets'] else "✗"

    report_lines.append(f"**Decision:** {decision_icon} {decision}")
    report_lines.append("")

    if comparison['meets_targets']:
        report_lines.append("All Phase 0A targets met. Student encoder is ready for Phase 1 training.")
    else:
        report_lines.append("Phase 0A targets not met. Architecture redesign required before Phase 1.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Parameter Count
    report_lines.append("## 1. Parameter Count")
    report_lines.append("")
    report_lines.append("| Model | Total | Frame Blocks | Global Blocks | Patch Embed | Special Tokens |")
    report_lines.append("|-------|-------|--------------|---------------|-------------|----------------|")

    for model_name, results in [("Teacher", teacher_results), ("Student", student_results)]:
        params = results['parameters']
        report_lines.append(
            f"| {model_name} | "
            f"{format_number(params['total'])} | "
            f"{format_number(params['frame_blocks'])} | "
            f"{format_number(params['global_blocks'])} | "
            f"{format_number(params['patch_embed'])} | "
            f"{format_number(params['special_tokens'])} |"
        )

    report_lines.append("")
    param_target = "✓ PASS" if comparison['student_params'] <= 400_000_000 else "✗ FAIL"
    report_lines.append(f"**Reduction:** {comparison['parameter_reduction']:.2f}x fewer parameters")
    report_lines.append(f"**Target:** ≤ 400M parameters - {param_target}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Inference Latency
    report_lines.append("## 2. Inference Latency (FP16)")
    report_lines.append("")
    report_lines.append(f"**Configuration:**")
    report_lines.append(f"- Batch size: {student_results['batch_size']}")
    report_lines.append(f"- Frames per sample: {student_results['num_frames']}")
    report_lines.append(f"- Input shape: [{student_results['batch_size']}, {student_results['num_frames']}, 3, 518, 518]")
    report_lines.append("")

    report_lines.append("| Model | Mean | Median | Std | P95 | P99 | Throughput |")
    report_lines.append("|-------|------|--------|-----|-----|-----|------------|")

    for model_name, results in [("Teacher", teacher_results), ("Student", student_results)]:
        lat = results['latency']
        throughput = results['throughput_fps']
        report_lines.append(
            f"| {model_name} | "
            f"{lat['mean_ms']:.2f} ms | "
            f"{lat['median_ms']:.2f} ms | "
            f"{lat['std_ms']:.2f} ms | "
            f"{lat['p95_ms']:.2f} ms | "
            f"{lat['p99_ms']:.2f} ms | "
            f"{throughput:.2f} FPS |"
        )

    report_lines.append("")
    latency_target = "✓ PASS" if comparison['latency_speedup'] >= 1.5 else "✗ FAIL"
    report_lines.append(f"**Speedup:** {comparison['latency_speedup']:.2f}x faster")
    report_lines.append(f"**Target:** ≥ 1.5x speedup - {latency_target}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Memory Usage
    report_lines.append("## 3. Peak Memory Usage (FP16)")
    report_lines.append("")

    if comparison.get('memory_reduction') is not None:
        report_lines.append("| Model | Peak Allocated |")
        report_lines.append("|-------|----------------|")
        report_lines.append(f"| Teacher | {teacher_results['memory_gb']:.3f} GB |")
        report_lines.append(f"| Student | {student_results['memory_gb']:.3f} GB |")
        report_lines.append("")

        memory_target = "✓ PASS" if comparison['memory_reduction'] >= 2.0 else "✗ FAIL"
        report_lines.append(f"**Reduction:** {comparison['memory_reduction']:.2f}x less memory")
        report_lines.append(f"**Target:** ≥ 2.0x reduction - {memory_target}")
    else:
        report_lines.append("Memory measurement not available (CPU mode)")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Target Summary
    report_lines.append("## 4. Target Summary")
    report_lines.append("")
    report_lines.append("| Target | Threshold | Actual | Status |")
    report_lines.append("|--------|-----------|--------|--------|")

    # Parameters
    param_status = "✓ PASS" if comparison['student_params'] <= 400_000_000 else "✗ FAIL"
    report_lines.append(
        f"| Parameters | ≤ 400M | {format_number(comparison['student_params'])} | {param_status} |"
    )

    # Latency
    latency_status = "✓ PASS" if comparison['latency_speedup'] >= 1.5 else "✗ FAIL"
    report_lines.append(
        f"| Latency Speedup | ≥ 1.5x | {comparison['latency_speedup']:.2f}x | {latency_status} |"
    )

    # Memory
    if comparison.get('memory_reduction') is not None:
        memory_status = "✓ PASS" if comparison['memory_reduction'] >= 2.0 else "✗ FAIL"
        report_lines.append(
            f"| Memory Reduction | ≥ 2.0x | {comparison['memory_reduction']:.2f}x | {memory_status} |"
        )

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Decision
    report_lines.append("## 5. Decision")
    report_lines.append("")

    if comparison['meets_targets']:
        report_lines.append("### ✓ GO - Proceed to Phase 1")
        report_lines.append("")
        report_lines.append("**Reasons:**")
        report_lines.append(f"- Parameters: {format_number(comparison['student_params'])} ≤ 400M ✓")
        report_lines.append(f"- Latency: {comparison['latency_speedup']:.2f}x ≥ 1.5x ✓")
        if comparison.get('memory_reduction') is not None:
            report_lines.append(f"- Memory: {comparison['memory_reduction']:.2f}x ≥ 2.0x ✓")
        report_lines.append("")
        report_lines.append("**Recommendation:** Begin Phase 1 distillation training.")
        report_lines.append("")
        report_lines.append("**Next Steps:**")
        report_lines.append("1. Implement distillation loss (MSE + Cosine)")
        report_lines.append("2. Implement token sampling utilities")
        report_lines.append("3. Implement training pipeline")
        report_lines.append("4. Run sanity check (3-5 epochs)")
        report_lines.append("5. Run full training (40-50 epochs)")
    else:
        report_lines.append("### ✗ NO-GO - Architecture Redesign Required")
        report_lines.append("")
        report_lines.append("**Failed Targets:**")

        if comparison['student_params'] > 400_000_000:
            report_lines.append(f"- ✗ Parameters: {format_number(comparison['student_params'])} > 400M")
            report_lines.append(f"  - Current: {format_number(comparison['student_params'])}")
            report_lines.append(f"  - Target: ≤ 400M")
            report_lines.append(f"  - Suggestion: Reduce depth to 16 layers or dimension to 704")

        if comparison['latency_speedup'] < 1.5:
            report_lines.append(f"- ✗ Latency: {comparison['latency_speedup']:.2f}x < 1.5x")
            report_lines.append(f"  - Current speedup: {comparison['latency_speedup']:.2f}x")
            report_lines.append(f"  - Target: ≥ 1.5x")
            report_lines.append(f"  - Suggestion: Profile bottleneck layers and optimize")

        if comparison.get('memory_reduction') is not None and comparison['memory_reduction'] < 2.0:
            report_lines.append(f"- ✗ Memory: {comparison['memory_reduction']:.2f}x < 2.0x")
            report_lines.append(f"  - Current reduction: {comparison['memory_reduction']:.2f}x")
            report_lines.append(f"  - Target: ≥ 2.0x")
            report_lines.append(f"  - Suggestion: Enable gradient checkpointing or reduce batch size")

        report_lines.append("")
        report_lines.append("**Recommendation:** Redesign architecture and re-run Phase 0A.")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Appendix
    report_lines.append("## Appendix: Raw Results")
    report_lines.append("")
    report_lines.append("### Student Encoder")
    report_lines.append("```")
    report_lines.append(f"Architecture:")
    report_lines.append(f"  - Depth: 18 layers")
    report_lines.append(f"  - Dimension: 768")
    report_lines.append(f"  - Heads: 12")
    report_lines.append(f"  - Cached layers: [3, 8, 13, 17]")
    report_lines.append(f"  - Parameters: {format_number(comparison['student_params'])}")
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### Teacher Encoder")
    report_lines.append("```")
    report_lines.append(f"Architecture:")
    report_lines.append(f"  - Depth: 24 layers")
    report_lines.append(f"  - Dimension: 1024")
    report_lines.append(f"  - Heads: 16")
    report_lines.append(f"  - Cached layers: [4, 11, 17, 23]")
    report_lines.append(f"  - Parameters: {format_number(comparison['teacher_params'])}")
    report_lines.append("```")

    # Write report
    report_content = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"\n✓ Report saved to: {output_path}")
