# Phase 0A: Benchmarking Plan

**Date:** 2026-08-24  
**Duration:** 1-2 hours (plus implementation time)  
**Status:** Ready to execute

---

## Objective

**Measure actual student encoder performance BEFORE committing to 3-7 days of training.**

If benchmarks fail → redesign architecture (saves days of wasted training).  
If benchmarks pass → proceed to Phase 1 with confidence.

---

## Success Criteria

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| **Parameters** | ≤ 400M | TBD | ⏳ |
| **Latency Speedup** | ≥ 1.5x | TBD | ⏳ |
| **Memory Reduction** | ≥ 2.0x | TBD | ⏳ |

**GO Decision:** All three criteria met ✓  
**NO-GO Decision:** Any criterion fails ✗

---

## What Phase 0A Does

```
1. Load teacher encoder (frozen, FP16)
2. Initialize student encoder with DINOv2 weights
3. Measure both models:
   - Parameter count
   - Inference latency (FP16)
   - Peak memory usage (FP16)
4. Generate comparison report
5. Make GO/NO-GO decision
```

**What it does NOT do:**
- ❌ Train the student
- ❌ Compute distillation loss
- ❌ Evaluate task metrics (IoU, F1, etc.)
- ❌ Run on real data

**Input:** Random tensor `[1, 8, 3, 518, 518]` (dummy data)  
**Output:** `docs/benchmark_report.md` with GO/NO-GO decision

---

## Benchmark Configuration

### Hardware
- Device: CUDA GPU (A100 recommended, other GPUs acceptable)
- VRAM: ≥16GB recommended
- CPU: Fallback if GPU unavailable (slower, but works)

### Precision
- Teacher: FP16
- Student: FP16
- Input: FP16

### Measurement Settings
```python
Latency:
  warmup_iters: 20
  measurement_iters: 100
  batch_size: 1
  num_frames: 8
  
Memory:
  batch_size: 1
  num_frames: 8
  
Throughput:
  Calculated from latency: 1000 / latency_ms
```

---

## Implementation Steps

### Step 1: Implement Student Encoder

**File:** `student/aggregator.py`

**Requirements:**
- StudentAggregator class
- 18 frame blocks, 18 global blocks
- Cached layers: [3, 8, 13, 17]
- embed_dim=768, num_heads=12
- Same token structure as teacher (1374 tokens)

**Self-contained:** Copy necessary code from teacher, no imports from parent dirs

---

### Step 2: Implement DINOv2 Initialization

**File:** `student/initialization.py`

**Requirements:**
- `load_dinov2_vitb14_reg()` - Load pretrained model from torch.hub
- `initialize_student_from_dinov2(student, dinov2)` - Transfer weights
- Transfer patch embedding
- Transfer blocks 0-11 (pretrained)
- Leave blocks 12-17 random
- Initialize special tokens (random, std=1e-6)

---

### Step 3: Implement Benchmarking Tools

**File:** `benchmarking/metrics.py`

```python
def count_parameters(model: nn.Module) -> dict:
    """
    Count parameters with breakdown.
    
    Returns:
        {
            'patch_embed': int,
            'frame_blocks': int,
            'global_blocks': int,
            'special_tokens': int,
            'total': int
        }
    """

def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda',
    warmup: int = 20,
    iters: int = 100
) -> dict:
    """
    Measure inference latency with CUDA synchronization.
    
    Returns:
        {
            'mean_ms': float,
            'std_ms': float,
            'median_ms': float,
            'p95_ms': float,
            'p99_ms': float
        }
    """

def measure_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Measure peak memory allocation.
    
    Returns:
        peak_memory_gb: float
    """

def calculate_throughput(latency_ms: float) -> float:
    """
    Calculate throughput in FPS.
    
    Returns:
        fps: float
    """
```

---

**File:** `benchmarking/benchmark.py`

```python
def benchmark_student(
    student_model: nn.Module,
    teacher_model: nn.Module,
    device: str = 'cuda'
) -> tuple[dict, dict]:
    """
    Run full benchmark suite on both models.
    
    Returns:
        student_metrics: dict
        teacher_metrics: dict
    """

def compare_with_teacher(
    student_metrics: dict,
    teacher_metrics: dict
) -> dict:
    """
    Generate comparison metrics.
    
    Returns:
        {
            'parameter_reduction': float,  # e.g., 2.6x
            'latency_speedup': float,      # e.g., 1.7x
            'memory_reduction': float,     # e.g., 2.1x
            'meets_targets': bool
        }
    """
```

---

**File:** `benchmarking/report.py`

```python
def generate_report(
    student_metrics: dict,
    teacher_metrics: dict,
    comparison: dict,
    output_path: str = 'docs/benchmark_report.md'
) -> None:
    """
    Generate markdown benchmark report.
    """

def make_go_nogo_decision(comparison: dict) -> str:
    """
    Evaluate against targets and make decision.
    
    Returns:
        'GO' or 'NO-GO' with reasoning
    """
```

---

### Step 4: Implement Entry Point

**File:** `benchmark_student.py`

```python
#!/usr/bin/env python3
"""
Phase 0A: Student Encoder Benchmarking

Measures student encoder performance before training.
Generates GO/NO-GO decision for Phase 1.
"""

import argparse
import torch
from student import StudentAggregator, initialize_student_from_dinov2
from benchmarking import benchmark_student, compare_with_teacher, generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--teacher_checkpoint', 
                       default='../../vggt-unified/checkpoints/vggt_unified_fp16.pt')
    args = parser.parse_args()
    
    # 1. Load teacher encoder (frozen)
    teacher = load_teacher_encoder(args.teacher_checkpoint, args.device)
    
    # 2. Initialize student encoder with DINOv2
    student = StudentAggregator()
    initialize_student_from_dinov2(student)
    student = student.half().to(args.device).eval()
    
    # 3. Run benchmarks
    student_metrics, teacher_metrics = benchmark_student(
        student, teacher, args.device
    )
    
    # 4. Generate comparison
    comparison = compare_with_teacher(student_metrics, teacher_metrics)
    
    # 5. Generate report
    generate_report(student_metrics, teacher_metrics, comparison)
    
    # 6. Output decision
    decision = make_go_nogo_decision(comparison)
    print(f"\n{'='*60}")
    print(f"DECISION: {decision}")
    print(f"{'='*60}\n")
    print(f"Report saved to: docs/benchmark_report.md")

if __name__ == '__main__':
    main()
```

---

### Step 5: Execute

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python benchmark_student.py --device cuda

# Review report
cat docs/benchmark_report.md
```

---

## Expected Output

### docs/benchmark_report.md

```markdown
# Phase 0A Benchmark Report

**Date:** 2026-08-24  
**Device:** CUDA (NVIDIA A100)  
**Precision:** FP16

---

## Results

### Parameter Count

| Model | Total | Frame Blocks | Global Blocks | Patch Embed | Special Tokens |
|-------|-------|--------------|---------------|-------------|----------------|
| Teacher | 885M | 342M | 342M | 0.6M | 0.006M |
| Student | 342M | 127M | 127M | 0.6M | 0.006M |

**Reduction:** 2.6x fewer parameters ✓

---

### Inference Latency (FP16)

| Model | Mean | Std | Median | P95 | P99 |
|-------|------|-----|--------|-----|-----|
| Teacher | 250ms | 5ms | 248ms | 258ms | 262ms |
| Student | 147ms | 3ms | 146ms | 152ms | 155ms |

**Speedup:** 1.7x faster ✓

---

### Peak Memory (FP16)

| Model | Peak Allocated |
|-------|----------------|
| Teacher | 10.2 GB |
| Student | 4.8 GB |

**Reduction:** 2.1x less memory ✓

---

## Decision: GO ✓

All targets met:
- ✓ Parameters: 342M ≤ 400M target
- ✓ Latency: 1.7x ≥ 1.5x target
- ✓ Memory: 2.1x ≥ 2.0x target

**Recommendation:** Proceed to Phase 1 (Training)
```

---

## Troubleshooting

### Issue: Teacher checkpoint not found

```
FileNotFoundError: ../../vggt-unified/checkpoints/vggt_unified_fp16.pt
```

**Solution:**
```bash
# Check if checkpoint exists
ls ../../vggt-unified/checkpoints/

# If missing, specify alternate path
python benchmark_student.py --teacher_checkpoint /path/to/checkpoint.pt
```

---

### Issue: DINOv2 download fails

```
RuntimeError: torch.hub download failed
```

**Solution:**
```bash
# Manually download DINOv2 first
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')"

# Then run benchmark
python benchmark_student.py
```

---

### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use CPU (slower but works)
python benchmark_student.py --device cpu

# Or use smaller batch size (not recommended, affects measurements)
```

---

### Issue: Different results on different GPUs

**Expected:** Latency varies by GPU (A100 faster than V100, etc.)  
**Not a problem:** As long as speedup ratio (student/teacher) meets target

---

## What Happens After Phase 0A

### If GO Decision

1. Update STATUS.md: Phase 0A complete ✓
2. Archive benchmark report
3. Proceed to Phase 1 implementation:
   - Implement distillation loss
   - Implement token sampling
   - Implement training pipeline
4. Run sanity check (3-5 epochs)
5. Run full training (40-50 epochs)

### If NO-GO Decision

1. Document failure reasons
2. Analyze bottlenecks:
   - Parameter count too high? → Reduce depth/width
   - Latency too slow? → Profile layers, optimize bottlenecks
   - Memory too high? → Reduce batch size, enable gradient checkpointing
3. Redesign architecture
4. Re-run Phase 0A
5. Iterate until GO decision

---

## Time Budget

| Task | Duration | Notes |
|------|----------|-------|
| **Implementation** | 4-6 hours | Writing all code |
| **DINOv2 Download** | 10-30 min | First time only (~350MB) |
| **Benchmark Execution** | 5-10 min | 100 iterations each |
| **Report Review** | 5 min | Read and decide |
| **Total** | 5-7 hours | One-time investment |

**ROI:** Saves 3-7 days if architecture is wrong!

---

## Files Checklist

### Required for Phase 0A

- [x] README.md
- [x] STATUS.md
- [x] requirements.txt
- [x] docs/architecture.md
- [x] docs/phase_0a_plan.md
- [ ] student/aggregator.py
- [ ] student/initialization.py
- [ ] student/__init__.py
- [ ] benchmarking/metrics.py
- [ ] benchmarking/benchmark.py
- [ ] benchmarking/report.py
- [ ] benchmarking/__init__.py
- [ ] benchmark_student.py

### Generated by Phase 0A

- [ ] docs/benchmark_report.md

---

## Next Steps

After documentation complete:

1. **Implement student/aggregator.py** (core architecture)
2. **Implement student/initialization.py** (DINOv2 loading)
3. **Implement benchmarking/** (measurement tools)
4. **Implement benchmark_student.py** (entry point)
5. **Execute and validate** (run benchmarks)

**Current status:** Documentation complete ✓, ready for implementation.
