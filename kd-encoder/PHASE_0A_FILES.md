# Phase 0A Implementation - File Manifest

## Directory Structure

```
vggt-KD/kd-encoder/
├── docs/                              # Documentation
├── student/                           # Student encoder architecture
├── benchmarking/                      # Benchmarking tools
├── tests/                             # Unit tests
├── checkpoints/                       # Model checkpoints
├── logs/                              # Logs
└── benchmark_student.py               # Main entry point
```

---

## Phase 0A Files (Benchmarking Only)

### Priority 1: Core Architecture (Required)

#### 1. `student/aggregator.py`
**Purpose:** Student encoder architecture (18 layers, 768 dim)
**Dependencies:** Teacher encoder structure from `../fastVGGT/encoder/`
**Key components:**
- StudentAggregator class
- 18 frame attention blocks
- 18 global attention blocks
- Cached layers: [3, 8, 13, 17]
- embed_dim: 768, num_heads: 12
- Same token structure as teacher (1374 tokens)

#### 2. `student/initialization.py`
**Purpose:** Load DINOv2 ViT-Base pretrained weights
**Dependencies:** torch.hub, student/aggregator.py
**Key functions:**
- `load_dinov2_vitb14_reg()` - Load pretrained model
- `initialize_student_from_dinov2()` - Transfer weights to student
- Initialize blocks 0-11 from pretrained, 12-17 random

#### 3. `student/__init__.py`
**Purpose:** Module exports
**Exports:**
```python
from .aggregator import StudentAggregator
from .initialization import initialize_student_from_dinov2
```

---

### Priority 2: Benchmarking Tools (Required)

#### 4. `benchmarking/metrics.py`
**Purpose:** Measurement utilities
**Key functions:**
- `count_parameters(model)` - Parameter count with breakdown
- `measure_latency(model, input, warmup, iters)` - FP16 latency measurement
- `measure_memory(model, input)` - Peak memory allocation
- `calculate_throughput(latency_ms)` - FPS calculation

#### 5. `benchmarking/benchmark.py`
**Purpose:** Main benchmarking script
**Key functions:**
- `benchmark_student(student_model, device)` - Run all benchmarks
- `compare_with_teacher(student_metrics, teacher_metrics)` - Generate comparison
- Returns: dict with all metrics

#### 6. `benchmarking/report.py`
**Purpose:** Generate benchmark report
**Key functions:**
- `generate_report(student_metrics, teacher_metrics, output_path)` - Create markdown report
- `make_go_nogo_decision(comparison)` - Evaluate against targets
- Saves to: `docs/benchmark_report.md`

#### 7. `benchmarking/__init__.py`
**Purpose:** Module exports
**Exports:**
```python
from .metrics import count_parameters, measure_latency, measure_memory
from .benchmark import benchmark_student, compare_with_teacher
from .report import generate_report, make_go_nogo_decision
```

---

### Priority 3: Entry Point (Required)

#### 8. `benchmark_student.py`
**Purpose:** Phase 0A main entry point
**Usage:** `python benchmark_student.py --device cuda`
**Flow:**
1. Load teacher checkpoint from `../fastVGGT/checkpoints/` or `../../vggt-unified/checkpoints/`
2. Initialize student encoder with DINOv2 weights
3. Run benchmarks on both models
4. Generate comparison report
5. Output GO/NO-GO decision

---

### Priority 4: Documentation (Required)

#### 9. `docs/phase_0a_plan.md`
**Purpose:** Phase 0A execution plan
**Sections:**
- Objectives
- Success criteria
- Implementation steps
- File dependencies

#### 10. `docs/architecture.md`
**Purpose:** Student encoder architecture documentation
**Sections:**
- Architecture specification
- Layer mapping (teacher → student)
- Parameter breakdown
- Initialization strategy

#### 11. `README.md`
**Purpose:** Project overview and usage
**Sections:**
- What is kd-encoder
- Phase 0A instructions
- Requirements
- Quick start

#### 12. `STATUS.md`
**Purpose:** Current implementation status
**Tracks:**
- Phase 0A: In progress
- Completed files
- Pending files
- Benchmark results

#### 13. `requirements.txt`
**Purpose:** Python dependencies
**Contents:**
```
torch>=2.0.0
numpy>=1.24.0
```

---

## Files NOT Needed for Phase 0A

These will be implemented in later phases:

### Distillation (Phase 1)
- `distillation/loss.py`
- `distillation/projection.py`
- `distillation/token_sampling.py`

### Training (Phase 1)
- `training/config.py`
- `training/dataset.py`
- `training/dataloader.py`
- `training/trainer.py`
- `training/validate.py`
- `training/optimizer.py`
- `training/scheduler.py`
- `training/checkpoints.py`

### Scripts (Phase 1)
- `train.py`
- `sanity_check.py`

### Tests (Phase 1)
- All test files

---

## Implementation Order

**Step 1:** Documentation
- [ ] `README.md`
- [ ] `STATUS.md`
- [ ] `requirements.txt`
- [ ] `docs/architecture.md`
- [ ] `docs/phase_0a_plan.md`

**Step 2:** Student Architecture
- [ ] `student/aggregator.py`
- [ ] `student/initialization.py`
- [ ] `student/__init__.py`

**Step 3:** Benchmarking
- [ ] `benchmarking/metrics.py`
- [ ] `benchmarking/benchmark.py`
- [ ] `benchmarking/report.py`
- [ ] `benchmarking/__init__.py`

**Step 4:** Entry Point
- [ ] `benchmark_student.py`

**Step 5:** Execution
- [ ] Run benchmarks
- [ ] Generate report
- [ ] Make GO/NO-GO decision

---

## Expected Outputs

After Phase 0A completion:

1. **Benchmark Report:** `docs/benchmark_report.md`
   - Student parameter count (target: ~342M)
   - Latency comparison (target: ≥1.5x speedup)
   - Memory comparison (target: ≥2x reduction)
   - GO/NO-GO decision

2. **Updated STATUS.md**
   - Phase 0A: Complete ✓
   - Benchmark results
   - Next steps

---

## Teacher Checkpoint Location

**Primary:** `../../vggt-unified/checkpoints/vggt_unified_fp16.pt`
**Backup:** `../fastVGGT/checkpoints/vggt_unified_fp16.pt`

---

## Success Criteria

Phase 0A is successful if:
- ✓ Student encoder initializes without errors
- ✓ Parameters ≤ 400M
- ✓ Latency ≥ 1.5x faster than teacher (FP16)
- ✓ Memory ≥ 2x less than teacher (FP16)
- ✓ Benchmark report generated

If any criterion fails → redesign architecture before Phase 1.
