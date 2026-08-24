# VGGT Encoder Knowledge Distillation

Student encoder training via feature-based knowledge distillation from VGGT-1B teacher.

---

## Overview

**Goal:** Distill VGGT encoder (885M params) → Student encoder (342M params)

**Approach:**
- Feature-based distillation (not task-based)
- Token-sampled feature matching (133 tokens per frame)
- DINOv2 ViT-Base pretrained initialization
- 70% MSE + 30% Cosine loss

**Key Targets:**
- Parameters: ~342M (2.6x reduction)
- Latency: ≥1.5x speedup
- Memory: ≥2x reduction

---

## Project Structure

```
kd-encoder/
├── student/                    # Student encoder architecture
│   ├── aggregator.py          # 18-layer, 768-dim encoder
│   └── initialization.py      # DINOv2 weight loading
├── benchmarking/              # Phase 0A benchmarking
│   ├── metrics.py             # Parameter/latency/memory measurement
│   ├── benchmark.py           # Main benchmarking
│   └── report.py              # Report generation
├── distillation/              # (Phase 1) Distillation components
│   ├── loss.py                # DistillationLoss class
│   ├── projection.py          # Projection heads (1536→2048)
│   └── token_sampling.py      # Token sampling utilities
├── training/                  # (Phase 1) Training pipeline
│   ├── config.py              # Training configuration
│   ├── dataset.py             # Dataset loader
│   ├── trainer.py             # Training loop
│   └── validate.py            # Validation loop
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── checkpoints/               # Model checkpoints
├── logs/                      # Training logs
├── benchmark_student.py       # Phase 0A entry point
├── train.py                   # (Phase 1) Training entry point
└── sanity_check.py            # (Phase 1) Sanity check
```

---

## Phase 0A: Benchmarking (Current Phase)

**Objective:** Validate student architecture before training

**What it does:**
1. Initialize student encoder with DINOv2 weights
2. Measure parameters, latency, memory
3. Compare with teacher encoder
4. Generate GO/NO-GO decision

**Usage:**
```bash
cd vggt-KD/kd-encoder

# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python benchmark_student.py --device cuda

# Output: docs/benchmark_report.md
```

**Success Criteria:**
- ✓ Student initializes without errors
- ✓ Parameters ≤ 400M
- ✓ Latency ≥ 1.5x faster than teacher (FP16)
- ✓ Memory ≥ 2x less than teacher (FP16)

**If benchmarks pass → Proceed to Phase 1 (Training)**
**If benchmarks fail → Redesign architecture**

---

## Phase 1: Training (Future)

**Objective:** Train student to match teacher features

**What it does:**
1. Load teacher encoder (frozen)
2. Load student encoder (trainable)
3. Train with distillation loss for 40-50 epochs
4. Save best checkpoint

**Usage:**
```bash
# Sanity check (3-5 epochs, small dataset)
python sanity_check.py --images_dir data/images/ --epochs 5

# Full training
python train.py --images_dir data/images/ --epochs 50
```

**Expected duration:** 3-7 days on GPU

---

## Architecture

### Teacher (VGGT-1B)
```
Parameters: 885M
Dimension:  1024
Depth:      24 layers
Heads:      16
Cached:     [4, 11, 17, 23]
Output:     [B, S, P, 2048] (frame+global concatenated)
```

### Student
```
Parameters: ~342M (estimated)
Dimension:  768
Depth:      18 layers
Heads:      12
Cached:     [3, 8, 13, 17]
Output:     [B, S, P, 1536] (frame+global concatenated)
```

### Initialization
```
Source: DINOv2 ViT-Base (768 dim, 12 layers)

Student blocks 0-11:  DINOv2 pretrained weights
Student blocks 12-17: Random initialization
Patch embedding:      DINOv2 pretrained
Special tokens:       Random initialization
```

### Layer Mapping (Teacher → Student)
```
Teacher Layer → Student Layer
    4         →      3
   11         →      8
   17         →     13
   23         →     17
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA-capable GPU (for benchmarking/training)
- Teacher checkpoint: `../../vggt-unified/checkpoints/vggt_unified_fp16.pt`

---

## Current Status

**Phase 0A:** In Progress
- [ ] Student encoder implementation
- [ ] DINOv2 initialization
- [ ] Benchmarking tools
- [ ] Benchmark execution

**Phase 1:** Not Started
- [ ] Distillation loss
- [ ] Token sampling
- [ ] Training pipeline

See [STATUS.md](STATUS.md) for detailed progress.

---

## Documentation

- [Architecture Details](docs/architecture.md)
- [Phase 0A Plan](docs/phase_0a_plan.md)
- [Benchmark Report](docs/benchmark_report.md) (generated after Phase 0A)
- [Full Specification](../../.claude/plans/deep-gathering-plum.md)

---

## Quick Start (Phase 0A)

```bash
# 1. Navigate to project
cd vggt-KD/kd-encoder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run benchmark
python benchmark_student.py --device cuda

# 4. Check report
cat docs/benchmark_report.md
```

---

## Notes

- **Self-contained:** All code copied to this directory (no imports from parent)
- **Teacher checkpoint:** Must exist at `../../vggt-unified/checkpoints/vggt_unified_fp16.pt`
- **DINOv2 download:** Automatically downloads from torch.hub on first run
- **Phase 0A duration:** ~1-2 hours (includes DINOv2 download)

---

## References

- VGGT Paper: [Visual Geometry Grounded Transformer]
- DINOv2: https://github.com/facebookresearch/dinov2
- Distillation Plan: `../../.claude/plans/deep-gathering-plum.md`
