# Phase 0A Benchmark Report

**Date:** 2026-08-24 13:33:50
**Device:** cpu
**Precision:** FP16

---

## Executive Summary

**Decision:** ✗ **NO-GO**

Phase 0A targets not met. Architecture redesign required before Phase 1.

---

## 1. Parameter Count

| Model | Total | Frame Blocks | Global Blocks | Patch Embed | Special Tokens |
|-------|-------|--------------|---------------|-------------|----------------|
| Teacher | 605.23M | 302.31M | 302.31M | 603.14K | 10.24K |
| Student | 255.69M | 127.61M | 127.61M | 452.35K | 7.68K |

**Reduction:** 2.37x fewer parameters
**Target:** ≤ 400M parameters - ✓ PASS

---

## 2. Inference Latency (FP16)

**Configuration:**
- Batch size: 1
- Frames per sample: 8
- Input shape: [1, 8, 3, 518, 518]

| Model | Mean | Median | Std | P95 | P99 | Throughput |
|-------|------|--------|-----|-----|-----|------------|
| Teacher | 4.13 ms | 4.15 ms | 0.49 ms | 4.81 ms | 4.92 ms | 242.31 FPS |
| Student | 29899.40 ms | 29861.46 ms | 572.42 ms | 30739.16 ms | 30742.99 ms | 0.03 FPS |

**Speedup:** 0.00x faster
**Target:** ≥ 1.5x speedup - ✗ FAIL

---

## 3. Peak Memory Usage (FP16)

Memory measurement not available (CPU mode)

---

## 4. Target Summary

| Target | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Parameters | ≤ 400M | 255.69M | ✓ PASS |
| Latency Speedup | ≥ 1.5x | 0.00x | ✗ FAIL |

---

## 5. Decision

### ✗ NO-GO - Architecture Redesign Required

**Failed Targets:**
- ✗ Latency: 0.00x < 1.5x
  - Current speedup: 0.00x
  - Target: ≥ 1.5x
  - Suggestion: Profile bottleneck layers and optimize

**Recommendation:** Redesign architecture and re-run Phase 0A.

---

## Appendix: Raw Results

### Student Encoder
```
Architecture:
  - Depth: 18 layers
  - Dimension: 768
  - Heads: 12
  - Cached layers: [3, 8, 13, 17]
  - Parameters: 255.69M
```

### Teacher Encoder
```
Architecture:
  - Depth: 24 layers
  - Dimension: 1024
  - Heads: 16
  - Cached layers: [4, 11, 17, 23]
  - Parameters: 605.23M
```