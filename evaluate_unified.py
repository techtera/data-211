"""
Unified Model Production-Readiness Evaluation.

Evaluates: regression, latency, GPU memory, stability, deployment readiness.
Accuracy metrics skipped (no ground truth masks available).

Usage:
    python evaluate_unified.py --image_dir rgb_reg/ --num_images 50
"""

import sys
import argparse
import time
import csv
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model import VGGTUnified

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 518


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor


def get_image_paths(image_dir, num_images=50):
    image_dir = Path(image_dir)
    paths = sorted([
        p for p in image_dir.glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    return paths[:num_images]


# ============================================================
# SECTION 1: ACCURACY (SKIPPED - no ground truth)
# ============================================================

def evaluate_accuracy():
    return {
        "status": "SKIPPED",
        "reason": "No ground truth masks available in rgb_reg/",
        "obj_iou": None,
        "obj_dice": None,
        "edge_bf1": None,
    }


# ============================================================
# SECTION 2: REGRESSION CHECK
# ============================================================

def evaluate_regression(image_paths, unified_checkpoint, obj_checkpoint, edge_checkpoint):
    """Compare unified model outputs vs standalone-loaded model outputs."""
    print("\n" + "=" * 60)
    print("SECTION 2: REGRESSION CHECK")
    print("=" * 60)
    print("Comparing: unified checkpoint vs separate decoder checkpoints")

    # Load unified model
    print("\nLoading unified model...")
    model_unified = VGGTUnified(load_encoder=False)
    model_unified.load_unified_checkpoint(unified_checkpoint)
    model_unified = model_unified.to(DEVICE).eval()

    # Load model with separate checkpoints (same encoder, separate decoder weights)
    print("Loading standalone model (separate checkpoints)...")
    model_standalone = VGGTUnified(load_encoder=False)
    model_standalone.load_unified_checkpoint(unified_checkpoint)
    # Reload decoders from their original checkpoints
    model_standalone.load_decoder_checkpoint("obj", obj_checkpoint, device=DEVICE)
    model_standalone.load_decoder_checkpoint("edge", edge_checkpoint, device=DEVICE)
    model_standalone = model_standalone.to(DEVICE).eval()

    obj_disagreements = []
    edge_disagreements = []
    obj_iou_diffs = []
    edge_mae_diffs = []

    num_images = len(image_paths)
    print(f"Comparing on {num_images} images...")

    for i, img_path in enumerate(image_paths):
        img_tensor = load_image(str(img_path)).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_unified = model_unified(img_tensor, task="both")
            out_standalone = model_standalone(img_tensor, task="both")

        # Object mask comparison
        obj_u = out_unified["obj_mask"].argmax(dim=2).float()
        obj_s = out_standalone["obj_mask"].argmax(dim=2).float()
        pixel_disagree_obj = (obj_u != obj_s).float().mean().item() * 100
        obj_disagreements.append(pixel_disagree_obj)

        # IoU between unified and standalone obj predictions
        intersection = ((obj_u == 1) & (obj_s == 1)).sum().item()
        union = ((obj_u == 1) | (obj_s == 1)).sum().item()
        iou = intersection / max(union, 1)
        obj_iou_diffs.append(1.0 - iou)

        # Edge mask comparison (continuous probability)
        edge_u = out_unified["edge_mask"]
        edge_s = out_standalone["edge_mask"]
        mae = (edge_u - edge_s).abs().mean().item()
        edge_mae_diffs.append(mae)

        # Binary edge disagreement
        edge_bin_u = (edge_u > 0.5).float()
        edge_bin_s = (edge_s > 0.5).float()
        pixel_disagree_edge = (edge_bin_u != edge_bin_s).float().mean().item() * 100
        edge_disagreements.append(pixel_disagree_edge)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_images}]")

    # Cleanup
    del model_unified, model_standalone
    torch.cuda.empty_cache()
    gc.collect()

    # Stats
    obj_disagree = np.array(obj_disagreements)
    edge_disagree = np.array(edge_disagreements)

    results = {
        "obj_pixel_disagreement_mean": float(obj_disagree.mean()),
        "obj_pixel_disagreement_max": float(obj_disagree.max()),
        "obj_iou_diff_mean": float(np.mean(obj_iou_diffs)),
        "edge_mae_mean": float(np.mean(edge_mae_diffs)),
        "edge_pixel_disagreement_mean": float(edge_disagree.mean()),
        "edge_pixel_disagreement_max": float(edge_disagree.max()),
        "images_gt1pct_obj_degrade": int((obj_disagree > 1.0).sum()),
        "images_gt3pct_obj_degrade": int((obj_disagree > 3.0).sum()),
        "images_gt5pct_obj_degrade": int((obj_disagree > 5.0).sum()),
        "images_gt1pct_edge_degrade": int((edge_disagree > 1.0).sum()),
        "images_gt3pct_edge_degrade": int((edge_disagree > 3.0).sum()),
        "images_gt5pct_edge_degrade": int((edge_disagree > 5.0).sum()),
    }

    print(f"\n  Object mask pixel disagreement: {results['obj_pixel_disagreement_mean']:.4f}%")
    print(f"  Object mask IoU diff (1-IoU): {results['obj_iou_diff_mean']:.6f}")
    print(f"  Edge mask MAE: {results['edge_mae_mean']:.6f}")
    print(f"  Edge mask pixel disagreement: {results['edge_pixel_disagreement_mean']:.4f}%")
    print(f"  Images >1% obj degradation: {results['images_gt1pct_obj_degrade']}/{num_images}")
    print(f"  Images >1% edge degradation: {results['images_gt1pct_edge_degrade']}/{num_images}")

    return results


# ============================================================
# SECTION 3: LATENCY BENCHMARK
# ============================================================

def evaluate_latency(image_paths, unified_checkpoint, warmup=5, runs=20):
    """Measure per-stage latency for FP32 and FP16."""
    print("\n" + "=" * 60)
    print("SECTION 3: LATENCY BENCHMARK")
    print("=" * 60)

    results = {}

    for precision, ckpt in [("fp32", unified_checkpoint.replace("_fp16", "")),
                            ("fp16", unified_checkpoint)]:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"  Skipping {precision}: {ckpt} not found")
            continue

        print(f"\n--- {precision.upper()} ---")
        print(f"  Loading {ckpt}...")

        model = VGGTUnified(load_encoder=False)
        model.load_unified_checkpoint(ckpt)
        model = model.to(DEVICE).eval()

        # Use first image for benchmarking
        img_tensor = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Warmup
        print(f"  Warmup ({warmup} runs)...")
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(img_tensor, task="both")

        # Timed runs
        print(f"  Benchmarking ({runs} runs)...")
        latencies = defaultdict(list)
        for _ in range(runs):
            with torch.no_grad():
                out = model(img_tensor, task="both")
            for stage, val in out["latency"].items():
                latencies[stage].append(val)

        # Stats
        prefix = precision
        for stage, values in latencies.items():
            arr = np.array(values) * 1000  # to ms
            results[f"{prefix}_{stage}_mean"] = float(arr.mean())
            results[f"{prefix}_{stage}_p50"] = float(np.percentile(arr, 50))
            results[f"{prefix}_{stage}_p95"] = float(np.percentile(arr, 95))
            results[f"{prefix}_{stage}_min"] = float(arr.min())

        total_mean = results[f"{prefix}_total_mean"]
        print(f"  Total: {total_mean:.1f}ms (mean)")
        print(f"    Encoder: {results.get(f'{prefix}_encoder_mean', 0):.1f}ms")
        print(f"    Obj decoder: {results.get(f'{prefix}_obj_decoder_mean', 0):.1f}ms")
        print(f"    Edge decoder: {results.get(f'{prefix}_edge_decoder_mean', 0):.1f}ms")
        print(f"    FPS: {1000/total_mean:.1f}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Speedup calculation (standalone = 2 encoder passes + 2 decoders)
    if "fp16_encoder_mean" in results:
        enc = results["fp16_encoder_mean"]
        obj_dec = results.get("fp16_obj_decoder_mean", 0)
        edge_dec = results.get("fp16_edge_decoder_mean", 0)
        standalone_est = (enc + obj_dec) + (enc + edge_dec)
        unified_total = results["fp16_total_mean"]
        speedup = standalone_est / unified_total
        results["speedup_vs_standalone"] = float(speedup)
        results["encoder_reuse_savings_ms"] = float(enc)
        print(f"\n  Speedup vs standalone (est): {speedup:.2f}x")
        print(f"  Encoder reuse savings: {enc:.1f}ms")

    return results


# ============================================================
# SECTION 4: GPU MEMORY
# ============================================================

def evaluate_memory(image_paths, unified_checkpoint):
    """Measure peak GPU memory for FP32 and FP16."""
    print("\n" + "=" * 60)
    print("SECTION 4: GPU MEMORY")
    print("=" * 60)

    results = {}

    for precision, ckpt in [("fp32", unified_checkpoint.replace("_fp16", "")),
                            ("fp16", unified_checkpoint)]:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"  Skipping {precision}: {ckpt} not found")
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        print(f"\n--- {precision.upper()} ---")
        model = VGGTUnified(load_encoder=False)
        model.load_unified_checkpoint(ckpt)
        model = model.to(DEVICE).eval()

        img_tensor = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _ = model(img_tensor, task="both")

        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        results[f"{precision}_peak_allocated_gb"] = float(peak_alloc)
        results[f"{precision}_peak_reserved_gb"] = float(peak_reserved)

        print(f"  Peak allocated: {peak_alloc:.2f} GB")
        print(f"  Peak reserved:  {peak_reserved:.2f} GB")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ============================================================
# SECTION 5: STABILITY TESTING
# ============================================================

def evaluate_stability(image_paths, unified_checkpoint):
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("SECTION 5: STABILITY TESTING")
    print("=" * 60)

    results = {"tests_passed": 0, "tests_failed": 0, "failures": []}

    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(unified_checkpoint)
    model = model.to(DEVICE).eval()

    # Test 1: Single image inference
    print("  [1] Single image inference...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img, task="both")
        assert "obj_mask" in out and "edge_mask" in out
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"single_image: {e}")

    # Test 2: Multiple images sequentially
    print("  [2] Sequential multi-image inference...", end=" ")
    try:
        for p in image_paths[:5]:
            img = load_image(str(p)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img, task="both")
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"multi_image: {e}")

    # Test 3: All task modes
    print("  [3] All task modes (obj/edge/both/cascade)...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        for task in ["obj", "edge", "both", "cascade"]:
            with torch.no_grad():
                out = model(img, task=task)
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"task_modes: {e}")

    # Test 4: Output shape validation
    print("  [4] Output shape validation...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img, task="both")
        assert out["obj_mask"].shape == (1, 1, 2, IMAGE_SIZE, IMAGE_SIZE), \
            f"obj_mask shape: {out['obj_mask'].shape}"
        assert out["edge_mask"].shape == (1, 1, 1, IMAGE_SIZE, IMAGE_SIZE), \
            f"edge_mask shape: {out['edge_mask'].shape}"
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"output_shape: {e}")

    # Test 5: Output range validation
    print("  [5] Output range validation...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img, task="both")
        edge_probs = out["edge_mask"]
        assert edge_probs.min() >= 0 and edge_probs.max() <= 1, \
            f"edge_mask range: [{edge_probs.min():.4f}, {edge_probs.max():.4f}]"
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"output_range: {e}")

    # Test 6: Deterministic output
    print("  [6] Deterministic output (same input → same output)...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out1 = model(img, task="both")
            out2 = model(img, task="both")
        assert torch.allclose(out1["obj_mask"], out2["obj_mask"]), "obj_mask not deterministic"
        assert torch.allclose(out1["edge_mask"], out2["edge_mask"]), "edge_mask not deterministic"
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"deterministic: {e}")

    # Test 7: Corrupted image handling
    print("  [7] Corrupted/invalid input handling...", end=" ")
    try:
        # Zero image
        zero_img = torch.zeros(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            out = model(zero_img, task="both")
        # White image
        white_img = torch.ones(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            out = model(white_img, task="both")
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"corrupted_input: {e}")

    # Test 8: FP16 inference consistency
    print("  [8] FP16 model produces valid outputs...", end=" ")
    try:
        img = load_image(str(image_paths[0])).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img, task="both")
        assert not torch.isnan(out["obj_mask"]).any(), "NaN in obj_mask"
        assert not torch.isnan(out["edge_mask"]).any(), "NaN in edge_mask"
        assert not torch.isinf(out["obj_mask"]).any(), "Inf in obj_mask"
        assert not torch.isinf(out["edge_mask"]).any(), "Inf in edge_mask"
        print("PASS")
        results["tests_passed"] += 1
    except Exception as e:
        print(f"FAIL: {e}")
        results["tests_failed"] += 1
        results["failures"].append(f"fp16_validity: {e}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Results: {results['tests_passed']} passed, {results['tests_failed']} failed")
    return results


# ============================================================
# SECTION 6 & 7: DEPLOYMENT READINESS & RECOMMENDATION
# ============================================================

def generate_report(accuracy, regression, latency, memory, stability, output_dir):
    """Generate final evaluation report."""
    print("\n" + "=" * 60)
    print("SECTION 6: DEPLOYMENT READINESS")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine pass/fail
    checks = {}

    # Accuracy
    checks["Accuracy"] = ("SKIPPED", "No ground truth available")

    # Regression
    obj_disagree = regression.get("obj_pixel_disagreement_mean", 0)
    edge_disagree = regression.get("edge_pixel_disagreement_mean", 0)
    if obj_disagree < 0.1 and edge_disagree < 0.1:
        checks["Regression"] = ("PASS", f"Obj: {obj_disagree:.4f}%, Edge: {edge_disagree:.4f}%")
    elif obj_disagree < 1.0 and edge_disagree < 1.0:
        checks["Regression"] = ("PASS", f"Obj: {obj_disagree:.4f}%, Edge: {edge_disagree:.4f}% (minor diff)")
    else:
        checks["Regression"] = ("FAIL", f"Obj: {obj_disagree:.2f}%, Edge: {edge_disagree:.2f}%")

    # Latency
    fp16_total = latency.get("fp16_total_mean", 999)
    if fp16_total < 300:
        checks["Latency"] = ("PASS", f"FP16: {fp16_total:.1f}ms on A100")
    else:
        checks["Latency"] = ("WARN", f"FP16: {fp16_total:.1f}ms — may exceed 1s on Orin NX")

    # Memory
    fp16_mem = memory.get("fp16_peak_allocated_gb", 999)
    if fp16_mem < 16:
        checks["Memory"] = ("PASS", f"FP16 peak: {fp16_mem:.2f} GB (fits Orin NX 16GB)")
    else:
        checks["Memory"] = ("FAIL", f"FP16 peak: {fp16_mem:.2f} GB (exceeds Orin NX 16GB)")

    # Stability
    if stability["tests_failed"] == 0:
        checks["Stability"] = ("PASS", f"{stability['tests_passed']}/{stability['tests_passed']} tests passed")
    else:
        checks["Stability"] = ("FAIL", f"{stability['tests_failed']} tests failed")

    # Print table
    print(f"\n{'Metric':<15} {'Result':<8} {'Details'}")
    print("-" * 65)
    for metric, (result, details) in checks.items():
        print(f"{metric:<15} {result:<8} {details}")

    # Recommendation
    print("\n" + "=" * 60)
    print("SECTION 7: FINAL RECOMMENDATION")
    print("=" * 60)

    fails = sum(1 for _, (r, _) in checks.items() if r == "FAIL")
    if fails == 0:
        recommendation = "GO"
        next_step = "FP16 ONNX export → ONNX Runtime benchmark on target hardware"
    else:
        recommendation = "NO-GO"
        next_step = "Fix failing checks before proceeding"

    print(f"\n  Recommendation: {'✓' if recommendation == 'GO' else '✗'} {recommendation}")
    print(f"  Next step: {next_step}")

    # Write evaluation markdown
    md_path = output_dir / "unified_model_evaluation.md"
    with open(md_path, "w") as f:
        f.write("# Unified Model Production-Readiness Evaluation\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n\n")

        f.write("## 1. Accuracy\n\n")
        f.write(f"**Status:** {accuracy['status']}\n")
        f.write(f"**Reason:** {accuracy['reason']}\n\n")

        f.write("## 2. Regression Check\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        for k, v in regression.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")

        f.write("## 3. Latency Benchmark\n\n")
        f.write("| Metric | Value (ms) |\n|--------|------------|\n")
        for k, v in sorted(latency.items()):
            if isinstance(v, float):
                f.write(f"| {k} | {v:.2f} |\n")
        f.write("\n")

        f.write("## 4. GPU Memory\n\n")
        f.write("| Metric | Value (GB) |\n|--------|------------|\n")
        for k, v in memory.items():
            f.write(f"| {k} | {v:.2f} |\n")
        f.write("\n")

        f.write("## 5. Stability\n\n")
        f.write(f"- Tests passed: {stability['tests_passed']}\n")
        f.write(f"- Tests failed: {stability['tests_failed']}\n")
        if stability["failures"]:
            f.write("- Failures:\n")
            for fail in stability["failures"]:
                f.write(f"  - {fail}\n")
        f.write("\n")

        f.write("## 6. Deployment Readiness\n\n")
        f.write("| Metric | Result | Details |\n|--------|--------|--------|\n")
        for metric, (result, details) in checks.items():
            f.write(f"| {metric} | {result} | {details} |\n")
        f.write("\n")

        f.write("## 7. Recommendation\n\n")
        f.write(f"**{recommendation}**\n\n")
        f.write(f"Next step: {next_step}\n")

    print(f"\n  Report saved: {md_path}")

    # Write metrics CSV
    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", "value"])
        for k, v in regression.items():
            writer.writerow(["regression", k, v])
        for k, v in memory.items():
            writer.writerow(["memory", k, v])
        writer.writerow(["stability", "tests_passed", stability["tests_passed"]])
        writer.writerow(["stability", "tests_failed", stability["tests_failed"]])

    # Write latency CSV
    latency_csv = output_dir / "latency.csv"
    with open(latency_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "stage", "mean_ms", "p50_ms", "p95_ms", "min_ms"])
        for precision in ["fp32", "fp16"]:
            for stage in ["encoder", "obj_decoder", "edge_decoder", "total"]:
                key_mean = f"{precision}_{stage}_mean"
                if key_mean in latency:
                    writer.writerow([
                        precision, stage,
                        f"{latency[f'{precision}_{stage}_mean']:.2f}",
                        f"{latency[f'{precision}_{stage}_p50']:.2f}",
                        f"{latency[f'{precision}_{stage}_p95']:.2f}",
                        f"{latency[f'{precision}_{stage}_min']:.2f}",
                    ])

    print(f"  Metrics saved: {metrics_csv}")
    print(f"  Latency saved: {latency_csv}")

    return recommendation


def main():
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument("--image_dir", type=str, default="rgb_reg/")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--unified_checkpoint", type=str, default="checkpoints/vggt_unified_fp16.pt")
    parser.add_argument("--obj_checkpoint", type=str, default="checkpoints/obj_mask.pth")
    parser.add_argument("--edge_checkpoint", type=str, default="checkpoints/edge_mask.pt")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark_runs", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("UNIFIED MODEL PRODUCTION-READINESS EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Images: {args.image_dir} ({args.num_images} samples)")
    print(f"Checkpoint: {args.unified_checkpoint}")

    image_paths = get_image_paths(args.image_dir, args.num_images)
    if not image_paths:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)

    # Run all evaluations
    accuracy = evaluate_accuracy()
    regression = evaluate_regression(
        image_paths, args.unified_checkpoint,
        args.obj_checkpoint, args.edge_checkpoint
    )
    latency = evaluate_latency(
        image_paths, args.unified_checkpoint,
        warmup=args.warmup, runs=args.benchmark_runs
    )
    memory_results = evaluate_memory(image_paths, args.unified_checkpoint)
    stability = evaluate_stability(image_paths, args.unified_checkpoint)

    # Generate report
    recommendation = generate_report(
        accuracy, regression, latency, memory_results, stability, args.output_dir
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
