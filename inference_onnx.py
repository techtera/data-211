"""
ONNX Runtime inference with latency benchmarking.

Uses 3 ONNX models: shared encoder + 2 decoders.
ONNX Runtime auto-selects the best execution provider (CUDA EP, TRT EP, CPU EP).

Usage:
    python inference_onnx.py --image path/to/image.png --task both
    python inference_onnx.py --image path/to/image.png --task both --benchmark 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Install onnxruntime: pip install onnxruntime-gpu")


IMAGE_SIZE = 518


def get_providers(device_id=1):
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
    providers.append("CPUExecutionProvider")
    return providers


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr


def run_inference(encoder_session, obj_session, edge_session, images):
    """Single inference pass. Returns (results_dict, per_stage_latency)."""
    latency = {}

    t0 = time.perf_counter()
    encoder_outputs = encoder_session.run(None, {"images": images})
    latency["encoder"] = time.perf_counter() - t0

    encoder_output_names = [o.name for o in encoder_session.get_outputs()]
    token_dict = dict(zip(encoder_output_names, encoder_outputs))

    results = {}

    if obj_session:
        obj_inputs = {
            "layer_4": token_dict["layer_4_tokens"],
            "layer_11": token_dict["layer_11_tokens"],
            "layer_17": token_dict["layer_17_tokens"],
            "layer_23": token_dict["layer_23_tokens"],
            "images": images,
        }
        t1 = time.perf_counter()
        obj_out = obj_session.run(None, obj_inputs)
        latency["obj_decoder"] = time.perf_counter() - t1
        results["obj_mask"] = obj_out[0]

    if edge_session:
        edge_inputs = {
            "layer_4": token_dict["layer_4_tokens"],
            "layer_11": token_dict["layer_11_tokens"],
            "layer_17": token_dict["layer_17_tokens"],
            "layer_23": token_dict["layer_23_tokens"],
        }
        t1 = time.perf_counter()
        edge_out = edge_session.run(None, edge_inputs)
        latency["edge_decoder"] = time.perf_counter() - t1
        results["edge_mask"] = edge_out[0]

    latency["total"] = sum(latency.values())
    return results, latency


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime Unified Inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--task", type=str, default="both", choices=["obj", "edge", "both"])
    parser.add_argument("--encoder_onnx", type=str, default="onnx_models/encoder.onnx")
    parser.add_argument("--obj_onnx", type=str, default="onnx_models/obj_decoder.onnx")
    parser.add_argument("--edge_onnx", type=str, default="onnx_models/edge_decoder.onnx")
    parser.add_argument("--benchmark", type=int, default=10,
                        help="Number of timed runs (after warmup)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    args = parser.parse_args()

    providers = get_providers()
    print(f"Execution providers: {providers}")

    print("Loading encoder...")
    encoder_session = ort.InferenceSession(args.encoder_onnx, providers=providers)

    obj_session = None
    edge_session = None

    if args.task in ("obj", "both"):
        print("Loading obj-mask decoder...")
        obj_session = ort.InferenceSession(args.obj_onnx, providers=providers)

    if args.task in ("edge", "both"):
        print("Loading edge-mask decoder...")
        edge_session = ort.InferenceSession(args.edge_onnx, providers=providers)

    img = load_image(args.image)
    images = img[np.newaxis, np.newaxis, ...]  # [1, 1, 3, 518, 518]

    # Warmup
    print(f"\nWarmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        run_inference(encoder_session, obj_session, edge_session, images)

    # Benchmark
    print(f"Benchmarking ({args.benchmark} runs)...")
    all_latencies = []
    for i in range(args.benchmark):
        results, latency = run_inference(encoder_session, obj_session, edge_session, images)
        all_latencies.append(latency)

    # Compute stats
    stages = list(all_latencies[0].keys())
    print(f"\n{'Stage':<15} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("-" * 55)
    for stage in stages:
        values = [l[stage] for l in all_latencies]
        mean = np.mean(values)
        mn = np.min(values)
        mx = np.max(values)
        std = np.std(values)
        print(f"{stage:<15} {mean*1000:>7.1f}ms {mn*1000:>7.1f}ms {mx*1000:>7.1f}ms {std*1000:>7.1f}ms")

    print(f"\nOutput shapes:")
    if "obj_mask" in results:
        print(f"  obj_mask: {results['obj_mask'].shape}")
    if "edge_mask" in results:
        print(f"  edge_mask: {results['edge_mask'].shape}")


if __name__ == "__main__":
    main()
