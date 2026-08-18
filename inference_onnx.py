"""
ONNX Runtime inference — runs on any hardware (A100, Orin NX, CPU).

Uses 3 ONNX models: shared encoder + 2 decoders.
ONNX Runtime auto-selects the best execution provider (CUDA EP, TRT EP, CPU EP).

Usage:
    python inference_onnx.py --image path/to/image.png --task both
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


def get_providers():
    available = ort.get_available_providers()
    providers = []
    if "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime Unified Inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--task", type=str, default="both", choices=["obj", "edge", "both"])
    parser.add_argument("--encoder_onnx", type=str, default="checkpoints/encoder.onnx")
    parser.add_argument("--obj_onnx", type=str, default="checkpoints/obj_decoder.onnx")
    parser.add_argument("--edge_onnx", type=str, default="checkpoints/edge_decoder.onnx")
    args = parser.parse_args()

    providers = get_providers()
    print(f"Execution providers: {providers}")

    # Load encoder session
    print("Loading encoder...")
    encoder_session = ort.InferenceSession(args.encoder_onnx, providers=providers)

    # Load decoder sessions
    obj_session = None
    edge_session = None

    if args.task in ("obj", "both"):
        print("Loading obj-mask decoder...")
        obj_session = ort.InferenceSession(args.obj_onnx, providers=providers)

    if args.task in ("edge", "both"):
        print("Loading edge-mask decoder...")
        edge_session = ort.InferenceSession(args.edge_onnx, providers=providers)

    # Prepare input: [B=1, S=1, 3, 518, 518]
    img = load_image(args.image)
    images = img[np.newaxis, np.newaxis, ...]  # [1, 1, 3, 518, 518]

    # Warmup
    encoder_outputs = encoder_session.run(None, {"images": images})

    # Timed inference
    t0 = time.perf_counter()

    # Encoder forward
    encoder_outputs = encoder_session.run(None, {"images": images})

    # Map encoder outputs to named tensors
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
        obj_out = obj_session.run(None, obj_inputs)
        results["obj_mask"] = obj_out[0]

    if edge_session:
        edge_inputs = {
            "layer_4": token_dict["layer_4_tokens"],
            "layer_11": token_dict["layer_11_tokens"],
            "layer_17": token_dict["layer_17_tokens"],
            "layer_23": token_dict["layer_23_tokens"],
        }
        edge_out = edge_session.run(None, edge_inputs)
        results["edge_mask"] = edge_out[0]

    elapsed = time.perf_counter() - t0

    print(f"\nInference time: {elapsed:.4f}s")
    if "obj_mask" in results:
        print(f"  obj_mask shape: {results['obj_mask'].shape}")
    if "edge_mask" in results:
        print(f"  edge_mask shape: {results['edge_mask'].shape}")


if __name__ == "__main__":
    main()
