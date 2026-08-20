"""
Run inference on unlabeled images and save predictions.

Usage:
    python run_inference_save.py \
        --checkpoint /checkpoints/vggt_unified_fp16.pt \
        --images_dir /path/to/unlabeled_images/ \
        --output_dir predictions/baseline_24blocks \
        --num_warmup 5 \
        --num_profile 50

Saves:
    - predictions/*.npz (obj_mask, edge_mask for each image)
    - latency_stats.json (profiling results)
    - config.json (model configuration)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from model import VGGTUnified


def load_image(path: str, size: int = 518) -> torch.Tensor:
    """Load and preprocess image to [1, 1, 3, H, W]."""
    img = Image.open(path).convert("RGB").resize((size, size))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 518, 518]
    return tensor


def get_model_config(model):
    """Extract model configuration."""
    try:
        agg = model.aggregator
        return {
            'depth': agg.depth,
            'cached_layer_indices': list(agg.cached_layer_indices),
            'embed_dim': 1024,
            'num_heads': 16,
            'patch_size': agg.patch_size,
        }
    except Exception as e:
        return {'error': f'Could not extract config: {e}'}


def profile_latency(model, device, num_warmup, num_profile):
    """Profile model latency."""
    dummy_input = torch.rand(1, 1, 3, 518, 518).to(device)
    if dummy_input.dtype != model.aggregator.camera_token.dtype:
        dummy_input = dummy_input.half()

    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input, task="both")

    print(f"Profiling ({num_profile} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_profile), desc="Profiling"):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(dummy_input, task="both")

            if device.type == 'cuda':
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference and save predictions")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to unified checkpoint")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Directory with unlabeled images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save predictions")
    parser.add_argument("--num_warmup", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--num_profile", type=int, default=50,
                       help="Number of profiling iterations")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    print("="*70)
    print("INFERENCE & PREDICTION SAVING")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()

    model_config = get_model_config(model)
    print("Model Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    print()

    # Profile latency
    print("="*70)
    print("LATENCY PROFILING")
    print("="*70)
    latency_stats = profile_latency(model, device, args.num_warmup, args.num_profile)

    print(f"Mean:   {latency_stats['mean_ms']:.2f} ms")
    print(f"Median: {latency_stats['median_ms']:.2f} ms")
    print(f"Std:    {latency_stats['std_ms']:.2f} ms")
    print(f"P95:    {latency_stats['p95_ms']:.2f} ms")
    print()

    # Find images
    images_dir = Path(args.images_dir)
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    if not image_paths:
        print(f"ERROR: No images found in {images_dir}")
        return

    print("="*70)
    print(f"INFERENCE ON {len(image_paths)} IMAGES")
    print("="*70)

    # Run inference and save predictions
    for img_path in tqdm(image_paths, desc="Processing"):
        img_name = img_path.stem

        # Load image
        img_tensor = load_image(str(img_path)).to(device)

        # Run inference
        with torch.no_grad():
            results = model(img_tensor, task="both")

        # Save predictions
        pred_file = pred_dir / f"{img_name}.npz"
        np.savez_compressed(
            pred_file,
            obj_mask=results['obj_mask'][0, 0].cpu().numpy(),  # [2, 518, 518]
            edge_mask=results['edge_mask'][0, 0].cpu().numpy(),  # [1, 518, 518]
        )

    # Save metadata
    metadata = {
        'checkpoint': str(args.checkpoint),
        'images_dir': str(args.images_dir),
        'num_images': len(image_paths),
        'model_config': model_config,
        'latency_stats': latency_stats,
        'device': str(device),
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "latency_stats.json", 'w') as f:
        json.dump(latency_stats, f, indent=2)

    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Predictions saved: {pred_dir}")
    print(f"Config saved: {output_dir / 'config.json'}")
    print(f"Latency saved: {output_dir / 'latency_stats.json'}")
    print()
    print(f"Mean latency: {latency_stats['mean_ms']:.2f} ms")
    print(f"Images processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
