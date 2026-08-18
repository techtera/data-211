"""
Unified inference script for VGGT multi-task model.

Usage:
    python inference.py --image path/to/image.png --task cascade
    python inference.py --image_dir path/to/images/ --task obj
    python inference.py --image path/to/image.png --task cascade --save_dir output/
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize

from model import VGGTUnified


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 518


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor


OUTPUT_W, OUTPUT_H = 1280, 720
EDGE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(description="VGGT Unified Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--task", type=str, default="cascade",
                        choices=["obj", "edge", "both", "cascade"])
    parser.add_argument("--unified_checkpoint", type=str, default=None,
                        help="Path to unified checkpoint (skips HF download + separate decoders)")
    parser.add_argument("--obj_checkpoint", type=str, default="checkpoints/obj_mask.pth")
    parser.add_argument("--edge_checkpoint", type=str, default="checkpoints/edge_mask.pt")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save output visualizations")
    parser.add_argument("--no_encoder", action="store_true",
                        help="Skip loading encoder from HuggingFace (use random weights)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Task: {args.task}")

    if args.unified_checkpoint:
        model = VGGTUnified(load_encoder=False)
        model.load_unified_checkpoint(args.unified_checkpoint)
    else:
        model = VGGTUnified(load_encoder=not args.no_encoder)

        if args.task in ("obj", "both", "cascade") and args.obj_checkpoint:
            print(f"Loading obj-mask decoder from {args.obj_checkpoint}")
            model.load_decoder_checkpoint("obj", args.obj_checkpoint)

        if args.task in ("edge", "both", "cascade") and args.edge_checkpoint:
            ckpt_path = Path(args.edge_checkpoint)
            if ckpt_path.exists():
                print(f"Loading edge-mask decoder from {args.edge_checkpoint}")
                model.load_decoder_checkpoint("edge", args.edge_checkpoint)
            else:
                print(f"WARNING: Edge checkpoint not found at {args.edge_checkpoint}, using random weights")

    model = model.to(DEVICE)
    model.eval()

    # Load images
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir).resolve()
        image_paths = sorted([str(p) for p in image_dir.glob("*.png")])
        if not image_paths:
            image_paths = sorted([str(p) for p in image_dir.glob("*.jpg")])
        print(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("Provide --image or --image_dir")
        return

    print(f"Processing {len(image_paths)} images...")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Warmup with first image
    img_tensor = load_image(image_paths[0]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = model(img_tensor, task=args.task)

    latencies = []

    for i, img_path in enumerate(image_paths):
        img_tensor = load_image(img_path).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            results = model(img_tensor, task=args.task)

        latency = results.pop("latency")
        latencies.append(latency["total"])

        if save_dir and args.task == "cascade":
            stem = Path(img_path).stem
            edge_mask = results["edge_mask"]
            prob_map = edge_mask[0, 0].view(1, 1, 518, 518)
            prob_map = F.interpolate(
                prob_map, size=(OUTPUT_H, OUTPUT_W),
                mode="bilinear", align_corners=False,
            )
            binary_mask = (prob_map.squeeze().cpu() >= EDGE_THRESHOLD).numpy().astype(np.uint8)
            skeleton = skeletonize(binary_mask).astype(np.uint8) * 255
            Image.fromarray(skeleton, mode="L").save(save_dir / f"{stem}_edge_roi.png")

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            avg_ms = sum(latencies) / len(latencies) * 1000
            print(f"  [{i+1}/{len(image_paths)}] avg latency: {avg_ms:.1f} ms")

    # Print final stats
    avg_ms = sum(latencies) / len(latencies) * 1000
    print(f"\nDone. {len(image_paths)} images processed.")
    print(f"Latency — avg: {avg_ms:.1f} ms | "
          f"min: {min(latencies)*1000:.1f} ms | max: {max(latencies)*1000:.1f} ms")
    if save_dir:
        print(f"Results saved to {save_dir}/ ({OUTPUT_W}x{OUTPUT_H})")


if __name__ == "__main__":
    main()
