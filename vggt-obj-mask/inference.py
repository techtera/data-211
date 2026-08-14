"""
Inference script for VGGT + SegFormer.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image_dir path/to/images/ --output_dir results/
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image

from fine_tuning.config import DEVICE
from fine_tuning.model_builder import build_model


IMAGE_SIZE = 518


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    resized = cv2.resize(
        original, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )

    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor, original


@torch.no_grad()
def predict(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(1).to(DEVICE, non_blocking=True)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
        outputs = model(image_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    logits = outputs["mask_logits"]
    prediction = torch.argmax(logits, dim=1)
    mask = prediction.squeeze().cpu().numpy().astype(np.uint8)

    latency_ms = (t1 - t0) * 1000
    return mask, latency_ms


def save_outputs(mask, original_image, output_dir, stem):
    mask_resized = cv2.resize(
        mask,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_visual = (mask_resized * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/{stem}_mask.png", mask_visual)

    overlay = original_image.copy()
    overlay[mask_resized == 1] = [255, 0, 0]
    blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
    cv2.imwrite(
        f"{output_dir}/{stem}_overlay.png",
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
    )


def main():
    parser = argparse.ArgumentParser(description="VGGT + SegFormer Inference")
    parser.add_argument("--image", type=str, default=None, help="Path to a single input image")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint"
    )
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Output directory")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image_dir")

    print(f"\nDevice: {DEVICE}")
    print("Loading model...")

    model = build_model()
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Checkpoint loaded\n")

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    os.makedirs(args.output_dir, exist_ok=True)
    latencies = []

    print(f"Running inference on {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        image_tensor, original_image = load_image(img_path)
        mask, latency_ms = predict(model, image_tensor)
        latencies.append(latency_ms)

        print(f"  Latency: {latency_ms:.1f} ms")

        save_outputs(mask, original_image, args.output_dir, img_path.stem)
        print(f"  Saved: {img_path.stem}_mask.png, {img_path.stem}_overlay.png")

    print(f"\nDone. {len(image_paths)} image(s) processed.")
    if latencies:
        print(
            f"Latency — avg: {sum(latencies)/len(latencies):.1f} ms | "
            f"min: {min(latencies):.1f} ms | max: {max(latencies):.1f} ms"
        )


if __name__ == "__main__":
    main()