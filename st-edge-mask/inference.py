#!/usr/bin/env python3
"""
Edge Mask Inference Script

Runs inference on images using a trained checkpoint.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/checkpoint_best.pt \
                        --image path/to/image.jpg \
                        --output output.png

    # Batch inference on directory
    python inference.py --checkpoint checkpoints/checkpoint_best.pt \
                        --input_dir path/to/images/ \
                        --output_dir output/

    # With visualization overlay
    python inference.py --checkpoint checkpoints/checkpoint_best.pt \
                        --image path/to/image.jpg \
                        --output output.png \
                        --overlay

    # Custom threshold
    python inference.py --checkpoint checkpoints/checkpoint_best.pt \
                        --image path/to/image.jpg \
                        --output output.png \
                        --threshold 0.7
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time

sys.path.insert(0, "../kd-encoder")

from student import StudentAggregator
from edge_mask.model import StudentEdgeMask


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', '?')
    print(f"  Checkpoint epoch: {epoch}")

    # Load student encoder
    from fine_tuning.config import STUDENT_CHECKPOINT
    student_ckpt = torch.load(STUDENT_CHECKPOINT, map_location='cpu')
    state_dict = student_ckpt.get('student_state_dict', student_ckpt.get('model_state_dict', student_ckpt))

    student = StudentAggregator()
    student.load_state_dict(state_dict)
    student.eval()
    student.requires_grad_(False)

    # Build model
    model = StudentEdgeMask(student).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("  ✓ Model loaded")
    return model


def preprocess_image(image_path, size=518):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize to 518x518
    img = img.resize((size, size), Image.BILINEAR)

    # Convert to tensor [1, 1, 3, 518, 518] (with S=1 dimension)
    img_tensor = torch.from_numpy(np.array(img)).float()
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions

    return img_tensor, original_size, img


def run_inference(model, image_tensor, threshold=0.5, device='cuda'):
    """Run inference on a single image."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        start = time.time()
        # Model in eval mode returns sigmoid output [B, S, 1, H, W]
        output = model(image_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        latency = (time.time() - start) * 1000  # ms

        # Threshold and squeeze to [H, W]
        pred = (output.squeeze() > threshold).cpu().numpy().astype(np.uint8)

    return pred, latency


def save_mask(mask, output_path, original_size=None):
    """Save binary edge mask as image."""
    # Convert to 0/255
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode='L')

    # Resize back to original if needed
    if original_size:
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)

    mask_pil.save(output_path)


def create_overlay(image, mask, color=(0, 255, 0), alpha=0.7):
    """Create visualization with edge overlay."""
    overlay = np.array(image).copy()

    # Create colored edge overlay (default: green)
    edge_overlay = overlay.copy()
    edge_overlay[mask == 1] = color

    # Blend only where edges exist
    result = overlay.copy()
    edge_pixels = mask == 1
    result[edge_pixels] = (overlay[edge_pixels] * (1 - alpha) +
                           edge_overlay[edge_pixels] * alpha).astype(np.uint8)

    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description='Edge Mask Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--image', type=str,
                        help='Path to single input image')
    parser.add_argument('--input_dir', type=str,
                        help='Path to input directory (batch mode)')
    parser.add_argument('--output', type=str,
                        help='Path to output mask (single image mode)')
    parser.add_argument('--output_dir', type=str,
                        help='Path to output directory (batch mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Edge detection threshold (0-1, default: 0.5)')
    parser.add_argument('--overlay', action='store_true',
                        help='Save visualization overlay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.input_dir:
        parser.error("Must provide either --image or --input_dir")

    if args.image and not args.output:
        parser.error("Must provide --output when using --image")

    if args.input_dir and not args.output_dir:
        parser.error("Must provide --output_dir when using --input_dir")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}\n")

    # Load model
    model = load_model(args.checkpoint, device)

    # Single image mode
    if args.image:
        print(f"\nProcessing: {args.image}")

        # Preprocess
        img_tensor, original_size, img_pil = preprocess_image(args.image)

        # Inference
        mask, latency = run_inference(model, img_tensor, args.threshold, device)
        print(f"  Latency: {latency:.2f}ms")

        # Stats
        edge_pixels = mask.sum()
        total_pixels = mask.size
        edge_density = edge_pixels / total_pixels * 100
        print(f"  Edge density: {edge_density:.2f}% ({edge_pixels:,}/{total_pixels:,} pixels)")

        # Save mask
        save_mask(mask, args.output, original_size)
        print(f"  ✓ Mask saved: {args.output}")

        # Save overlay if requested
        if args.overlay:
            overlay_path = args.output.replace('.png', '_overlay.png')
            overlay_img = create_overlay(img_pil, mask)
            overlay_img.save(overlay_path)
            print(f"  ✓ Overlay saved: {overlay_path}")

    # Batch mode
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(input_dir.glob(ext))

        print(f"\nFound {len(image_files)} images in {input_dir}")
        print("Processing...\n")

        total_latency = 0
        for i, img_path in enumerate(image_files, 1):
            # Preprocess
            img_tensor, original_size, img_pil = preprocess_image(img_path)

            # Inference
            mask, latency = run_inference(model, img_tensor, args.threshold, device)
            total_latency += latency

            # Save mask
            output_path = output_dir / f"{img_path.stem}_edges.png"
            save_mask(mask, output_path, original_size)

            # Save overlay if requested
            if args.overlay:
                overlay_path = output_dir / f"{img_path.stem}_overlay.png"
                overlay_img = create_overlay(img_pil, mask)
                overlay_img.save(overlay_path)

            if i % 10 == 0:
                print(f"  [{i}/{len(image_files)}] Processed {img_path.name} ({latency:.2f}ms)")

        avg_latency = total_latency / len(image_files)
        print(f"\n✓ Complete!")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Throughput: {1000/avg_latency:.2f} images/sec")
        print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
