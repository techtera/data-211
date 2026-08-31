#!/usr/bin/env python3
"""
Object Mask Inference Script

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
from obj_mask.model import StudentObjMask


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
    model = StudentObjMask(student).to(device)
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

    # Convert to tensor [1, 3, 518, 518]
    img_tensor = torch.from_numpy(np.array(img)).float()
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_tensor, original_size, img


def run_inference(model, image_tensor, device='cuda'):
    """Run inference on a single image."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        start = time.time()
        logits = model(image_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        latency = (time.time() - start) * 1000  # ms

        # Get prediction [B, H, W]
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred, latency


def save_mask(mask, output_path, original_size=None):
    """Save binary mask as image."""
    # Convert to 0/255
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode='L')

    # Resize back to original if needed
    if original_size:
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)

    mask_pil.save(output_path)


def create_overlay(image, mask, alpha=0.5):
    """Create visualization with mask overlay."""
    # Convert mask to RGB (object = red)
    overlay = np.array(image).copy()
    red_mask = np.zeros_like(overlay)
    red_mask[mask == 1] = [255, 0, 0]  # Red for object

    # Blend
    result = (overlay * (1 - alpha) + red_mask * alpha).astype(np.uint8)
    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description='Object Mask Inference')
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
    print(f"Device: {device}\n")

    # Load model
    model = load_model(args.checkpoint, device)

    # Single image mode
    if args.image:
        print(f"\nProcessing: {args.image}")

        # Preprocess
        img_tensor, original_size, img_pil = preprocess_image(args.image)

        # Inference
        mask, latency = run_inference(model, img_tensor, device)
        print(f"  Latency: {latency:.2f}ms")

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
            mask, latency = run_inference(model, img_tensor, device)
            total_latency += latency

            # Save mask
            output_path = output_dir / f"{img_path.stem}_mask.png"
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
