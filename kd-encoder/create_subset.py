#!/usr/bin/env python3
"""
Create a random subset of images for faster training.

Usage:
    python create_subset.py --input train_images --output train_images_half --ratio 0.5
"""

import argparse
import os
import shutil
import random
from pathlib import Path


def create_subset(input_dir, output_dir, ratio=0.5, seed=42):
    """
    Create a random subset of images.

    Args:
        input_dir: Source image directory
        output_dir: Destination directory
        ratio: Fraction of images to keep (0.5 = half)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    all_images = []

    print(f"Scanning {input_dir}...")
    for ext in image_extensions:
        all_images.extend(list(input_path.glob(f"*{ext}")))
        all_images.extend(list(input_path.glob(f"*{ext.upper()}")))

    total = len(all_images)
    target = int(total * ratio)

    print(f"Found {total} images")
    print(f"Target: {target} images ({ratio*100:.0f}%)")

    # Random sample
    random.seed(seed)
    selected = random.sample(all_images, target)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy selected images
    print(f"\nCopying {len(selected)} images to {output_dir}...")
    for i, img_path in enumerate(selected):
        if (i + 1) % 1000 == 0:
            print(f"  Copied {i+1}/{len(selected)}...")

        dest = output_path / img_path.name
        shutil.copy2(img_path, dest)

    print(f"\n✓ Done! Created subset with {len(selected)} images")
    print(f"  Source: {input_dir} ({total} images)")
    print(f"  Destination: {output_dir} ({len(selected)} images)")
    print(f"  Reduction: {100 - ratio*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Create image dataset subset")
    parser.add_argument('--input', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for subset')
    parser.add_argument('--ratio', type=float, default=0.5,
                       help='Fraction of images to keep (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return

    if os.path.exists(args.output):
        response = input(f"Warning: '{args.output}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
        shutil.rmtree(args.output)

    create_subset(args.input, args.output, args.ratio, args.seed)


if __name__ == '__main__':
    main()
