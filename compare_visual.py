"""
Create side-by-side visual comparisons between two model predictions.

Usage:
    python compare_visual.py \
        --baseline_dir predictions/baseline_24blocks \
        --test_dir predictions/truncated_20blocks \
        --images_dir /path/to/unlabeled_images/ \
        --output_dir comparisons/24vs20blocks \
        --num_samples 50

Creates:
    - Side-by-side comparison images (obj masks and edge masks)
    - Difference statistics
    - Latency comparison
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image


def load_prediction(npz_path):
    """Load prediction from .npz file."""
    data = np.load(npz_path)
    return {
        'obj_mask': torch.from_numpy(data['obj_mask']),  # [2, 518, 518]
        'edge_mask': torch.from_numpy(data['edge_mask']),  # [1, 518, 518]
    }


def compute_differences(baseline_pred, test_pred):
    """Compute pixel-wise differences."""
    # Object mask difference
    baseline_obj = baseline_pred['obj_mask'].argmax(dim=0)  # [518, 518]
    test_obj = test_pred['obj_mask'].argmax(dim=0)

    obj_diff = (baseline_obj != test_obj).float()
    obj_diff_percent = (obj_diff.sum() / obj_diff.numel()).item() * 100

    # Edge mask difference
    baseline_edge = baseline_pred['edge_mask'][0]  # [518, 518]
    test_edge = test_pred['edge_mask'][0]

    edge_diff = torch.abs(baseline_edge - test_edge)
    edge_diff_mean = edge_diff.mean().item()
    edge_diff_max = edge_diff.max().item()

    return {
        'obj_diff_percent': obj_diff_percent,
        'edge_diff_mean': edge_diff_mean,
        'edge_diff_max': edge_diff_max,
    }


def create_comparison_image(img_path, baseline_pred, test_pred, save_path):
    """Create side-by-side comparison visualization."""
    # Load original image
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (518, 518))

    # Get predictions
    baseline_obj = baseline_pred['obj_mask'].argmax(dim=0).numpy().astype(np.uint8)
    test_obj = test_pred['obj_mask'].argmax(dim=0).numpy().astype(np.uint8)

    baseline_edge = (baseline_pred['edge_mask'][0].numpy() > 0.5).astype(np.uint8) * 255
    test_edge = (test_pred['edge_mask'][0].numpy() > 0.5).astype(np.uint8) * 255

    # Create colored overlays for object masks
    baseline_obj_color = img.copy()
    baseline_obj_color[baseline_obj == 1] = [0, 255, 0]  # Green

    test_obj_color = img.copy()
    test_obj_color[test_obj == 1] = [0, 255, 0]  # Green

    # Difference map for object masks
    obj_diff = (baseline_obj != test_obj).astype(np.uint8) * 255
    obj_diff_color = cv2.applyColorMap(obj_diff, cv2.COLORMAP_HOT)

    # Edge difference
    edge_diff = np.abs(baseline_edge.astype(int) - test_edge.astype(int)).astype(np.uint8)
    edge_diff_color = cv2.applyColorMap(edge_diff, cv2.COLORMAP_HOT)

    # Layout: 4 rows x 3 columns
    # Row 1: Original | Baseline Obj | Test Obj
    # Row 2: Obj Diff | Baseline Edge | Test Edge
    # Row 3: Edge Diff | Empty | Empty

    row1 = np.hstack([img, baseline_obj_color, test_obj_color])
    row2 = np.hstack([
        obj_diff_color,
        cv2.cvtColor(baseline_edge, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(test_edge, cv2.COLOR_GRAY2BGR)
    ])
    row3 = np.hstack([
        edge_diff_color,
        np.zeros_like(img),
        np.zeros_like(img)
    ])

    vis = np.vstack([row1, row2, row3])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)

    # Row 1 labels
    cv2.putText(vis, 'Original', (10, 30), font, font_scale, color, thickness)
    cv2.putText(vis, 'Baseline Obj', (528, 30), font, font_scale, color, thickness)
    cv2.putText(vis, 'Test Obj', (1046, 30), font, font_scale, color, thickness)

    # Row 2 labels
    cv2.putText(vis, 'Obj Diff', (10, 548), font, font_scale, color, thickness)
    cv2.putText(vis, 'Baseline Edge', (528, 548), font, font_scale, color, thickness)
    cv2.putText(vis, 'Test Edge', (1046, 548), font, font_scale, color, thickness)

    # Row 3 labels
    cv2.putText(vis, 'Edge Diff', (10, 1066), font, font_scale, color, thickness)

    cv2.imwrite(str(save_path), vis)


def main():
    parser = argparse.ArgumentParser(description="Visual comparison between model predictions")
    parser.add_argument("--baseline_dir", type=str, required=True,
                       help="Baseline predictions directory")
    parser.add_argument("--test_dir", type=str, required=True,
                       help="Test predictions directory")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Original images directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for comparisons")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of comparison images to create (default: all)")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    test_dir = Path(args.test_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visual_comparisons"
    vis_dir.mkdir(exist_ok=True)

    print("="*70)
    print("VISUAL COMPARISON")
    print("="*70)
    print(f"Baseline: {baseline_dir}")
    print(f"Test:     {test_dir}")
    print(f"Images:   {images_dir}")
    print(f"Output:   {output_dir}")
    print()

    # Load configs
    baseline_config = json.load(open(baseline_dir / "config.json"))
    test_config = json.load(open(test_dir / "config.json"))

    print("Baseline Model:")
    for key, value in baseline_config['model_config'].items():
        print(f"  {key}: {value}")

    print("\nTest Model:")
    for key, value in test_config['model_config'].items():
        print(f"  {key}: {value}")

    # Latency comparison
    baseline_latency = baseline_config['latency_stats']
    test_latency = test_config['latency_stats']

    speedup = (baseline_latency['mean_ms'] - test_latency['mean_ms']) / baseline_latency['mean_ms'] * 100

    print()
    print("="*70)
    print("LATENCY COMPARISON")
    print("="*70)
    print(f"Baseline mean: {baseline_latency['mean_ms']:.2f} ms")
    print(f"Test mean:     {test_latency['mean_ms']:.2f} ms")
    print(f"Speedup:       {speedup:+.1f}%")
    print()

    # Find all predictions
    baseline_pred_dir = baseline_dir / "predictions"
    test_pred_dir = test_dir / "predictions"

    baseline_files = sorted(list(baseline_pred_dir.glob("*.npz")))
    test_files = sorted(list(test_pred_dir.glob("*.npz")))

    # Match files
    common_files = []
    for bf in baseline_files:
        tf = test_pred_dir / bf.name
        if tf.exists():
            common_files.append(bf.stem)

    if not common_files:
        print("ERROR: No common predictions found")
        return

    print(f"Found {len(common_files)} common predictions")

    # Compute differences for all
    print("\nComputing differences...")
    all_diffs = []
    for img_name in tqdm(common_files, desc="Computing"):
        baseline_pred = load_prediction(baseline_pred_dir / f"{img_name}.npz")
        test_pred = load_prediction(test_pred_dir / f"{img_name}.npz")

        diff = compute_differences(baseline_pred, test_pred)
        diff['image'] = img_name
        all_diffs.append(diff)

    # Sort by object difference (largest first)
    all_diffs.sort(key=lambda x: x['obj_diff_percent'], reverse=True)

    # Aggregate stats
    obj_diffs = [d['obj_diff_percent'] for d in all_diffs]
    edge_diffs = [d['edge_diff_mean'] for d in all_diffs]

    print()
    print("="*70)
    print("QUALITY DIFFERENCES")
    print("="*70)
    print(f"Object mask diff (mean): {np.mean(obj_diffs):.2f}% pixels")
    print(f"Object mask diff (max):  {np.max(obj_diffs):.2f}% pixels")
    print(f"Edge mask diff (mean):   {np.mean(edge_diffs):.4f}")
    print(f"Edge mask diff (max):    {np.max([d['edge_diff_max'] for d in all_diffs]):.4f}")
    print()

    # Select images to visualize
    if args.num_samples == -1 or args.num_samples >= len(all_diffs):
        selected = all_diffs
    else:
        # Take top N with largest differences
        selected = all_diffs[:args.num_samples]

    print(f"Creating {len(selected)} comparison images...")

    for i, diff in enumerate(tqdm(selected, desc="Creating visuals")):
        img_name = diff['image']
        img_path = images_dir / f"{img_name}.png"
        if not img_path.exists():
            img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        baseline_pred = load_prediction(baseline_pred_dir / f"{img_name}.npz")
        test_pred = load_prediction(test_pred_dir / f"{img_name}.npz")

        save_name = f"{i:03d}_{img_name}_objdiff{diff['obj_diff_percent']:.1f}.png"
        create_comparison_image(
            img_path,
            baseline_pred,
            test_pred,
            vis_dir / save_name
        )

    # Save summary
    summary = {
        'baseline': baseline_config,
        'test': test_config,
        'latency_comparison': {
            'baseline_mean_ms': baseline_latency['mean_ms'],
            'test_mean_ms': test_latency['mean_ms'],
            'speedup_percent': speedup,
        },
        'quality_comparison': {
            'obj_diff_mean_percent': float(np.mean(obj_diffs)),
            'obj_diff_max_percent': float(np.max(obj_diffs)),
            'edge_diff_mean': float(np.mean(edge_diffs)),
            'edge_diff_max': float(np.max([d['edge_diff_max'] for d in all_diffs])),
            'num_images': len(common_files),
        },
        'all_differences': all_diffs[:100],  # Save top 100
    }

    with open(output_dir / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Speedup:         {speedup:+.1f}%")
    print(f"Obj diff (mean): {np.mean(obj_diffs):.2f}% pixels")
    print(f"Edge diff (mean): {np.mean(edge_diffs):.4f}")
    print()
    print(f"Visual comparisons saved to: {vis_dir}")
    print(f"Summary saved to: {output_dir / 'comparison_summary.json'}")
    print()

    if speedup > 10 and np.mean(obj_diffs) < 5.0:
        print("✅ Good speedup with acceptable quality difference")
    elif speedup > 0:
        print("⚠️  Some speedup, review visual comparisons for quality")
    else:
        print("❌ No speedup observed")


if __name__ == "__main__":
    main()
