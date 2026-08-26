"""
Simple FastVGGT Inference Script

This script shows you how to run inference with FastVGGT.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import VGGTUnified


def load_image(image_path, size=(518, 518)):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    return img_tensor


def load_images_from_folder(folder_path, num_frames=None, size=(518, 518)):
    """Load multiple images from a folder."""
    import glob

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                        glob.glob(os.path.join(folder_path, "*.png")))

    if not image_files:
        raise ValueError(f"No images found in {folder_path}")

    if num_frames:
        image_files = image_files[:num_frames]

    print(f"Loading {len(image_files)} images...")
    images = []
    for img_path in image_files:
        img = load_image(img_path, size)
        images.append(img)

    # Stack to [S, 3, H, W]
    images_tensor = torch.stack(images, dim=0)
    return images_tensor


def create_dummy_images(num_frames=5, size=(518, 518)):
    """Create dummy images for testing (if you don't have real images)."""
    print(f"Creating {num_frames} dummy images...")
    images = torch.rand(num_frames, 3, *size)
    return images


def save_results(results, output_dir="output"):
    """Save inference results."""
    os.makedirs(output_dir, exist_ok=True)

    obj_mask = results['obj_mask'][0]  # [S, 2, H, W]
    edge_mask = results['edge_mask'][0]  # [S, 1, H, W]

    num_frames = obj_mask.shape[0]

    for i in range(num_frames):
        # Object mask (convert to binary)
        obj_pred = obj_mask[i].argmax(dim=0).cpu().numpy()  # [H, W]
        obj_img = Image.fromarray((obj_pred * 255).astype(np.uint8))
        obj_img.save(os.path.join(output_dir, f"frame_{i:03d}_obj_mask.png"))

        # Edge mask
        edge_pred = edge_mask[i, 0].cpu().numpy()  # [H, W]
        edge_img = Image.fromarray((edge_pred * 255).astype(np.uint8))
        edge_img.save(os.path.join(output_dir, f"frame_{i:03d}_edge_mask.png"))

    print(f"✓ Results saved to {output_dir}/")


def measure_inference_latency(model, images, task, device, num_runs=3):
    """
    Accurately measure inference latency without including unnecessary overhead.

    Args:
        model: The model to test
        images: Input images tensor
        task: Task to run
        device: Device (cuda/cpu)
        num_runs: Number of runs to average

    Returns:
        avg_time_ms: Average inference time in milliseconds
        std_time_ms: Standard deviation in milliseconds
        breakdown: Dictionary with component timings
    """
    import time

    # Warmup (don't count)
    with torch.no_grad():
        _ = model(images, task=task)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []

    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()

        with torch.no_grad():
            result = model(images, task=task)

        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)  # Returns ms
        else:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        times.append(elapsed)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time, result['latency']


def run_inference(
    checkpoint_path='checkpoints/vggt_unified_fp16.pt',
    image_folder=None,
    num_frames=5,
    task='cascade',
    use_fastvggt=True,
    merge_ratio=0.9,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_output=True,
):
    """
    Run inference with FastVGGT.

    Args:
        checkpoint_path: Path to checkpoint
        image_folder: Folder with images (optional, will use dummy images if None)
        num_frames: Number of frames to process
        task: 'cascade', 'obj', 'edge', or 'both'
        use_fastvggt: Enable FastVGGT token merging
        merge_ratio: Token merging ratio (0.9 = merge 90%)
        device: 'cuda' or 'cpu'
        save_output: Save results to disk
    """
    print("=" * 80)
    print("FastVGGT Inference")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"FastVGGT enabled: {use_fastvggt}")
    if use_fastvggt:
        print(f"Merge ratio: {merge_ratio}")
    print()

    # Load model
    print("Loading model...")
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Enable FastVGGT if requested
    if use_fastvggt:
        # RoPE-First: RoPE applied before merging for full position preservation
        model.aggregator.enable_token_merging(merge_ratio=merge_ratio)
        print(f"✓ FastVGGT enabled (merge_ratio={merge_ratio}, RoPE-First architecture)")

    print()

    # Load images
    if image_folder and os.path.exists(image_folder):
        images = load_images_from_folder(image_folder, num_frames)
    else:
        if image_folder:
            print(f"⚠ Folder '{image_folder}' not found, using dummy images")
        images = create_dummy_images(num_frames)

    # Add batch dimension: [S, 3, H, W] -> [1, S, 3, H, W]
    images = images.unsqueeze(0).to(device)
    print(f"✓ Input shape: {list(images.shape)}")
    print()

    # Run inference with accurate latency measurement
    print(f"Running inference ({task} task)...")
    print("Measuring latency (warmup + 3 runs)...")
    print()

    avg_time, std_time, breakdown = measure_inference_latency(
        model, images, task, device, num_runs=3
    )

    # Run once more to get results for saving
    with torch.no_grad():
        results = model(images, task=task)

    print(f"✓ Inference complete!")
    print()

    # Print detailed latency breakdown
    print("=" * 80)
    print("LATENCY BREAKDOWN")
    print("=" * 80)
    print(f"Encoder:              {breakdown['encoder']*1000:>8.2f} ms")
    if 'obj_decoder' in breakdown:
        print(f"Obj decoder:          {breakdown['obj_decoder']*1000:>8.2f} ms")
    if 'edge_decoder' in breakdown:
        print(f"Edge decoder:         {breakdown['edge_decoder']*1000:>8.2f} ms")
    if 'roi_extraction' in breakdown:
        print(f"ROI extraction:       {breakdown['roi_extraction']*1000:>8.2f} ms")
    print("-" * 80)
    print(f"Total (sum):          {breakdown['total']*1000:>8.2f} ms")
    print()
    print(f"Measured (avg):       {avg_time:>8.2f} ms ± {std_time:.2f} ms")
    print(f"Per frame:            {avg_time / images.shape[1]:>8.2f} ms/frame")
    print(f"Throughput:           {1000 / (avg_time / images.shape[1]):>8.2f} FPS")
    print("=" * 80)
    print()

    # Print output shapes
    if 'obj_mask' in results:
        print(f"  - Obj mask shape: {list(results['obj_mask'].shape)}")
    if 'edge_mask' in results:
        print(f"  - Edge mask shape: {list(results['edge_mask'].shape)}")
    if 'roi_bbox' in results:
        num_bboxes = sum(1 for bbox in results['roi_bbox'][0] if bbox is not None)
        print(f"  - ROI bboxes detected: {num_bboxes}/{num_frames}")
    print()

    # Save results
    if save_output:
        save_results(results)

    print("=" * 80)
    print("✅ Inference complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FastVGGT inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/vggt_unified_fp16.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--images',
        type=str,
        default=None,
        help='Folder with input images (optional, uses dummy images if not provided)'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=5,
        help='Number of frames to process'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='cascade',
        choices=['cascade', 'obj', 'edge', 'both'],
        help='Task to run'
    )
    parser.add_argument(
        '--no-fastvggt',
        action='store_true',
        help='Disable FastVGGT (run baseline)'
    )
    parser.add_argument(
        '--merge-ratio',
        type=float,
        default=0.9,
        help='Token merging ratio (0.9 = merge 90%%)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output images'
    )

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        image_folder=args.images,
        num_frames=args.num_frames,
        task=args.task,
        use_fastvggt=not args.no_fastvggt,
        merge_ratio=args.merge_ratio,
        device=args.device,
        save_output=not args.no_save,
    )
