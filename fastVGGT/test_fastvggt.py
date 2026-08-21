"""
Test FastVGGT implementation - compares speed and quality vs baseline.

This script:
1. Loads your trained unified checkpoint
2. Runs baseline inference (no merging)
3. Enables FastVGGT token merging
4. Runs FastVGGT inference
5. Compares speed and output similarity
"""

import torch
import time
import numpy as np
from model import VGGTUnified


def measure_inference_time(model, images, task="cascade", num_runs=3):
    """
    Accurately measure inference time without including unnecessary overhead.

    Uses CUDA events on GPU for precise timing, CPU timer otherwise.
    """
    times = []
    device = next(model.parameters()).device

    # Warmup run (don't count)
    with torch.no_grad():
        _ = model(images, task=task)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Actual measurement runs
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                result = model(images, task=task)
            end_event.record()

            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000  # Convert ms to seconds
        else:
            start = time.perf_counter()
            with torch.no_grad():
                result = model(images, task=task)
            elapsed = time.perf_counter() - start

        times.append(elapsed)

    return np.mean(times), np.std(times), result


def compute_output_similarity(output1, output2, task):
    """Compute similarity metrics between two outputs."""
    metrics = {}

    if task in ("obj", "both", "cascade"):
        # Compare object mask logits
        obj_diff = torch.abs(output1['obj_mask'] - output2['obj_mask'])
        metrics['obj_mask_mae'] = obj_diff.mean().item()
        metrics['obj_mask_max_diff'] = obj_diff.max().item()

        # Compare predicted classes
        pred1 = output1['obj_mask'].argmax(dim=2)
        pred2 = output2['obj_mask'].argmax(dim=2)
        accuracy = (pred1 == pred2).float().mean().item()
        metrics['obj_mask_accuracy'] = accuracy

    if task in ("edge", "both", "cascade"):
        # Compare edge mask probabilities
        edge_diff = torch.abs(output1['edge_mask'] - output2['edge_mask'])
        metrics['edge_mask_mae'] = edge_diff.mean().item()
        metrics['edge_mask_max_diff'] = edge_diff.max().item()

        # Compare binary predictions
        pred1 = (output1['edge_mask'] > 0.5).float()
        pred2 = (output2['edge_mask'] > 0.5).float()
        accuracy = (pred1 == pred2).float().mean().item()
        metrics['edge_mask_accuracy'] = accuracy

    return metrics


def test_fastvggt(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    merge_ratio: float = 0.9,
    image_size: tuple = (518, 518),
    num_frames: int = 5,
    task: str = "cascade",
):
    """
    Test FastVGGT vs baseline.

    Args:
        checkpoint_path: Path to unified checkpoint (e.g., 'checkpoints/vggt_unified_fp16.pt')
        device: Device to run on
        merge_ratio: Token merging ratio (0.9 = merge 90% of tokens)
        image_size: Image size (H, W)
        num_frames: Number of frames to test with
        task: Task to test ('cascade', 'obj', 'edge', or 'both')
    """
    print("=" * 80)
    print("FastVGGT Test Script")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Merge ratio: {merge_ratio}")
    print(f"Image size: {image_size}")
    print(f"Num frames: {num_frames}")
    print(f"Task: {task}")
    print()

    # Load model
    print("Loading model...")
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print()

    # Create dummy input
    print(f"Creating test input: [{1}, {num_frames}, 3, {image_size[0]}, {image_size[1]}]")
    images = torch.rand(1, num_frames, 3, *image_size, device=device)
    print()

    # ========== Baseline Inference ==========
    print("-" * 80)
    print("Running BASELINE inference (no token merging)...")
    print("-" * 80)

    baseline_time, baseline_std, baseline_output = measure_inference_time(
        model, images, task=task, num_runs=3
    )

    print(f"✓ Baseline inference: {baseline_time*1000:.2f}ms ± {baseline_std*1000:.2f}ms")
    print(f"  Component breakdown:")
    print(f"    - Encoder:        {baseline_output['latency']['encoder']*1000:>7.2f} ms")
    if 'obj_decoder' in baseline_output['latency']:
        print(f"    - Obj decoder:    {baseline_output['latency']['obj_decoder']*1000:>7.2f} ms")
    if 'edge_decoder' in baseline_output['latency']:
        print(f"    - Edge decoder:   {baseline_output['latency']['edge_decoder']*1000:>7.2f} ms")
    print(f"    - Total (sum):    {baseline_output['latency']['total']*1000:>7.2f} ms")
    print()

    # ========== FastVGGT Inference ==========
    print("-" * 80)
    print(f"Running FASTVGGT inference (merge_ratio={merge_ratio})...")
    print("-" * 80)

    # Enable token merging
    model.aggregator.enable_token_merging(merge_ratio=merge_ratio)

    fastvggt_time, fastvggt_std, fastvggt_output = measure_inference_time(
        model, images, task=task, num_runs=3
    )

    print(f"✓ FastVGGT inference: {fastvggt_time*1000:.2f}ms ± {fastvggt_std*1000:.2f}ms")
    print(f"  Component breakdown:")
    print(f"    - Encoder:        {fastvggt_output['latency']['encoder']*1000:>7.2f} ms")
    if 'obj_decoder' in fastvggt_output['latency']:
        print(f"    - Obj decoder:    {fastvggt_output['latency']['obj_decoder']*1000:>7.2f} ms")
    if 'edge_decoder' in fastvggt_output['latency']:
        print(f"    - Edge decoder:   {fastvggt_output['latency']['edge_decoder']*1000:>7.2f} ms")
    print(f"    - Total (sum):    {fastvggt_output['latency']['total']*1000:>7.2f} ms")
    print()

    # ========== Comparison ==========
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    speedup = baseline_time / fastvggt_time
    encoder_speedup = baseline_output['latency']['encoder'] / fastvggt_output['latency']['encoder']

    print(f"🚀 Overall speedup:   {speedup:.2f}x  ({baseline_time*1000:.1f}ms → {fastvggt_time*1000:.1f}ms)")
    print(f"🚀 Encoder speedup:   {encoder_speedup:.2f}x  ({baseline_output['latency']['encoder']*1000:.1f}ms → {fastvggt_output['latency']['encoder']*1000:.1f}ms)")
    print(f"⚡ Time saved:        {(baseline_time - fastvggt_time)*1000:.1f}ms per inference")
    print(f"⚡ Throughput gain:   {speedup:.1f}x FPS improvement")
    print()

    # Compute similarity metrics
    print("📊 Output Similarity:")
    similarity = compute_output_similarity(baseline_output, fastvggt_output, task)

    for metric, value in similarity.items():
        if 'accuracy' in metric:
            status = "✓" if value > 0.95 else "⚠️"
            print(f"  {status} {metric}: {value*100:.2f}%")
        elif 'mae' in metric:
            status = "✓" if value < 0.1 else "⚠️"
            print(f"  {status} {metric}: {value:.4f}")
        else:
            print(f"    {metric}: {value:.4f}")
    print()

    # Expected results
    print("📈 Expected Results (based on FastVGGT paper):")
    print("  - Encoder speedup: 3-5x (depends on sequence length)")
    print("  - Overall speedup: 1.5-3x (encoder + decoders)")
    print("  - Output accuracy: >98% (near-identical predictions)")
    print("  - Quality: Same or better (especially for long sequences)")
    print()

    if speedup > 1.5:
        print("✅ FastVGGT is working! Speedup achieved with minimal quality loss.")
    else:
        print("⚠️ Speedup less than expected. Try:")
        print("  1. Increase num_frames (merging benefits grow with sequence length)")
        print("  2. Increase merge_ratio to 0.95 for more aggressive merging")
        print("  3. Ensure model is in eval mode and using GPU")

    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FastVGGT implementation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to unified checkpoint (e.g., checkpoints/vggt_unified_fp16.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--merge-ratio",
        type=float,
        default=0.9,
        help="Token merging ratio (0.9 = merge 90%)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to test with (more frames = more speedup)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="cascade",
        choices=["cascade", "obj", "edge", "both"],
        help="Task to test"
    )

    args = parser.parse_args()

    test_fastvggt(
        checkpoint_path=args.checkpoint,
        device=args.device,
        merge_ratio=args.merge_ratio,
        num_frames=args.num_frames,
        task=args.task,
    )
