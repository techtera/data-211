"""
FastVGGT Example Usage

Simple examples showing how to use FastVGGT with your trained model.
"""

import torch
from model import VGGTUnified


def example_1_basic_usage():
    """Example 1: Basic FastVGGT usage - single toggle"""
    print("=" * 80)
    print("Example 1: Basic FastVGGT Usage")
    print("=" * 80)

    # Load your trained model
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt')
    model.eval()

    # Create dummy input
    images = torch.rand(1, 10, 3, 518, 518)

    print("Running baseline inference...")
    baseline_result = model(images, task='cascade')
    print(f"Baseline time: {baseline_result['latency']['total']*1000:.2f}ms")

    # Enable FastVGGT (one line!)
    print("\nEnabling FastVGGT...")
    model.aggregator.enable_token_merging(merge_ratio=0.9)

    print("Running FastVGGT inference...")
    fastvggt_result = model(images, task='cascade')
    print(f"FastVGGT time: {fastvggt_result['latency']['total']*1000:.2f}ms")

    speedup = baseline_result['latency']['total'] / fastvggt_result['latency']['total']
    print(f"\n🚀 Speedup: {speedup:.2f}x")
    print()


def example_2_comparison():
    """Example 2: Compare different merge ratios"""
    print("=" * 80)
    print("Example 2: Compare Different Merge Ratios")
    print("=" * 80)

    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt')
    model.eval()

    images = torch.rand(1, 10, 3, 518, 518)

    # Baseline
    print("Baseline (no merging):")
    result = model(images, task='obj')
    baseline_time = result['latency']['encoder']
    print(f"  Encoder time: {baseline_time*1000:.2f}ms")

    # Test different ratios
    ratios = [0.7, 0.8, 0.9, 0.95]
    for ratio in ratios:
        model.aggregator.enable_token_merging(merge_ratio=ratio)
        result = model(images, task='obj')
        time_ms = result['latency']['encoder'] * 1000
        speedup = baseline_time / result['latency']['encoder']
        print(f"  Merge ratio {ratio}: {time_ms:.2f}ms (speedup: {speedup:.2f}x)")
        model.aggregator.disable_token_merging()

    print()


def example_3_long_sequences():
    """Example 3: Process long sequences efficiently"""
    print("=" * 80)
    print("Example 3: Long Sequence Processing")
    print("=" * 80)

    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt')
    model.eval()
    model.aggregator.enable_token_merging(merge_ratio=0.9)

    # Process video sequences of different lengths
    sequence_lengths = [5, 10, 20, 50]

    print("Processing video sequences with FastVGGT enabled:")
    for num_frames in sequence_lengths:
        images = torch.rand(1, num_frames, 3, 518, 518)

        result = model(images, task='cascade')
        total_time = result['latency']['total']
        per_frame = total_time / num_frames

        print(f"  {num_frames} frames: {total_time*1000:.2f}ms total, "
              f"{per_frame*1000:.2f}ms per frame")

    print()


def example_4_production_pipeline():
    """Example 4: Production inference pipeline"""
    print("=" * 80)
    print("Example 4: Production Pipeline")
    print("=" * 80)

    # Setup model once
    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt')

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Enable FastVGGT for production
    model.aggregator.enable_token_merging(
        merge_ratio=0.9,
        salient_stride=10,
        apply_from_block=0,
    )

    print("✓ Model loaded and FastVGGT enabled")
    print("✓ Ready for inference")

    # Simulate processing a batch
    print("\nProcessing batch...")
    images = torch.rand(1, 15, 3, 518, 518)
    if torch.cuda.is_available():
        images = images.cuda()

    with torch.no_grad():
        result = model(images, task='cascade')

    print(f"✓ Processed {images.shape[1]} frames in {result['latency']['total']*1000:.2f}ms")
    print(f"  - Obj mask shape: {result['obj_mask'].shape}")
    print(f"  - Edge mask shape: {result['edge_mask'].shape}")
    print(f"  - ROI bboxes: {len(result['roi_bbox'][0])} detected")
    print()


def example_5_adaptive_merging():
    """Example 5: Adaptive merging based on sequence length"""
    print("=" * 80)
    print("Example 5: Adaptive Merging")
    print("=" * 80)

    model = VGGTUnified(load_encoder=False)
    model.load_unified_checkpoint('checkpoints/vggt_unified_fp16.pt')
    model.eval()

    def process_with_adaptive_merging(images, model):
        """Adaptively choose merge ratio based on sequence length."""
        num_frames = images.shape[1]

        if num_frames < 5:
            # Short sequence - no merging needed
            print(f"  {num_frames} frames: No merging (too short)")
            model.aggregator.disable_token_merging()
        elif num_frames < 15:
            # Medium sequence - moderate merging
            print(f"  {num_frames} frames: merge_ratio=0.8 (moderate)")
            model.aggregator.enable_token_merging(merge_ratio=0.8)
        else:
            # Long sequence - aggressive merging
            print(f"  {num_frames} frames: merge_ratio=0.9 (aggressive)")
            model.aggregator.enable_token_merging(merge_ratio=0.9)

        with torch.no_grad():
            return model(images, task='obj')

    # Test with different lengths
    print("Adaptive merging strategy:")
    for num_frames in [3, 10, 25]:
        images = torch.rand(1, num_frames, 3, 518, 518)
        result = process_with_adaptive_merging(images, model)
        print(f"    Time: {result['latency']['total']*1000:.2f}ms\n")

    print()


if __name__ == "__main__":
    import sys

    examples = {
        '1': example_1_basic_usage,
        '2': example_2_comparison,
        '3': example_3_long_sequences,
        '4': example_4_production_pipeline,
        '5': example_5_adaptive_merging,
    }

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print(f"Available examples: {', '.join(examples.keys())}")
    else:
        print("FastVGGT Examples")
        print("=" * 80)
        print("\nAvailable examples:")
        print("  1 - Basic usage (baseline vs FastVGGT)")
        print("  2 - Compare different merge ratios")
        print("  3 - Long sequence processing")
        print("  4 - Production pipeline")
        print("  5 - Adaptive merging strategy")
        print("\nUsage: python example_usage.py <example_number>")
        print("Example: python example_usage.py 1")
