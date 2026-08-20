"""
Comprehensive latency profiling for VGGT unified model.

This script measures the execution time of each major component:
- DINOv2 ViT-L backbone (patch embedding, 24 blocks)
- VGGT Aggregator frame blocks (24 blocks)
- VGGT Aggregator global blocks (24 blocks)
- Feature extraction and caching
- Decoder (SegFormer or UNet++)
- Post-processing and upsampling

Usage:
    python profile_latency.py --checkpoint path/to/checkpoint.pt --task cascade --num_warmup 10 --num_iters 100
"""

import argparse
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from model import VGGTUnified


class ProfiledAggregator:
    """Wrapper to profile aggregator components."""

    def __init__(self, aggregator):
        self.aggregator = aggregator
        self.timings = {}

    def __call__(self, images):
        B, S, C_in, H, W = images.shape
        device = images.device

        # Normalize
        images = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        images = images.view(B * S, C_in, H, W)

        # ============================================================
        # 1. DINOv2 Backbone (Patch Embedding)
        # ============================================================
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        patch_tokens = self.aggregator.patch_embed(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        self.timings['dinov2_backbone'] = time.perf_counter() - t0

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Prepare tokens
        camera_token = self._slice_expand_and_flatten(
            self.aggregator.camera_token, B, S
        )
        register_token = self._slice_expand_and_flatten(
            self.aggregator.register_token, B, S
        )
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # Position embeddings
        pos = None
        if self.aggregator.rope is not None:
            pos = self.aggregator.position_getter(
                B * S, H // self.aggregator.patch_size,
                W // self.aggregator.patch_size, device=device
            )
            if self.aggregator.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(
                    B * S, self.aggregator.patch_start_idx, 2
                ).to(device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        # ============================================================
        # 2. VGGT Frame Attention Blocks (24 blocks)
        # ============================================================
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        frame_time = 0.0
        global_time = 0.0

        for block_pair_idx in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_frame = time.perf_counter()

                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    frame_time += time.perf_counter() - t_frame

                elif attn_type == "global":
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_global = time.perf_counter()

                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    global_time += time.perf_counter() - t_global

            # Cache features
            for i in range(len(frame_intermediates)):
                layer_idx = len(output_list)
                if layer_idx in self.aggregator.cached_layer_indices:
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)
                else:
                    output_list.append(None)

        self.timings['vggt_frame_blocks'] = frame_time
        self.timings['vggt_global_blocks'] = global_time
        self.timings['vggt_total'] = frame_time + global_time

        return output_list, self.aggregator.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        for _ in range(self.aggregator.aa_block_size):
            tokens = self.aggregator.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        for _ in range(self.aggregator.aa_block_size):
            tokens = self.aggregator.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    @staticmethod
    def _slice_expand_and_flatten(token_tensor, B, S):
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        combined = torch.cat([query, others], dim=1)
        combined = combined.view(B * S, *combined.shape[2:])
        return combined


class ProfiledVGGTUnified(VGGTUnified):
    """VGGT model with detailed latency profiling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiled_aggregator = ProfiledAggregator(self.aggregator)
        self.timings = {}

    def forward(self, images: torch.Tensor, task: str = "both"):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, S = images.shape[:2]
        device = images.device

        # ============================================================
        # ENCODER (DINOv2 + VGGT Aggregator)
        # ============================================================
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.profiled_aggregator(images)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        encoder_total = time.perf_counter() - t0

        # Copy component timings
        self.timings.update(self.profiled_aggregator.timings)
        self.timings['encoder_total'] = encoder_total

        results = {}

        # ============================================================
        # OBJ DECODER (SegFormer)
        # ============================================================
        if task in ("obj", "both", "cascade"):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            obj_logits = self.obj_decoder(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )

            if obj_logits.dim() == 4:
                C, H, W = obj_logits.shape[1:]
                obj_logits = obj_logits.view(B, S, C, H, W)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.timings['obj_decoder'] = time.perf_counter() - t1
            results['obj_mask'] = obj_logits

        # ============================================================
        # EDGE DECODER (UNet++)
        # ============================================================
        if task in ("edge", "both"):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            edge_probs = self.edge_decoder(aggregated_tokens_list, B, S)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.timings['edge_decoder'] = time.perf_counter() - t2
            results['edge_mask'] = edge_probs

        # ============================================================
        # CASCADE MODE (Obj → ROI → Edge)
        # ============================================================
        if task == "cascade":
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            roi_mask = obj_logits.argmax(dim=2, keepdim=True).float()
            roi_bboxes = self._extract_roi_bboxes(roi_mask)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.timings['roi_extraction'] = time.perf_counter() - t3
            results['roi_bbox'] = roi_bboxes

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            edge_probs = self.edge_decoder(aggregated_tokens_list, B, S)
            edge_probs = edge_probs * roi_mask

            if device.type == 'cuda':
                torch.cuda.synchronize()
            self.timings['edge_decoder_cascade'] = time.perf_counter() - t4
            results['edge_mask'] = edge_probs

        # Total
        self.timings['total'] = sum(
            v for k, v in self.timings.items()
            if not k.endswith('_total')
        )

        results['latency'] = self.timings
        return results


def load_image(path: str, size: int = 518) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(path).convert("RGB").resize((size, size))
    tensor = torch.from_numpy(np.array(img)).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    return tensor


def print_profiling_results(timings_list, task):
    """Print detailed profiling statistics."""

    # Aggregate results
    keys = timings_list[0].keys()
    stats = {}

    for key in keys:
        values = [t[key] * 1000 for t in timings_list]  # Convert to ms
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
        }

    print("\n" + "="*80)
    print(f"LATENCY PROFILING RESULTS ({len(timings_list)} iterations, task={task})")
    print("="*80)

    # Calculate percentages
    total_mean = stats['total']['mean']

    # Component breakdown
    print("\n--- ENCODER COMPONENTS ---")
    print(f"DINOv2 Backbone (24 blocks):")
    print(f"  {stats['dinov2_backbone']['mean']:7.2f} ms ± {stats['dinov2_backbone']['std']:5.2f} ms  "
          f"({stats['dinov2_backbone']['mean']/total_mean*100:5.1f}%)")

    print(f"\nVGGT Frame Blocks (24 blocks):")
    print(f"  {stats['vggt_frame_blocks']['mean']:7.2f} ms ± {stats['vggt_frame_blocks']['std']:5.2f} ms  "
          f"({stats['vggt_frame_blocks']['mean']/total_mean*100:5.1f}%)")

    print(f"\nVGGT Global Blocks (24 blocks):")
    print(f"  {stats['vggt_global_blocks']['mean']:7.2f} ms ± {stats['vggt_global_blocks']['std']:5.2f} ms  "
          f"({stats['vggt_global_blocks']['mean']/total_mean*100:5.1f}%)")

    print(f"\n  VGGT Total (Frame + Global):")
    print(f"  {stats['vggt_total']['mean']:7.2f} ms ± {stats['vggt_total']['std']:5.2f} ms  "
          f"({stats['vggt_total']['mean']/total_mean*100:5.1f}%)")

    print(f"\n  ENCODER TOTAL:")
    print(f"  {stats['encoder_total']['mean']:7.2f} ms ± {stats['encoder_total']['std']:5.2f} ms  "
          f"({stats['encoder_total']['mean']/total_mean*100:5.1f}%)")

    print("\n--- DECODER COMPONENTS ---")
    if 'obj_decoder' in stats:
        print(f"Obj Decoder (SegFormer):")
        print(f"  {stats['obj_decoder']['mean']:7.2f} ms ± {stats['obj_decoder']['std']:5.2f} ms  "
              f"({stats['obj_decoder']['mean']/total_mean*100:5.1f}%)")

    if 'edge_decoder' in stats:
        print(f"Edge Decoder (UNet++):")
        print(f"  {stats['edge_decoder']['mean']:7.2f} ms ± {stats['edge_decoder']['std']:5.2f} ms  "
              f"({stats['edge_decoder']['mean']/total_mean*100:5.1f}%)")

    if 'roi_extraction' in stats:
        print(f"\nROI Extraction:")
        print(f"  {stats['roi_extraction']['mean']:7.2f} ms ± {stats['roi_extraction']['std']:5.2f} ms  "
              f"({stats['roi_extraction']['mean']/total_mean*100:5.1f}%)")

    print("\n" + "-"*80)
    print(f"TOTAL INFERENCE:")
    print(f"  {stats['total']['mean']:7.2f} ms ± {stats['total']['std']:5.2f} ms")
    print(f"  Min: {stats['total']['min']:6.2f} ms  |  Max: {stats['total']['max']:6.2f} ms  "
          f"|  Median: {stats['total']['median']:6.2f} ms")
    print("="*80)

    # Per-block estimates
    print("\n--- PER-BLOCK LATENCY ESTIMATES ---")
    dinov2_per_block = stats['dinov2_backbone']['mean'] / 24
    frame_per_block = stats['vggt_frame_blocks']['mean'] / 24
    global_per_block = stats['vggt_global_blocks']['mean'] / 24

    print(f"DINOv2 per block:     {dinov2_per_block:6.3f} ms")
    print(f"VGGT Frame per block: {frame_per_block:6.3f} ms")
    print(f"VGGT Global per block: {global_per_block:6.3f} ms")
    print(f"VGGT avg per block:   {(frame_per_block + global_per_block)/2:6.3f} ms")

    # Truncation estimates
    print("\n--- TRUNCATION SPEEDUP ESTIMATES ---")
    for blocks_removed in [4, 6, 8]:
        remaining_blocks = 24 - blocks_removed

        # Estimate new times
        new_frame_time = stats['vggt_frame_blocks']['mean'] * (remaining_blocks / 24)
        new_global_time = stats['vggt_global_blocks']['mean'] * (remaining_blocks / 24)
        new_vggt_total = new_frame_time + new_global_time

        new_total = (stats['dinov2_backbone']['mean'] +
                    new_vggt_total +
                    stats.get('obj_decoder', {}).get('mean', 0) +
                    stats.get('edge_decoder', {}).get('mean', 0))

        speedup = (total_mean - new_total) / total_mean * 100

        print(f"\nRemove last {blocks_removed} block pairs (→ {remaining_blocks} pairs):")
        print(f"  New VGGT time: {new_vggt_total:6.2f} ms (from {stats['vggt_total']['mean']:6.2f} ms)")
        print(f"  New total time: {new_total:6.2f} ms (from {total_mean:6.2f} ms)")
        print(f"  Expected speedup: {speedup:5.1f}%")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Profile VGGT unified model latency")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to unified checkpoint")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to test image (uses random tensor if not provided)")
    parser.add_argument("--task", type=str, default="cascade",
                       choices=["obj", "edge", "both", "cascade"])
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--num_warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--num_iters", type=int, default=100,
                       help="Number of profiling iterations")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = ProfiledVGGTUnified(load_encoder=False)
    model.load_unified_checkpoint(args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()

    # Prepare input
    if args.image:
        print(f"Loading image: {args.image}")
        img_tensor = load_image(args.image).unsqueeze(0).unsqueeze(0).to(device)
    else:
        print("Using random tensor input")
        img_tensor = torch.rand(1, 1, 3, 518, 518).to(device)

    # Warmup
    print(f"\nWarming up ({args.num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(img_tensor, task=args.task)

    # Profile
    print(f"\nProfiling ({args.num_iters} iterations)...")
    timings_list = []

    with torch.no_grad():
        for i in range(args.num_iters):
            results = model(img_tensor, task=args.task)
            timings_list.append(results['latency'].copy())

            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{args.num_iters}")

    # Print results
    stats = print_profiling_results(timings_list, args.task)


if __name__ == "__main__":
    main()
