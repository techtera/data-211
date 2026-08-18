"""
Export the VGGT Aggregator (shared encoder) to ONNX format.

Outputs 4 named tensors corresponding to cached layers [4, 11, 17, 23].
These are consumed by both the obj-mask and edge-mask decoders.

Usage:
    python export/export_encoder_onnx.py --checkpoint path/to/model.pth --output encoder.onnx
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from encoder import Aggregator


def main():
    parser = argparse.ArgumentParser(description="Export VGGT Aggregator to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to full model checkpoint (extracts aggregator weights)")
    parser.add_argument("--output", type=str, default="checkpoints/encoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    print("Building Aggregator...")
    aggregator = Aggregator(img_size=518, patch_size=14, embed_dim=1024)

    print(f"Loading weights from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    agg_state = {}
    for k, v in state_dict.items():
        if k.startswith("aggregator."):
            agg_state[k.replace("aggregator.", "")] = v

    aggregator.load_state_dict(agg_state)
    aggregator.eval()

    if hasattr(aggregator, "_resnet_mean"):
        aggregator._resnet_mean = aggregator._resnet_mean.cpu()
    if hasattr(aggregator, "_resnet_std"):
        aggregator._resnet_std = aggregator._resnet_std.cpu()

    if args.fp16:
        aggregator = aggregator.half()
        dummy = torch.randn(1, 1, 3, 518, 518, dtype=torch.float16)
    else:
        dummy = torch.randn(1, 1, 3, 518, 518)

    print(f"Exporting to {args.output} (opset {args.opset})...")

    # The aggregator returns (list[Tensor|None], int).
    # We need a wrapper to flatten outputs for ONNX.

    class AggregatorONNXWrapper(torch.nn.Module):
        def __init__(self, aggregator):
            super().__init__()
            self.aggregator = aggregator
            self.cached_indices = sorted(aggregator.cached_layer_indices)

        def forward(self, images):
            output_list, _ = self.aggregator(images)
            # Return only non-None cached layers
            results = []
            for idx in self.cached_indices:
                if output_list[idx] is not None:
                    results.append(output_list[idx])
            return tuple(results)

    wrapper = AggregatorONNXWrapper(aggregator)
    wrapper.eval()

    output_names = [f"layer_{idx}_tokens" for idx in sorted(aggregator.cached_layer_indices)]

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes={
            "images": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 1: "seq_len"} for name in output_names},
        },
    )

    print(f"ONNX export complete: {args.output}")
    print(f"Output names: {output_names}")


if __name__ == "__main__":
    main()
