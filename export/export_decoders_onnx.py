"""
Export obj-mask and edge-mask decoders to separate ONNX files.

Each decoder takes pre-computed aggregator tokens as input.

Usage:
    python export/export_decoders_onnx.py --obj_checkpoint checkpoints/obj_mask.pth
    python export/export_decoders_onnx.py --edge_checkpoint checkpoints/edge_mask.pth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from decoders.obj_mask import ObjMaskDecoder
from decoders.edge_mask import EdgeMaskDecoder


class ObjDecoderONNXWrapper(nn.Module):
    """Wraps ObjMaskDecoder to accept flat tensor inputs instead of a list."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.layer_indices = [4, 11, 17, 23]

    def forward(self, layer_4, layer_11, layer_17, layer_23, images):
        # Reconstruct the sparse aggregated_tokens_list
        tokens_list = [None] * 24
        tokens_list[4] = layer_4
        tokens_list[11] = layer_11
        tokens_list[17] = layer_17
        tokens_list[23] = layer_23

        patch_start_idx = 5
        return self.decoder(tokens_list, images=images, patch_start_idx=patch_start_idx)


class EdgeDecoderONNXWrapper(nn.Module):
    """Wraps EdgeMaskDecoder to accept flat tensor inputs."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, layer_4, layer_11, layer_17, layer_23):
        tokens_list = [None] * 24
        tokens_list[4] = layer_4
        tokens_list[11] = layer_11
        tokens_list[17] = layer_17
        tokens_list[23] = layer_23

        B = layer_4.shape[0]
        S = layer_4.shape[1]
        return self.decoder(tokens_list, B, S)


def export_obj_decoder(checkpoint_path, output_path, opset):
    print("Exporting obj-mask decoder...")

    decoder = ObjMaskDecoder(dim_in=2048, output_dim=2)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    decoder_prefix = "depth_head."
    new_state = {k[len(decoder_prefix):]: v for k, v in state_dict.items()
                 if k.startswith(decoder_prefix)}
    decoder.load_state_dict(new_state)
    decoder.eval()

    wrapper = ObjDecoderONNXWrapper(decoder)
    wrapper.eval()

    # Dummy inputs: aggregated tokens at each cached layer [B, S, Tokens, 2048]
    B, S, P = 1, 1, 1374  # 1 + 4 registers + 37*37 patches = 1374
    dummy_tokens = [torch.randn(B, S, P, 2048) for _ in range(4)]
    dummy_images = torch.randn(B, S, 3, 518, 518)

    torch.onnx.export(
        wrapper,
        (*dummy_tokens, dummy_images),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["layer_4", "layer_11", "layer_17", "layer_23", "images"],
        output_names=["mask_logits"],
        dynamic_axes={
            "layer_4": {0: "batch", 1: "seq_len"},
            "layer_11": {0: "batch", 1: "seq_len"},
            "layer_17": {0: "batch", 1: "seq_len"},
            "layer_23": {0: "batch", 1: "seq_len"},
            "images": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"  Saved: {output_path}")


def export_edge_decoder(checkpoint_path, output_path, opset):
    print("Exporting edge-mask decoder...")

    decoder = EdgeMaskDecoder()

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        decoder.load_state_dict(state_dict, strict=False)

    decoder.eval()

    wrapper = EdgeDecoderONNXWrapper(decoder)
    wrapper.eval()

    B, S, P = 1, 1, 1374
    dummy_tokens = [torch.randn(B, S, P, 2048) for _ in range(4)]

    torch.onnx.export(
        wrapper,
        tuple(dummy_tokens),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["layer_4", "layer_11", "layer_17", "layer_23"],
        output_names=["edge_mask"],
        dynamic_axes={
            "layer_4": {0: "batch", 1: "seq_len"},
            "layer_11": {0: "batch", 1: "seq_len"},
            "layer_17": {0: "batch", 1: "seq_len"},
            "layer_23": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export decoders to ONNX")
    parser.add_argument("--obj_checkpoint", type=str, default=None)
    parser.add_argument("--edge_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.obj_checkpoint:
        export_obj_decoder(args.obj_checkpoint, str(output_dir / "obj_decoder.onnx"), args.opset)

    if args.edge_checkpoint:
        export_edge_decoder(args.edge_checkpoint, str(output_dir / "edge_decoder.onnx"), args.opset)

    if not args.obj_checkpoint and not args.edge_checkpoint:
        print("Provide --obj_checkpoint and/or --edge_checkpoint")


if __name__ == "__main__":
    main()
