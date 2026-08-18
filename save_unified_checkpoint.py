"""
Save the full VGGT unified pipeline (encoder + both decoders) as a single checkpoint.

Usage:
    python save_unified_checkpoint.py
    python save_unified_checkpoint.py --output checkpoints/vggt_unified.pt
    python save_unified_checkpoint.py --fp16   # save in half precision
"""

import argparse
from pathlib import Path

import torch

from model import VGGTUnified


def main():
    parser = argparse.ArgumentParser(description="Save unified VGGT checkpoint")
    parser.add_argument("--obj_checkpoint", type=str, default="checkpoints/obj_mask.pth")
    parser.add_argument("--edge_checkpoint", type=str, default="checkpoints/edge_mask.pt")
    parser.add_argument("--output", type=str, default="checkpoints/vggt_unified.pt")
    parser.add_argument("--fp16", action="store_true", help="Save weights in FP16")
    args = parser.parse_args()

    print("Building unified model...")
    model = VGGTUnified(load_encoder=True)

    print(f"Loading obj decoder from {args.obj_checkpoint}")
    model.load_decoder_checkpoint("obj", args.obj_checkpoint)

    print(f"Loading edge decoder from {args.edge_checkpoint}")
    model.load_decoder_checkpoint("edge", args.edge_checkpoint)

    model.eval()

    state_dict = model.state_dict()

    if args.fp16:
        print("Converting to FP16...")
        state_dict = {k: v.half() for k, v in state_dict.items()}

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": state_dict,
        "config": {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "obj_dim_in": 2048,
            "obj_output_dim": 2,
            "fp16": args.fp16,
        },
    }

    torch.save(checkpoint, str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved unified checkpoint: {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
