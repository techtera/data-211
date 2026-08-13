"""
Inference script for VGGT + UNet++ Edge Mask.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --checkpoint checkpoints/best_model.pt
    python inference.py --image_dir path/to/images/ --output_dir results/
"""

import sys
sys.path.insert(0, ".")

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from vggt.models.vggt import VGGT
from edge_mask.model import VGGTEdgeMask
from fine_tuning.config import PRETRAINED_MODEL, IMAGE_SIZE


# ============================================================
# Load Model
# ============================================================

def load_model(checkpoint_path, device):
    """
    Load VGGT encoder + edge mask decoder from checkpoint.
    """

    print(f"Loading VGGT from {PRETRAINED_MODEL}...")
    vggt_model = VGGT.from_pretrained(PRETRAINED_MODEL)

    model = VGGTEdgeMask(vggt_model.aggregator)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", "?")
    print(f"Checkpoint epoch: {epoch}, loss: {loss}")

    return model


# ============================================================
# Preprocess
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess a single image to [1, 1, 3, 518, 518].
    """

    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    tensor = transform(img)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


# ============================================================
# Predict
# ============================================================

@torch.no_grad()
def predict(model, image_tensor, device, threshold=0.5):
    """
    Run inference on a single image tensor.

    Returns binary mask at 1280x720.
    """

    image_tensor = image_tensor.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        prob_map = model(image_tensor)

    prob_map = F.interpolate(
        prob_map.view(1, 1, 518, 518),
        size=(720, 1280),
        mode="bilinear",
        align_corners=False,
    )

    prob_map = prob_map.squeeze().cpu()
    binary_mask = (prob_map >= threshold).float()

    return binary_mask.numpy()


# ============================================================
# Save Output
# ============================================================

def save_output(binary_mask, output_path):
    """
    Save binary mask as a 1280x720 image.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    mask_img.save(output_path)

    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="VGGT Edge Mask Inference"
    )

    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single input image",
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Directory of input images",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binarization threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect)",
    )

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image_dir")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    # Collect input images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    print(f"\nRunning inference on {len(image_paths)} image(s)...\n")

    output_dir = Path(args.output_dir)

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")

        image_tensor = preprocess_image(img_path)
        binary_mask = predict(
            model, image_tensor, device, args.threshold
        )

        output_path = output_dir / f"{img_path.stem}_mask.png"
        save_output(binary_mask, output_path)

    print(f"\nDone. {len(image_paths)} image(s) processed.")


if __name__ == "__main__":
    main()
