"""
Inference script for VGGT + SegFormer.
"""

import os

import cv2
import torch
import numpy as np
from PIL import Image

from fine_tuning.config import DEVICE
from fine_tuning.model_builder import build_model


# ============================================================
# Config
# ============================================================

IMAGE_PATH = "/Users/dikshitrishi/Terafac/IMG_2988.jpg"

CHECKPOINT_PATH = "checkpoints/best_model.pth"

IMAGE_SIZE = 518

OUTPUT_DIR = "inference_outputs"


# ============================================================
# Load Image
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = np.array(image)

    resized = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor, original


# ============================================================
# Main
# ============================================================

def main():

    print("\nLoading model...")

    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("✓ Checkpoint loaded")

    image_tensor, original_image = load_image(
        IMAGE_PATH
    )

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(1)

    image_tensor = image_tensor.to(DEVICE)

    print(
        f"Input Shape: {image_tensor.shape}"
    )

    with torch.no_grad():

        outputs = model(image_tensor)

        logits = outputs["mask_logits"]

        prediction = torch.argmax(
            logits,
            dim=1,
        )

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    mask = cv2.resize(
        mask,
        (
            original_image.shape[1],
            original_image.shape[0],
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Binary Mask
    # --------------------------------------------------------

    mask_visual = (
        mask * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = original_image.copy()

    overlay[mask == 1] = [
        255,
        0,
        0,
    ]

    blended = cv2.addWeighted(
        original_image,
        0.7,
        overlay,
        0.3,
        0,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        f"{OUTPUT_DIR}/mask.png",
        mask_visual,
    )

    cv2.imwrite(
        f"{OUTPUT_DIR}/overlay.png",
        cv2.cvtColor(
            blended,
            cv2.COLOR_RGB2BGR,
        ),
    )

    print("\n✓ Inference complete")
    print(
        f"✓ Results saved to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()