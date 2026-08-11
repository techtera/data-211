import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from fine_tuning.config import DEVICE
from fine_tuning.model_builder import build_model

IMAGE_DIR = "test_images_2"
CHECKPOINT_PATH = "checkpoints/best_model.pth"
IMAGE_SIZE = 518
OUTPUT_DIR = "inference_outputs_2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")
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


print("\nLoading model...")

model = build_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("✓ Checkpoint loaded")

valid_exts = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_exts)
]

print(f"\nFound {len(image_files)} images")

with torch.no_grad():

    for image_name in tqdm(image_files):

        image_path = os.path.join(
            IMAGE_DIR,
            image_name,
        )

        image_tensor, original_image = load_image(
            image_path
        )

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(1)

        image_tensor = image_tensor.to(DEVICE)

        outputs = model(image_tensor)

        logits = outputs["mask_logits"]

        logits = torch.nn.functional.interpolate(
            logits,
            size=(
                original_image.shape[0],
                original_image.shape[1],
            ),
            mode="bilinear",
            align_corners=False,
        )

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

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                image_name,
            ),
            cv2.cvtColor(
                blended,
                cv2.COLOR_RGB2BGR,
            ),
        )

print(f"\n✓ Saved {len(image_files)} overlay images to {OUTPUT_DIR}")