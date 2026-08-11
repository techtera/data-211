import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Dataset for loading YOLO segmentation annotations and converting them
    into binary masks for semantic segmentation.

    Expected dataset structure:

    root_dir/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    ├── labels/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    │
    ├── classes.txt
    └── notes.json
    """

    def __init__(self, root_dir, image_size=518, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root.
            image_size (int): Output image size.
            transform: Optional Albumentations transform.
        """

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")

        self.image_size = image_size
        self.transform = transform

        self.images = []

        for file in sorted(os.listdir(self.image_dir)):

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            label_name = os.path.splitext(file)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_name)

            if not os.path.exists(label_path):
                raise FileNotFoundError(
                    f"Missing label file for image '{file}'. "
                    f"Expected: {label_path}"
                )

            self.images.append(file)

        print(f"Loaded {len(self.images)} image-label pairs.")

    def __len__(self):
        return len(self.images)

    def _load_polygon(self, label_path):
        """
        Reads a YOLO segmentation label.

        Format:
            class_id x1 y1 x2 y2 ... xn yn

        Returns:
            np.ndarray of shape (N, 2)
        """

        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        with open(label_path, "r") as f:
            line = f.readline().strip()

        if line == "":
            raise ValueError(
                f"Label file is empty: {label_path}"
            )

        values = list(map(float, line.split()))

        # Need at least:
        # class_id + 3 polygon points
        if len(values) < 7:
            raise ValueError(
                f"Invalid polygon in {label_path}. "
                "A polygon must contain at least three vertices."
            )

        coordinates = np.array(values[1:], dtype=np.float32)

        if len(coordinates) % 2 != 0:
            raise ValueError(
                f"Invalid coordinate count in {label_path}."
            )

        polygon = coordinates.reshape(-1, 2)

        return polygon

    def _polygon_to_mask(self, polygon, height, width):
        """
        Converts normalized polygon coordinates
        into a binary segmentation mask.
        """

        if polygon.shape[0] < 3:
            raise ValueError(
                "Polygon must contain at least three vertices."
            )

        polygon = polygon.copy()

        # Convert normalized coordinates to pixel coordinates
        polygon[:, 0] = np.clip(
            polygon[:, 0] * width,
            0,
            width - 1,
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1] * height,
            0,
            height - 1,
        )

        polygon = np.round(polygon).astype(np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)

        cv2.fillPoly(mask, [polygon], 1)

        return mask

    def __getitem__(self, idx):

        # --------------------------------------------------
        # Image & Label Paths
        # --------------------------------------------------

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name,
        )

        label_name = os.path.splitext(image_name)[0] + ".txt"

        label_path = os.path.join(
            self.label_dir,
            label_name,
        )

        # --------------------------------------------------
        # Read Image
        # --------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        height, width = image.shape[:2]

        # --------------------------------------------------
        # Read Polygon
        # --------------------------------------------------

        polygon = self._load_polygon(label_path)

        # --------------------------------------------------
        # Polygon -> Binary Mask
        # --------------------------------------------------

        mask = self._polygon_to_mask(
            polygon,
            height,
            width,
        )

        # --------------------------------------------------
        # Resize Image
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # --------------------------------------------------
        # Resize Mask
        # --------------------------------------------------

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # --------------------------------------------------
        # Apply Transforms (Albumentations)
        # --------------------------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        else:

            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            mask = torch.from_numpy(mask).long()

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return image, mask