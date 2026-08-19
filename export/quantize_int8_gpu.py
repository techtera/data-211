"""
INT8 static quantization with GPU-based calibration.

Usage:
    python export/quantize_int8_gpu.py --calibration_dir rgb_reg/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import onnxruntime as ort

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


IMAGE_SIZE = 518


class EncoderCalibrationDataReader(CalibrationDataReader):

    def __init__(self, calibration_dir: str, num_samples: int = 100):
        self.image_paths = sorted(Path(calibration_dir).glob("*"))
        self.image_paths = [
            p for p in self.image_paths
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {calibration_dir}")

        self.image_paths = self.image_paths[:num_samples]
        self.index = 0
        self.total = len(self.image_paths)
        print(f"Calibration: {self.total} images from {calibration_dir}")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

        if self.index % 10 == 0 or self.index == 1:
            print(f"  [{self.index}/{self.total}] {img_path.name}")

        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = arr[np.newaxis, np.newaxis, ...]
        return {"images": arr}

    def rewind(self):
        self.index = 0


def main():
    parser = argparse.ArgumentParser(description="INT8 static quantization (GPU calibration)")
    parser.add_argument("--encoder_onnx", type=str, default="onnx_models/encoder.onnx")
    parser.add_argument("--calibration_dir", type=str, default="rgb_reg/")
    parser.add_argument("--output", type=str, default="onnx_models/encoder_int8.onnx")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("ERROR: CUDAExecutionProvider not available.")
        print("Install: pip install onnxruntime-gpu")
        sys.exit(1)

    print(f"Input:  {args.encoder_onnx}")
    print(f"Output: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    data_reader = EncoderCalibrationDataReader(
        args.calibration_dir, num_samples=args.num_samples
    )

    print("\nRunning quantize_static with CUDAExecutionProvider...")
    quantize_static(
        model_input=args.encoder_onnx,
        model_output=args.output,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True,
        calibration_providers=[
            ("CUDAExecutionProvider", {"device_id": 1}),
            "CPUExecutionProvider",
        ],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    # Size comparison
    def total_size(path):
        parent = Path(path).parent
        stem = Path(path).stem
        return sum(
            f.stat().st_size for f in parent.iterdir()
            if f.name.startswith(stem)
        ) / (1024 ** 2)

    input_mb = total_size(args.encoder_onnx)
    output_mb = total_size(args.output)
    print(f"\nDone! {input_mb:.0f} MB → {output_mb:.0f} MB ({output_mb/input_mb:.1%})")


if __name__ == "__main__":
    main()
