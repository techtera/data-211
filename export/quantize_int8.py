"""
INT8 post-training quantization for the ONNX encoder using ONNX Runtime.

Reads calibration images, runs them through the encoder to collect activation
statistics, then produces a quantized INT8 .onnx file.

Usage:
    python export/quantize_int8.py \
        --encoder_onnx checkpoints/encoder.onnx \
        --calibration_dir /path/to/calibration_images/ \
        --output checkpoints/encoder_int8.onnx \
        --num_samples 100
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


IMAGE_SIZE = 518


class EncoderCalibrationDataReader(CalibrationDataReader):
    """Feeds calibration images to the encoder for activation range collection."""

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
        print(f"Calibration: {len(self.image_paths)} images from {calibration_dir}")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        # [B=1, S=1, 3, 518, 518]
        arr = arr[np.newaxis, np.newaxis, ...]

        return {"images": arr}

    def rewind(self):
        self.index = 0


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize ONNX encoder")
    parser.add_argument("--encoder_onnx", type=str, default="onnx_models/encoder.onnx",
                        help="Path to FP32/FP16 encoder ONNX model")
    parser.add_argument("--calibration_dir", type=str, required=True,
                        help="Directory of calibration images")
    parser.add_argument("--output", type=str, default="onnx_models/encoder_int8.onnx",
                        help="Output path for INT8 model")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of calibration images to use")
    parser.add_argument("--per_channel", action="store_true", default=True,
                        help="Per-channel quantization (better accuracy, default)")
    parser.add_argument("--calibration_method", type=str, default="entropy",
                        choices=["entropy", "minmax", "percentile"],
                        help="Calibration method for range estimation")
    args = parser.parse_args()

    calibration_methods = {
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
    }

    print(f"Input model: {args.encoder_onnx}")
    print(f"Output model: {args.output}")
    print(f"Calibration method: {args.calibration_method}")
    print(f"Per-channel: {args.per_channel}")

    calibration_reader = EncoderCalibrationDataReader(
        args.calibration_dir, num_samples=args.num_samples
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=args.encoder_onnx,
        model_output=args.output,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=calibration_methods[args.calibration_method],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    input_size = Path(args.encoder_onnx).stat().st_size / (1024 ** 2)
    output_size = Path(args.output).stat().st_size / (1024 ** 2)
    print(f"\nDone! {input_size:.0f} MB → {output_size:.0f} MB ({output_size/input_size:.1%})")


if __name__ == "__main__":
    main()
