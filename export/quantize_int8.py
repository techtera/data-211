"""
INT8 quantization for the ONNX encoder using ONNX Runtime.

Supports two modes:
  --mode dynamic  (default) — quantizes weights only, no calibration needed, no OOM risk
  --mode static             — quantizes weights + activations, needs calibration images + GPU memory

Dynamic quantization is recommended for initial size reduction. For maximum INT8
performance on Orin NX, use TensorRT INT8 calibration on-device.

Usage:
    # Dynamic (no calibration needed):
    python export/quantize_int8.py --encoder_onnx onnx_models/encoder.onnx

    # Static (needs calibration images + enough RAM/VRAM):
    python export/quantize_int8.py --mode static --calibration_dir rgb_reg/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


IMAGE_SIZE = 518


class EncoderCalibrationDataReader(CalibrationDataReader):
    """Feeds calibration images to the encoder for activation range collection."""

    def __init__(self, calibration_dir: str, num_samples: int = 100, use_fp16: bool = False):
        self.image_paths = sorted(Path(calibration_dir).glob("*"))
        self.image_paths = [
            p for p in self.image_paths
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {calibration_dir}")

        self.image_paths = self.image_paths[:num_samples]
        self.use_fp16 = use_fp16
        self.index = 0
        print(f"Calibration: {len(self.image_paths)} images from {calibration_dir} (fp16={use_fp16})")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

        img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        arr = arr[np.newaxis, np.newaxis, ...]

        if self.use_fp16:
            arr = arr.astype(np.float16)

        return {"images": arr}

    def rewind(self):
        self.index = 0


def run_dynamic(args):
    """Dynamic quantization — weights only, no calibration, no OOM."""
    print("Mode: DYNAMIC (weight-only quantization)")
    print(f"Input model: {args.encoder_onnx}")
    print(f"Output model: {args.output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=args.encoder_onnx,
        model_output=args.output,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
        extra_options={
            "MatMulConstBOnly": True,
        },
    )

    print_size_comparison(args.encoder_onnx, args.output)


def run_static(args):
    """Static quantization — weights + activations, needs calibration."""
    import onnx

    print("Mode: STATIC (weights + activations)")
    print(f"Input model: {args.encoder_onnx}")
    print(f"Output model: {args.output}")
    print(f"Calibration method: {args.calibration_method}")

    if not args.calibration_dir:
        print("ERROR: --calibration_dir required for static mode")
        sys.exit(1)

    calibration_methods = {
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
    }

    model = onnx.load(args.encoder_onnx, load_external_data=False)
    input_type = model.graph.input[0].type.tensor_type.elem_type
    use_fp16 = (input_type == onnx.TensorProto.FLOAT16)
    del model
    print(f"Model input dtype: {'float16' if use_fp16 else 'float32'}")

    preprocessed_path = str(Path(args.encoder_onnx).with_suffix(".preprocessed.onnx"))
    print("Preprocessing model (shape inference + optimization)...")
    quant_pre_process(args.encoder_onnx, preprocessed_path, auto_merge=True)

    calibration_reader = EncoderCalibrationDataReader(
        args.calibration_dir, num_samples=args.num_samples, use_fp16=use_fp16
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=preprocessed_path,
        model_output=args.output,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=calibration_methods[args.calibration_method],
        use_external_data_format=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    print_size_comparison(args.encoder_onnx, args.output)


def print_size_comparison(input_path, output_path):
    def total_size(path):
        parent = Path(path).parent
        stem = Path(path).stem
        return sum(
            f.stat().st_size for f in parent.iterdir()
            if f.name.startswith(stem)
        ) / (1024 ** 2)

    input_mb = total_size(input_path)
    output_mb = total_size(output_path)
    print(f"\nDone! {input_mb:.0f} MB → {output_mb:.0f} MB ({output_mb/input_mb:.1%})")


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize ONNX encoder")
    parser.add_argument("--encoder_onnx", type=str, default="onnx_models/encoder.onnx")
    parser.add_argument("--output", type=str, default="onnx_models/encoder_int8.onnx")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--calibration_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--calibration_method", type=str, default="entropy",
                        choices=["entropy", "minmax", "percentile"])
    args = parser.parse_args()

    if args.mode == "dynamic":
        run_dynamic(args)
    else:
        run_static(args)


if __name__ == "__main__":
    main()
