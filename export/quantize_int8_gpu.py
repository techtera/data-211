"""
INT8 static quantization using GPU for calibration.

Uses ONNX Runtime's lower-level calibration API with CUDAExecutionProvider
so calibration inference runs on GPU (avoids CPU RAM OOM).

Usage:
    python export/quantize_int8_gpu.py \
        --encoder_onnx onnx_models/encoder.onnx \
        --calibration_dir rgb_reg/ \
        --output onnx_models/encoder_int8.onnx
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)
from onnxruntime.quantization.calibrate import create_calibrator


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
        print(f"Calibration: {len(self.image_paths)} images from {calibration_dir}")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        self.index += 1

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
    parser.add_argument("--calibration_method", type=str, default="minmax",
                        choices=["entropy", "minmax", "percentile"])
    args = parser.parse_args()

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("ERROR: CUDAExecutionProvider not available.")
        print("Install onnxruntime-gpu: pip install onnxruntime-gpu")
        sys.exit(1)

    calibration_methods = {
        "entropy": CalibrationMethod.Entropy,
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
    }

    print(f"Input model: {args.encoder_onnx}")
    print(f"Output model: {args.output}")
    print(f"Calibration method: {args.calibration_method}")

    # Step 1: Calibrate on GPU (skip preprocessing — protobuf 2GB limit on large models)
    print(f"\n[1/2] Running calibration on GPU ({args.num_samples} images)...")
    calibrator = create_calibrator(
        model=args.encoder_onnx,
        op_types_to_calibrate=None,
        augmented_model_path=str(Path(args.encoder_onnx).with_suffix(".augmented.onnx")),
        calibrate_method=calibration_methods[args.calibration_method],
        use_external_data_format=True,
        extra_options={
            "symmetric": True,
            "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
    )

    data_reader = EncoderCalibrationDataReader(
        args.calibration_dir, num_samples=args.num_samples
    )
    calibrator.collect_data(data_reader)
    tensors_range = calibrator.compute_data()
    print(f"  Collected ranges for {len(tensors_range)} tensors")

    # Step 2: Quantize with pre-computed ranges
    print(f"\n[2/2] Applying INT8 quantization...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
    from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry
    import onnx

    model = onnx.load(args.encoder_onnx)

    quantizer = ONNXQuantizer(
        model=model,
        per_channel=True,
        reduce_range=False,
        mode=QuantFormat.QDQ,
        static=True,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QInt8,
        tensors_range=tensors_range,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=list(QDQRegistry.keys()),
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(args.output, use_external_data_format=True)

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
