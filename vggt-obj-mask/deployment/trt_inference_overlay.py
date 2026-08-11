import os
import glob
import cv2
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

ENGINE_PATH = "deployment/onnx/vggt_segformer_clean_fp16.engine"
INPUT_DIR = "rgb_images"
OUTPUT_DIR = "deployment/trt_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Engine
# -------------------------

logger = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape = tuple(engine.get_tensor_shape(input_name))
output_shape = tuple(engine.get_tensor_shape(output_name))

print("Input :", input_name, input_shape)
print("Output:", output_name, output_shape)

# -------------------------
# Allocate Buffers
# -------------------------

input_size = int(np.prod(input_shape))
output_size = int(np.prod(output_shape))

d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)

stream = cuda.Stream()

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

# -------------------------
# Images
# -------------------------

image_paths = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.png"))
)

print(f"\nFound {len(image_paths)} images\n")

for idx, img_path in enumerate(image_paths):

    print(f"[{idx+1}/{len(image_paths)}] {os.path.basename(img_path)}")

    image = cv2.imread(img_path)

    if image is None:
        print("Failed:", img_path)
        continue

    original_h, original_w = image.shape[:2]

    # -------------------------
    # Preprocess
    # -------------------------

    image_resized = cv2.resize(
        image,
        (518, 518)
    )

    image_rgb = cv2.cvtColor(
        image_resized,
        cv2.COLOR_BGR2RGB
    )

    image_rgb = image_rgb.astype(np.float32) / 255.0

    inp = image_rgb.transpose(2, 0, 1)
    inp = np.expand_dims(inp, axis=0)
    inp = np.expand_dims(inp, axis=0)

    inp = np.ascontiguousarray(inp)

    # -------------------------
    # Inference
    # -------------------------

    cuda.memcpy_htod_async(
        d_input,
        inp,
        stream
    )

    context.execute_async_v3(
        stream_handle=stream.handle
    )

    output = np.empty(
        output_shape,
        dtype=np.float32
    )

    cuda.memcpy_dtoh_async(
        output,
        d_output,
        stream
    )

    stream.synchronize()

    # -------------------------
    # Postprocess
    # -------------------------

    logits = output[0]

    mask = np.argmax(
        logits,
        axis=0
    ).astype(np.uint8)

    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    # -------------------------
    # Overlay
    # -------------------------

    overlay = image.copy()

    overlay[mask == 1] = (0, 255, 0)

    result = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    out_name = (
        os.path.splitext(
            os.path.basename(img_path)
        )[0]
        + "_overlay.png"
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        out_name
    )

    cv2.imwrite(
        out_path,
        result
    )

print("\nDone.")
print(f"Saved overlays to: {OUTPUT_DIR}")
