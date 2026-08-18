# VGGT Unified — Multi-Task Inference Pipeline

Shared VGGT-1B encoder with two task-specific decoders (object segmentation + edge detection) in a single unified model.

## Files Required for Inference

Place the following checkpoint files in the `checkpoints/` directory before running inference:

| File | Description | Size |
|------|-------------|------|
| `checkpoints/obj_mask.pth` | Object segmentation decoder weights | ~4.4 GB |
| `checkpoints/edge_mask.pt` | Edge detection decoder weights | ~3.5 GB |
| `checkpoints/vggt_unified.pt` | Unified checkpoint (encoder + both decoders) | ~3.5 GB |
| `checkpoints/vggt_unified_fp16.pt` | Unified checkpoint in FP16 (smaller, faster) | ~1.7 GB |

> **Note:** You only need **one** of the following options:
> - Option A: `vggt_unified.pt` or `vggt_unified_fp16.pt` (single file, includes everything)
> - Option B: `obj_mask.pth` + `edge_mask.pt` (separate decoders; encoder downloads from HuggingFace automatically)

For ONNX inference, you also need:

| File | Description |
|------|-------------|
| `checkpoints/encoder.onnx` | Exported encoder in ONNX format |
| `checkpoints/obj_decoder.onnx` | Exported obj-mask decoder in ONNX format |
| `checkpoints/edge_decoder.onnx` | Exported edge-mask decoder in ONNX format |

## Setup

```bash
pip install torch torchvision numpy opencv-python pillow scikit-image huggingface_hub
```

For ONNX inference:
```bash
pip install onnxruntime-gpu  # or onnxruntime for CPU-only
```

## Inference Steps

### PyTorch Inference (GPU)

**Using the unified checkpoint (recommended):**
```bash
python inference.py --image path/to/image.png --task cascade \
    --unified_checkpoint checkpoints/vggt_unified_fp16.pt
```

**Using separate decoder checkpoints:**
```bash
python inference.py --image path/to/image.png --task cascade \
    --obj_checkpoint checkpoints/obj_mask.pth \
    --edge_checkpoint checkpoints/edge_mask.pt
```

**Batch inference on a directory:**
```bash
python inference.py --image_dir path/to/images/ --task cascade \
    --unified_checkpoint checkpoints/vggt_unified_fp16.pt \
    --save_dir output/
```

### Available Tasks

| Task | Description |
|------|-------------|
| `obj` | Object segmentation only |
| `edge` | Edge detection only |
| `both` | Both tasks independently |
| `cascade` | Object segmentation → ROI extraction → Edge detection within ROI |

### ONNX Inference (GPU / Jetson / CPU)

First export the models to ONNX:
```bash
python export/export_encoder_onnx.py --checkpoint checkpoints/vggt_unified.pt --output checkpoints/encoder.onnx
python export/export_decoders_onnx.py --checkpoint checkpoints/vggt_unified.pt
```

Then run inference:
```bash
python inference_onnx.py --image path/to/image.png --task both \
    --encoder_onnx checkpoints/encoder.onnx \
    --obj_onnx checkpoints/obj_decoder.onnx \
    --edge_onnx checkpoints/edge_decoder.onnx
```

### Creating the Unified Checkpoint

If you have the separate decoder checkpoints and want to create the unified one:
```bash
python save_unified_checkpoint.py \
    --obj_checkpoint checkpoints/obj_mask.pth \
    --edge_checkpoint checkpoints/edge_mask.pt \
    --output checkpoints/vggt_unified.pt

# FP16 variant (half the size):
python save_unified_checkpoint.py \
    --obj_checkpoint checkpoints/obj_mask.pth \
    --edge_checkpoint checkpoints/edge_mask.pt \
    --output checkpoints/vggt_unified_fp16.pt --fp16
```

## Project Structure

```
vggt-unified/
├── model.py                        # VGGTUnified model class
├── inference.py                    # PyTorch inference script
├── inference_onnx.py               # ONNX Runtime inference script
├── save_unified_checkpoint.py      # Combine encoder+decoders into one file
├── encoder/                        # VGGT-1B Aggregator (shared encoder)
│   ├── aggregator.py
│   ├── layers/                     # ViT layers (attention, MLP, patch embed, RoPE)
│   └── utils/                      # Geometry, pose encoding, helpers
├── decoders/
│   ├── obj_mask/                   # SegFormer-based object segmentation decoder
│   └── edge_mask/                  # UNet++-based edge detection decoder
└── export/
    ├── export_encoder_onnx.py      # Export encoder to ONNX
    └── export_decoders_onnx.py     # Export decoders to ONNX
```

## Output

- **Object mask:** `[B, S, 2, 518, 518]` logits (argmax for binary mask)
- **Edge mask:** `[B, S, 1, 518, 518]` sigmoid probabilities
- In `cascade` mode, edge detection runs only within the detected object ROI
