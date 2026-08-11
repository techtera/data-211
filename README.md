# VGGT + SegFormer Object Masking Model

## Overview

This project focuses on adapting the Visual Geometry Grounded Transformer (VGGT) architecture for binary object masking. The original VGGT model is primarily designed for 3D vision tasks such as camera pose estimation, depth prediction, point prediction, and tracking. In this project, the original segmentation/depth branch was replaced with a SegFormer-based decoder and fine-tuned for object mask generation.

The final model predicts a binary segmentation mask where:

- Background = 0 (Black)
- Object Region = 1 (White)

The project covers:

- Dataset preparation and validation
- VGGT architecture understanding
- SegFormer decoder integration
- Fine-tuning pipeline development
- Training and evaluation
- ONNX export
- TensorRT FP16 optimization
- High-performance deployment inference

---

# Problem Statement

The objective was to generate accurate object masks from RGB images using the VGGT encoder while leveraging the lightweight and efficient SegFormer decoder for segmentation.

The goals were:

1. Reuse the powerful VGGT feature extractor.
2. Replace the original DPT segmentation branch.
3. Fine-tune the architecture on a custom object masking dataset.
4. Deploy an optimized inference pipeline for production usage.

---

# Architecture

## Original VGGT

Input Image

↓

Patch Embedding

↓

Transformer Encoder

↓

Feature Aggregation

↓

DPT Head

↓

Depth / Segmentation Output

---

## Modified VGGT

Input Image

↓

Patch Embedding

↓

VGGT Transformer Encoder

↓

Feature Aggregation

↓

SegFormer Decoder

↓

2-Class Mask Prediction

↓

Object Mask

---

# Model Components

## Encoder

VGGT Aggregator

The encoder is responsible for:

- Image patch extraction
- Positional encoding
- Multi-scale transformer processing
- Feature aggregation

Input Size:

```python
(1, 1, 3, 518, 518)
```

Output:

Multi-scale transformer feature representations.

---

## Decoder

SegFormer Decoder

The decoder receives hierarchical features from the VGGT encoder and produces dense segmentation predictions.

Configuration:

```python
output_dim = 2
```

Classes:

```python
0 → Background
1 → Object
```

Output Shape:

```python
(1, 2, 518, 518)
```

---

# Dataset Pipeline

## Dataset Preparation

The dataset consists of:

```text
images/
masks/
```

Each image has a corresponding binary mask.

Mask Format:

```text
0 = Background
255 = Object
```

Converted during training to:

```text
0 = Background
1 = Object
```

---

## Preprocessing

Input images:

- Resize to 518 × 518
- Normalization using VGGT preprocessing
- Conversion to tensor format

Mask preprocessing:

- Resize
- Binary conversion
- Long tensor conversion

---

# Training Pipeline

## Loss Function

Cross Entropy Loss

```python
nn.CrossEntropyLoss()
```

Used because the task is binary semantic segmentation.

---

## Optimizer

```python
AdamW
```

Benefits:

- Stable convergence
- Better transformer training behavior
- Improved generalization

---

## Learning Strategy

Fine-tuning approach:

1. Load pretrained VGGT weights.
2. Replace DPT decoder.
3. Attach SegFormer decoder.
4. Train segmentation branch.
5. Fine-tune complete network.

---

# Implementation Details

## Custom Model Builder

Responsibilities:

- Build VGGT encoder
- Attach SegFormer decoder
- Load pretrained weights
- Restore checkpoints

---

## Forward Pass

Input:

```python
(1,1,3,518,518)
```

Flow:

```python
Image
 ↓
VGGT Aggregator
 ↓
Multi-scale Features
 ↓
SegFormer Decoder
 ↓
Mask Logits
```

Output Dictionary:

```python
{
    "pose_enc",
    "pose_enc_list",
    "mask_logits",
    "images"
}
```

Mask Output:

```python
predictions["mask_logits"]
```

Shape:

```python
(1,2,518,518)
```

---

# Training Results

The model successfully learned object boundaries and object regions from the custom dataset.

Observed improvements:

- Stable training convergence
- Accurate object localization
- Clean segmentation boundaries
- Robust mask generation on unseen samples

---

# Inference Pipeline

## PyTorch Inference

Workflow:

```text
Image
 ↓
VGGT Encoder
 ↓
SegFormer Decoder
 ↓
Mask Logits
 ↓
Argmax
 ↓
Binary Mask
```

Generated Outputs:

```text
_mask.png
```

---

## Overlay Visualization

For qualitative evaluation:

```text
Original Image
+
Predicted Mask
=
Overlay Image
```

Generated Outputs:

```text
_overlay.png
```

This allowed easy visual inspection of segmentation quality.

---

# ONNX Export

The trained PyTorch model was exported to ONNX.

Export Settings:

```python
opset_version = 17
```

Output:

```text
vggt_segformer_clean.onnx
```

Purpose:

- Hardware-independent deployment
- TensorRT conversion
- Runtime optimization

---

# TensorRT Optimization

## Motivation

Although PyTorch inference was functional, deployment required:

- Lower latency
- Better throughput
- Production-ready execution

TensorRT was used to optimize the model graph.

---

## Conversion Pipeline

PyTorch Checkpoint

↓

ONNX

↓

TensorRT FP16 Engine

↓

Optimized Deployment

---

## Final Engine

Generated:

```text
vggt_segformer_clean_fp16.engine
```

TensorRT Version:

```text
8.6.1
```

Precision:

```text
FP16
```

---

# TensorRT Engine Validation

Verified Engine IO:

Input:

```python
input
(1,1,3,518,518)
```

Output:

```python
mask_logits
(1,2,518,518)
```

Successfully matched PyTorch output structure.

---

# Performance Results

Hardware:

```text
NVIDIA A100 80GB
```

TensorRT FP16 Results:

| Metric | Value |
|----------|----------|
| Latency | ~29 ms |
| Throughput | ~34.6 FPS |
| Precision | FP16 |
| Input Resolution | 518 × 518 |

Performance Summary:

```text
Mean Latency:
≈ 29.1 ms

Throughput:
≈ 34.6 FPS
```

The optimized TensorRT engine provided a production-ready deployment path while maintaining segmentation quality.

---

# Project Structure

```text
vggt/
│
├── fine_tuning/
│   ├── dataset.py
│   ├── model_builder.py
│   ├── trainer.py
│   └── utils.py
│
├── vggt/
│   ├── models/
│   │   └── vggt_modifying.py
│   │
│   ├── heads/
│   │   └── segformer_head_for_dpt.py
│   │
│   └── layers/
│       └── rope.py
│
├── deployment/
│   ├── validate_pytorch_outputs.py
│   ├── trt_inference_overlay.py
│   └── onnx/
│
├── checkpoints/
│
├── export_onnx.py
│
└── README.md
```

---

# Key Learnings

During this project:

- Understood transformer-based vision architectures.
- Studied VGGT internals and feature aggregation.
- Integrated SegFormer with a pretrained transformer encoder.
- Built a complete segmentation fine-tuning pipeline.
- Learned ONNX deployment workflows.
- Learned TensorRT engine generation and validation.
- Built an optimized production inference pipeline.
- Benchmarked deployment performance on NVIDIA A100 GPUs.

---

# Final Outcome

Successfully developed and deployed a VGGT + SegFormer object masking model capable of generating accurate binary object masks from RGB images.

Final deployment includes:

- Fine-tuned segmentation model
- ONNX export pipeline
- TensorRT FP16 engine
- Overlay visualization pipeline
- Production-ready inference workflow

The project demonstrates an end-to-end workflow from research model modification and fine-tuning to optimized deployment and inference acceleration.

# Author

Dikshit Rishi  
AI/ML & Vision Engineer  
Terafac