# VGGT + SegFormer Edge Segmentation

## Project Overview

This project focuses on adapting Meta's Visual Geometry Grounded Transformer (VGGT) architecture for binary edge segmentation. The original VGGT architecture is designed for multi-view 3D understanding tasks such as camera pose estimation, depth prediction, point prediction, and tracking. For this work, the architecture was modified to perform edge segmentation by integrating a SegFormer-based decoder and training it on an edge-mask dataset.

The project also includes deployment optimization using ONNX and TensorRT to achieve low-latency inference on NVIDIA GPUs.

---

# Problem Statement

The objective was to build an edge segmentation model capable of generating binary edge masks where:

- Background pixels = 0 (Black)
- Edge pixels = 1 (White)

The solution needed to:

- Leverage VGGT's strong visual representation capabilities.
- Replace the original prediction head with a segmentation-focused decoder.
- Train and fine-tune the architecture on edge-mask data.
- Deploy the model in an optimized inference pipeline suitable for production use.

---

# Architecture

## Original VGGT Pipeline

```text
Input Image
      ↓
Patch Embedding
      ↓
Transformer Encoder
      ↓
Aggregator
      ↓
Task Heads
    ├── Camera Head
    ├── Depth Head
    ├── Point Head
    └── Track Head
```

The original architecture was not designed for segmentation.

---

## Modified VGGT Pipeline

```text
Input Image
      ↓
Patch Embedding
      ↓
Transformer Encoder
      ↓
Aggregator
      ↓
SegFormer Decoder
      ↓
2-Class Segmentation Mask
```

The original DPT-style prediction head was replaced with a SegFormer-based segmentation decoder.

Output classes:

- Class 0 → Background
- Class 1 → Edge

Final output shape:

```python
[B, 2, H, W]
```

---

# Development Process

## Phase 1: Environment Setup

### Tasks Completed

- Set up VGGT repository.
- Resolved dependency issues.
- Verified inference pipeline.
- Studied VGGT architecture and codebase.

### Key Learnings

- Transformer-based visual encoders.
- Patch embeddings.
- Multi-view feature aggregation.
- VGGT inference flow.

---

## Phase 2: Understanding VGGT

Studied:

### Aggregator

Responsible for:

- Image token generation.
- Transformer feature extraction.
- Feature aggregation.

Output:

```python
aggregated_tokens_list
patch_start_idx
```

---

### Camera Head

Produces:

```python
pose_enc
```

Used for camera pose estimation.

---

### DPT Head

Original dense prediction head.

Initially investigated whether it could be directly reused for segmentation.

---

# Phase 3: Decoder Investigation

Multiple approaches were evaluated.

### Option 1

Directly attach an external segmentation decoder.

Issues:

- Feature mismatch.
- Channel mismatch.
- Spatial resolution mismatch.

Result:

Not practical.

---

### Option 2

Fine-tune with a SegFormer decoder.

Result:

Selected approach.

---

# Phase 4: SegFormer Integration

Implemented:

```python
SegFormer Decoder
```

Integrated decoder into VGGT pipeline.

### Modifications

Replaced original prediction head:

```python
self.depth_head
```

with

```python
SegFormer-based decoder
```

Updated forward pass:

```python
predictions["mask_logits"]
```

Output:

```python
[B, 2, H, W]
```

---

# Phase 5: Dataset Pipeline

Implemented:

### Dataset Loader

- Image loading.
- Mask loading.
- Preprocessing.

### Augmentations

- Resizing.
- Tensor conversion.
- Normalization.

### Validation

Verified:

- Image-mask alignment.
- Tensor dimensions.
- Data integrity.

---

# Phase 6: Training Pipeline

Implemented complete training workflow.

### Components

- Training loop.
- Validation loop.
- Checkpoint saving.
- Best model selection.

### Loss Function

Cross Entropy Loss.

### Optimizer

AdamW.

### Monitoring

- Training loss.
- Validation loss.
- Segmentation quality.

---

# Phase 7: Fine-Tuning

Fine-tuned VGGT encoder with SegFormer decoder.

Training duration:

```text
65 Epochs
```

Monitored:

- Training convergence.
- Validation performance.
- Prediction quality.

---

# Results

The model successfully learned edge structures and produced clean binary edge masks.

Output:

```text
Input Image
      ↓
VGGT Encoder
      ↓
SegFormer Decoder
      ↓
Binary Edge Segmentation Mask
```

---

# Inference Optimization

After training was completed, focus shifted toward deployment optimization.

Goal:

Reduce inference latency and improve throughput.

---

## Step 1: ONNX Export

Created export pipeline.

Converted:

```text
PyTorch
      ↓
ONNX
```

Challenges encountered:

- Dynamic graph issues.
- Device mismatches.
- Output dictionary handling.
- Unsupported export paths.

Resolved all export issues.

Final exported model:

```text
vggt_segformer_clean.onnx
```

---

## Step 2: TensorRT Conversion

Converted ONNX model into TensorRT engine.

Pipeline:

```text
PyTorch
      ↓
ONNX
      ↓
TensorRT FP16 Engine
```

Generated:

```text
vggt_segformer_clean_fp16.engine
```

---

## TensorRT Validation

Verified engine:

### Input

```python
(1, 1, 3, 518, 518)
```

### Output

```python
mask_logits
(1, 2, 518, 518)
```

Confirmed:

- Correct tensor dimensions.
- Correct segmentation output.
- Successful inference execution.

---

# TensorRT Inference Pipeline

Implemented:

```text
Image
    ↓
Preprocessing
    ↓
TensorRT Engine
    ↓
Segmentation Mask
    ↓
Overlay Generation
```

Generated:

- Binary masks.
- Overlay visualizations.

Successfully tested on:

```text
100 validation images
```

---

# Performance Results

Hardware:

```text
NVIDIA A100 80GB
```

TensorRT FP16 Benchmark:

| Metric | Value |
|----------|----------|
| Throughput | 34.64 FPS |
| Mean Latency | 29.14 ms |
| GPU Compute Time | 28.69 ms |
| H2D Transfer | 0.27 ms |
| D2H Transfer | 0.17 ms |

---

# Final Deployment Artifacts

## Training Artifact

```text
best_model.pth
```

---

## ONNX Model

```text
vggt_segformer_clean.onnx
```

---

## TensorRT Engine

```text
vggt_segformer_clean_fp16.engine
```

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
│   ├── heads/
│   ├── layers/
│   └── models/
│
├── deployment/
│   ├── export_onnx.py
│   ├── validate_pytorch_outputs.py
│   ├── trt_inference_overlay.py
│   └── onnx/
│
├── checkpoints/
│
└── training/
```

---

# Key Outcomes

- Successfully adapted VGGT for edge segmentation.
- Integrated SegFormer decoder into VGGT architecture.
- Built complete dataset and training pipeline.
- Fine-tuned model for binary edge prediction.
- Exported model to ONNX.
- Converted model to TensorRT FP16.
- Validated deployment pipeline.
- Generated overlay-based inference outputs.
- Achieved ~34.6 FPS and ~29 ms latency on NVIDIA A100.

---

# Future Improvements

- INT8 TensorRT quantization.
- Dynamic batch support.
- TensorRT plugin optimization.
- Mixed-resolution inference.
- Multi-class segmentation support.
- Edge refinement post-processing.

---

# Author

Dikshit Rishi  
AI/ML & Vision Engineer  
Terafac