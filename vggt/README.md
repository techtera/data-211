# Fine-Tuning for Semantic Segmentation

This repository extends the original VGGT architecture by integrating a SegFormer decoder for semantic segmentation and provides a complete fine-tuning pipeline for custom datasets.

## Project Overview

The goal of this project is to adapt VGGT from a 3D geometry prediction model into a semantic segmentation framework while leveraging VGGT's powerful visual representations.

The original DPT-based segmentation branch has been replaced with a SegFormer decoder, and a dedicated training pipeline has been implemented for fine-tuning on custom datasets.

---

## Features

- Integrated SegFormer decoder with VGGT backbone
- End-to-end semantic segmentation pipeline
- Support for custom datasets in YOLO Segmentation format
- Automatic polygon-to-mask conversion
- Cross Entropy + Dice Loss
- TensorBoard logging
- Checkpoint management
- Modular training utilities

---

## Project Structure

```
fine_tuning/
├── checkpoints.py
├── config.py
├── dataset.py
├── logger.py
├── losses.py
├── model_builder.py
├── optimizer.py
├── trainer.py
└── utils.py
```

---

## Dataset Format

The training dataset follows the YOLO Segmentation format.

```
dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
├── labels/
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...
│
├── classes.txt
└── notes.json
```

Each label file contains a single polygon annotation:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

where

- `class_id` is the object class
- `(x, y)` are normalized polygon coordinates.

During loading, these polygons are automatically converted into binary segmentation masks.

---

## Training Pipeline

```
Image
    │
    ▼
VGGT Backbone
    │
    ▼
Multi-scale Feature Maps
(C1, C2, C3, C4)
    │
    ▼
SegFormer Decoder
    │
    ▼
Segmentation Logits
    │
    ▼
Upsampling
    │
    ▼
CrossEntropy + Dice Loss
```

---

## Loss Function

Training uses a combination of:

- Cross Entropy Loss
- Dice Loss

The final optimization objective is:

```
Loss = CrossEntropyLoss + DiceLoss
```

This improves segmentation performance, particularly for datasets with significant foreground/background imbalance.

---

## Current Status

Implemented:

- VGGT backbone integration
- SegFormer decoder integration
- Dataset loader
- Polygon-to-mask conversion
- Training pipeline
- Optimizer
- Combined loss function
- TensorBoard logging
- Checkpoint saving

Planned:

- Validation pipeline
- Evaluation metrics (IoU / Dice)
- Inference script
- Visualization utilities

> **Note:** This repository contains a custom extension of the original VGGT project for semantic segmentation using a SegFormer decoder and a dedicated fine-tuning pipeline.

---

## Maintainer

**Dikshit Rishi**

This repository extends the original VGGT project by adding a complete semantic segmentation fine-tuning pipeline based on a SegFormer decoder.