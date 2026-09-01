# Standalone Inference Script

## Overview

`infer_standalone.py` is a completely self-contained inference script with **NO external dependencies** on the project structure. All model architectures are inlined.

## Features

- **Zero project imports**: No imports from `student/`, `obj_mask/`, or any project modules
- **All architectures inlined**: 1135 lines, 13 model components
- **Standard library only**: torch, PIL, numpy, argparse, pathlib
- **Same CLI interface**: Compatible with existing inference workflow

## Architecture Components (All Inlined)

1. **StudentAggregator** (255M params)
   - 18-layer transformer with alternating frame/global attention
   - RoPE 2D position embeddings
   - Cached layers: [3, 8, 13, 17]

2. **DPTHead + SegFormer Decoder** (17M params)
   - Multi-scale feature pyramid
   - SegFormer decoder for segmentation

Total: **272.5M parameters**

## Usage

### Single Image

```bash
python infer_standalone.py input.jpg output.png
python infer_standalone.py input.jpg output.png --overlay
```

### Batch Processing

```bash
python infer_standalone.py input_dir/ output_dir/ --batch
python infer_standalone.py input_dir/ output_dir/ --batch --overlay
```

### Custom Checkpoints

```bash
python infer_standalone.py input.jpg output.png \
    --encoder /path/to/student.pt \
    --decoder /path/to/decoder.pt
```

## Checkpoints

**Default paths:**
- Encoder: `../kd-encoder/checkpoints_full/student_final.pt`
- Decoder: `checkpoints/checkpoint_best.pt`

## Verification

```bash
# Test help
python infer_standalone.py --help

# Verify imports
python -c "import infer_standalone; print('✓ OK')"

# Test model instantiation (no weights needed)
python -c "
import torch
import infer_standalone
model = infer_standalone.StudentObjMask(infer_standalone.StudentAggregator())
print(f'✓ Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')
"
```

## Input/Output Specs

- **Input**: RGB images, auto-resized to 518x518
- **Output**: Binary masks (0=background, 1=object)
- **Formats**: JPG, JPEG, PNG
- **Overlay**: Optional red mask visualization (50% alpha)

## Performance

- Forward pass tested: (1, 3, 518, 518) → (1, 2, 518, 518)
- Latency: Measured per image (GPU sync)
- Batch throughput: Reported images/sec

## File Size

- **Lines**: 1135
- **Components**: 13 model classes
- **Helper functions**: 6 utilities (UV grid, position embed, etc.)

## Deployment

This file is ready for:
- Standalone deployment
- Docker containers
- Testing environments
- Production serving

No project structure or additional files required.
