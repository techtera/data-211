# FastVGGT - Token Merging Acceleration

Training-free 3x encoder speedup using token merging. Uses existing checkpoints with no retraining.

Based on: [FastVGGT (ICLR 2026)](https://arxiv.org/abs/2509.02560)

## How It Works

```
Baseline:  1000 tokens -> Attention O(N^2)       -> 1000 tokens
FastVGGT:  1000 tokens -> Merge -> 100 tokens -> Attention O((N/10)^2) -> Unmerge -> 1000 tokens
```

Same checkpoint, same weights. Tokens are merged before attention and unmerged after, so decoders see identical input structure.

## Quick Start

```bash
# Run inference with FastVGGT enabled
python run_inference.py

# Your own images
python run_inference.py --images /path/to/images --num-frames 10

# Different tasks
python run_inference.py --task obj       # Object mask only
python run_inference.py --task edge      # Edge mask only
python run_inference.py --task cascade   # Obj + Edge in ROI (default)

# Baseline comparison (no token merging)
python run_inference.py --no-fastvggt

# Speed benchmark
python test_fastvggt.py
```

## Python API

```python
from model import VGGTUnified

model = VGGTUnified(load_encoder=False)
model.load_unified_checkpoint('path/to/checkpoint.pt')

model.aggregator.enable_token_merging(merge_ratio=0.9, disable_rope=True)
results = model(images, task='cascade')
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--images` | None | Image folder (dummy images if omitted) |
| `--num-frames` | 5 | Number of frames |
| `--task` | cascade | cascade / obj / edge / both |
| `--merge-ratio` | 0.9 | Fraction of tokens to merge (higher = faster) |
| `--no-fastvggt` | False | Disable token merging (baseline) |

## Performance

| Config | Encoder | Total | Quality |
|--------|---------|-------|---------|
| Baseline | 500ms | 750ms | 100% |
| FastVGGT (0.9) | 170ms | 350ms | 98-100% |
| **Speedup** | **3x** | **2.1x** | - |

Longer sequences benefit more (attention is O(N^2)).

## Tuning

```python
# Recommended (default)
model.aggregator.enable_token_merging(merge_ratio=0.9, disable_rope=True)

# Max speed (slight quality trade-off)
model.aggregator.enable_token_merging(merge_ratio=0.95, disable_rope=True)

# Max quality
model.aggregator.enable_token_merging(merge_ratio=0.8, disable_rope=True)

# Disable
model.aggregator.disable_token_merging()
```

`disable_rope=True` is recommended. RoPE positional embeddings combined with token merging can cause shape mismatches.

## Directory Structure

```
fastVGGT/
├── encoder/
│   └── aggregator.py      # Modified: token merging added
├── decoders/
│   ├── edge_mask/
│   └── obj_mask/
├── model.py               # Unified model
├── token_merging.py        # Core merge/unmerge logic
├── run_inference.py        # Main inference script
└── test_fastvggt.py        # Speed comparison
```

## References

- Paper: https://arxiv.org/abs/2509.02560
- Original repo: https://github.com/mystorm16/FastVGGT
