# VGGT Knowledge Distillation Pipeline

Complete pipeline for distilling VGGT-1B (909M) → Student encoder (255M) and training downstream decoders.

## Directory Structure

```
vggt-KD/
├── kd-encoder/           # Knowledge distillation (ongoing: epoch 36/80)
├── st-edge-mask/         # Edge decoder training (ready)
└── st-obj-mask/          # Object decoder training (ready)
```

## Quick Start

### Current: Wait for Student Training
KD training is ongoing on VM. Check status: `tail -f kd-encoder/nohup.out`

### Next: Train Decoders

**Edge Decoder:**
```bash
cd st-edge-mask
# Prepare data in data/rgb/ and data/masks/
python fine_tune.py
```

**Object Decoder:**
```bash
cd st-obj-mask
# IMPORTANT: First edit obj_mask/segformer_head.py (see st-obj-mask/README.md)
# Prepare data in data/images/ and data/masks/
python fine_tune.py
```

## Architecture

**Student Encoder:** 255M params, 1536-dim output (vs VGGT-1B: 909M, 2048-dim)
- Layers: 18 (vs 24)
- Cached: [3,8,13,17] (vs [4,11,17,23])
- Speed: ~3× faster

**Decoders:**
- Edge: UNet++ (~10M trainable params)
- Object: SegFormer (~15M trainable params)

Both adapted for 1536-dim student input.

## Training Timeline

1. ⏳ **Student KD**: ~40-50 hours remaining → `kd-encoder/checkpoints_full/student_final.pt`
2. ⏱️ **Edge decoder**: ~6-8 hours → `st-edge-mask/checkpoints/checkpoint_best.pt`
3. ⏱️ **Object decoder**: ~8-10 hours → `st-obj-mask/checkpoints/checkpoint_best.pt`

## Key Files

- `st-edge-mask/fine_tune.py` - Train edge decoder
- `st-obj-mask/fine_tune.py` - Train object decoder
- `st-edge-mask/README.md` - Edge training guide
- `st-obj-mask/README.md` - Object training guide (includes important setup step!)

## Configuration

Both decoders use configs in their `fine_tuning/config.py`:
- Batch size: 4 (adjust for your GPU)
- Learning rate: 3e-4 (edge), 1e-4 (object)
- Epochs: 100

## Next Steps After Training

1. Merge both decoders into unified checkpoint
2. Deploy to Jetson Orin NX (TensorRT, INT8)
3. Target: <1s latency for full pipeline

---

**Current Status:** Student distillation in progress (epoch 36/80, loss 0.2359)
