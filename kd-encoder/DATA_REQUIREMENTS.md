# Data Requirements for Distillation Training

## What Data is Needed?

**Simple answer:** A directory of RGB images (`.jpg`, `.png`, `.jpeg`)

### Directory Structure

```
your_image_directory/
├── image_001.jpg
├── image_002.jpg
├── image_003.jpg
├── ...
└── image_N.jpg
```

### Data Specifications

| Property | Requirement |
|----------|-------------|
| **Format** | `.jpg`, `.png`, `.jpeg` |
| **Minimum Images** | ~1000+ recommended (more = better) |
| **Image Size** | Any size (will be resized to 518×518) |
| **Color Space** | RGB (3 channels) |
| **Labels** | Not required! (unsupervised distillation) |

## Why No Labels Needed?

Distillation is **unsupervised** - we're training the student to match teacher's *features*, not predict labels. The teacher already knows how to extract good features, and we're teaching the student to do the same.

## Data Sources

You can use images from:
1. **Your existing dataset** (e.g., the same data used to train VGGT)
2. **ImageNet** (if you have access)
3. **COCO** (object detection dataset, but just using images)
4. **Your own unlabeled images** (drone footage, surveillance, etc.)

## Current Dataset Options

Based on your existing work with VGGT:

### Option 1: Use VGGT Training Data
If you trained VGGT on your own dataset, use the **same images**:
```bash
python train.py --image_dir /path/to/vggt/training/images
```

### Option 2: Use Multi-View Dataset
If you have multi-view data (8 frames per sample):
```
your_data/
├── sample_001/
│   ├── frame_0.jpg
│   ├── frame_1.jpg
│   ├── ...
│   └── frame_7.jpg
├── sample_002/
│   └── ...
```

**Note:** Current dataset implementation replicates single images to create sequences. For real multi-view data, you'll need to modify `training/dataset.py` to load multiple frames.

### Option 3: ImageNet Subset
Download a subset of ImageNet:
```bash
# Example with 10k images
python train.py --image_dir /path/to/imagenet/train
```

## How Much Data?

| Dataset Size | Expected Result |
|-------------|-----------------|
| 100-500 images | Proof of concept (will underfit) |
| 1k-5k images | Decent distillation |
| 10k-50k images | Good distillation |
| 100k+ images | Best results |

**Recommendation:** Start with 5k-10k images for reasonable results.

## Data Preprocessing

The training pipeline automatically handles:
- ✓ Resize to 518×518
- ✓ Normalize with ImageNet stats
- ✓ Convert to tensor
- ✓ Create frame sequences (replicates image 8 times currently)

**You only need to provide:** Directory of raw images!

## Quick Test

To verify your data works:
```bash
# Sanity check with your images
python sanity_check.py --image_dir /path/to/your/images --epochs 3
```

This will:
1. Load a few batches
2. Run 3 epochs
3. Verify everything works before full training

## Example Usage

```bash
# Sanity check (3-5 epochs, quick test)
python sanity_check.py --image_dir /data/images

# Full training (50 epochs)
python train.py --image_dir /data/images --epochs 50 --batch_size 4

# Resume training
python train.py --resume_from checkpoints/checkpoint_last.pt
```

## FAQ

**Q: Do I need the same images used to train the teacher?**  
A: No! Any diverse set of RGB images works. Distillation learns general feature extraction.

**Q: Can I use different images than the original VGGT?**  
A: Yes! As long as they're natural RGB images.

**Q: Do images need to be labeled?**  
A: No! Distillation is unsupervised.

**Q: What if images are different sizes?**  
A: That's fine - they'll be automatically resized to 518×518.

**Q: Can I use grayscale images?**  
A: No - must be RGB (3 channels).

**Q: How long does training take?**  
A: ~8-12 hours for 50 epochs on modern GPU (depends on dataset size).
