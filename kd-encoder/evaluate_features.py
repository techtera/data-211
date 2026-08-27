#!/usr/bin/env python3
"""
Evaluate student encoder quality by comparing features with teacher.

Tests feature similarity on sample images WITHOUT training decoders.

Usage:
    python evaluate_features.py --student checkpoints_full/checkpoint_epoch_36.pt \
                                 --teacher ../../vggt-unified/checkpoints/vggt_unified_fp16.pt \
                                 --images /path/to/test/images/*.jpg
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import glob
from PIL import Image
from torchvision import transforms
import sys

# Import student and teacher
from student import StudentAggregator
from load_real_teacher import load_real_teacher


def load_student(checkpoint_path, device):
    """Load student encoder from checkpoint."""
    print(f"\n[1] Loading student from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict', checkpoint))

    student = StudentAggregator()
    student.load_state_dict(state_dict)
    student.eval()
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', '?')

    print(f"  ✓ Student loaded: {student_params:,} params")
    print(f"  Epoch: {epoch}, Training Loss: {loss}")

    return student


def load_test_images(image_paths, max_images=20):
    """Load and preprocess test images."""
    print(f"\n[2] Loading test images...")

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])

    images = []
    valid_paths = []

    for img_path in image_paths[:max_images]:
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"  ⚠️  Skipped {img_path}: {e}")

    if not images:
        raise ValueError("No valid images found!")

    # Stack: [B, 3, H, W] -> [B, S=1, 3, H, W]
    batch = torch.stack(images).unsqueeze(1)

    print(f"  ✓ Loaded {len(images)} images")
    print(f"  Shape: {list(batch.shape)}")

    return batch, valid_paths


def extract_features(model, images, is_teacher=False):
    """Extract features from model."""
    with torch.no_grad():
        aggregated_list, patch_start = model(images)

    # Extract features from cached layers
    if is_teacher:
        layer_indices = [4, 11, 17, 23]  # Teacher layers
    else:
        layer_indices = [3, 8, 13, 17]   # Student layers

    features = []
    for idx in layer_indices:
        feat = aggregated_list[idx]
        if feat is not None:
            # Remove special tokens, keep only patch features
            feat = feat[:, :, patch_start:]  # [B, S, Patches, Dim]
            features.append(feat)

    return features


def compute_similarity(student_feats, teacher_feats):
    """
    Compute cosine similarity between student and teacher features.

    Handles dimension mismatch (student 1536-dim vs teacher 2048-dim)
    by comparing normalized feature distributions.
    """
    assert len(student_feats) == len(teacher_feats) == 4, "Expected 4 layer features"

    similarities = []

    for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
        # Flatten spatial dimensions: [B, S, Patches, Dim] -> [B, S*Patches, Dim]
        s_flat = s_feat.flatten(0, 2)  # [B*S*Patches, 1536]
        t_flat = t_feat.flatten(0, 2)  # [B*S*Patches, 2048]

        # Option 1: Compare normalized L2 norms (scale-invariant)
        s_norm = F.normalize(s_flat, dim=1)
        t_norm = F.normalize(t_flat, dim=1)

        # Project to common dimension via correlation
        # Compute feature-wise correlation
        correlation = torch.mm(s_norm.T, t_norm)  # [1536, 2048]

        # Average correlation strength
        sim = correlation.abs().mean().item()

        similarities.append(sim)

    return similarities


def compute_direct_similarity(student_feats, teacher_feats):
    """
    Compute direct cosine similarity on first N dimensions.

    Compares student[0:1536] with teacher[0:1536] directly.
    """
    similarities = []

    for s_feat, t_feat in zip(student_feats, teacher_feats):
        # Flatten
        s_flat = s_feat.flatten(0, 2)  # [N, 1536]
        t_flat = t_feat.flatten(0, 2)  # [N, 2048]

        # Compare first 1536 dimensions
        t_trimmed = t_flat[:, :1536]

        # Cosine similarity
        s_norm = F.normalize(s_flat, dim=1)
        t_norm = F.normalize(t_trimmed, dim=1)

        sim = (s_norm * t_norm).sum(dim=1).mean().item()
        similarities.append(sim)

    return similarities


def interpret_results(similarities, method="correlation"):
    """Interpret similarity scores."""
    avg_sim = sum(similarities) / len(similarities)

    print(f"\n{'='*60}")
    print(f"FEATURE SIMILARITY RESULTS ({method})")
    print(f"{'='*60}")

    print(f"\nPer-layer similarity:")
    layer_names = ["Early (3/4)", "Mid-Early (8/11)", "Mid-Late (13/17)", "Final (17/23)"]
    for i, (name, sim) in enumerate(zip(layer_names, similarities)):
        status = "✓" if sim > 0.75 else "⚠️" if sim > 0.65 else "✗"
        print(f"  {status} Layer {i+1} [{name}]: {sim:.4f}")

    print(f"\nOverall Average: {avg_sim:.4f}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if avg_sim > 0.85:
        quality = "EXCELLENT"
        emoji = "🎉"
        advice = "Student has learned teacher's feature space very well!"
        action = "✅ Proceed with decoder training confidently."
    elif avg_sim > 0.75:
        quality = "GOOD"
        emoji = "✅"
        advice = "Student captured most of teacher's knowledge."
        action = "✅ Safe to proceed with decoder training."
    elif avg_sim > 0.65:
        quality = "ACCEPTABLE"
        emoji = "⚠️"
        advice = "Student learned basics but missing some nuances."
        action = "⚠️  Decoder training will work, but performance may be slightly lower."
    else:
        quality = "POOR"
        emoji = "❌"
        advice = "Student did not learn teacher's features well."
        action = "❌ Consider retraining with adjusted hyperparameters."

    print(f"\n{emoji} Quality: {quality} (Score: {avg_sim:.4f})")
    print(f"\n{advice}")
    print(f"{action}")
    print(f"\n{'='*60}")

    return avg_sim


def main():
    parser = argparse.ArgumentParser(description="Evaluate student encoder feature similarity")
    parser.add_argument('--student', type=str,
                       default='checkpoints_full/checkpoint_epoch_36.pt',
                       help='Student checkpoint path')
    parser.add_argument('--teacher', type=str,
                       default='../../vggt-unified/checkpoints/vggt_unified_fp16.pt',
                       help='Teacher checkpoint path')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to test images (supports glob pattern)')
    parser.add_argument('--max_images', type=int, default=20,
                       help='Maximum number of images to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    args = parser.parse_args()

    print("="*60)
    print("STUDENT ENCODER FEATURE SIMILARITY EVALUATION")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load models
    student = load_student(args.student, device)

    print(f"\n[1.5] Loading teacher from {args.teacher}")
    teacher = load_real_teacher(args.teacher, device)
    teacher.eval()
    print(f"  ✓ Teacher loaded")

    # Load images
    image_paths = glob.glob(args.images)
    if not image_paths:
        print(f"\n❌ No images found at: {args.images}")
        return

    images, valid_paths = load_test_images(image_paths, args.max_images)
    images = images.to(device)

    # Extract features
    print(f"\n[3] Extracting features...")
    print(f"  Student layers: [3, 8, 13, 17]")
    print(f"  Teacher layers: [4, 11, 17, 23]")

    student_feats = extract_features(student, images, is_teacher=False)
    teacher_feats = extract_features(teacher, images, is_teacher=True)

    print(f"  ✓ Extracted {len(student_feats)} feature layers")
    print(f"    Student feature dims: {[f.shape[-1] for f in student_feats]}")
    print(f"    Teacher feature dims: {[f.shape[-1] for f in teacher_feats]}")

    # Compute similarities
    print(f"\n[4] Computing feature similarity...")

    # Method 1: Correlation-based (handles dimension mismatch)
    corr_sims = compute_similarity(student_feats, teacher_feats)
    avg_corr = interpret_results(corr_sims, method="Cross-Correlation")

    # Method 2: Direct comparison (first 1536 dims)
    print(f"\n{'='*60}")
    direct_sims = compute_direct_similarity(student_feats, teacher_feats)
    avg_direct = interpret_results(direct_sims, method="Direct Cosine (1536-dim)")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nCross-Correlation Score: {avg_corr:.4f}")
    print(f"Direct Cosine Score: {avg_direct:.4f}")
    print(f"\nBoth scores should be >0.75 for good distillation.")
    print(f"\nCheckpoint evaluated: {args.student}")
    print(f"Tested on {len(valid_paths)} images")


if __name__ == "__main__":
    main()
