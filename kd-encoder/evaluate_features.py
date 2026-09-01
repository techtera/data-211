#!/usr/bin/env python3
"""
COMPREHENSIVE Student Encoder Quality Evaluation

Measures:
1. Cross-correlation (feature similarity to teacher)
2. Feature variance (discriminative power for segmentation)
3. Activation sparsity (information density)
4. Feature statistics (mean, std, range)

Tests feature quality on sample images WITHOUT training decoders.

Usage:
    python evaluate_features.py --student checkpoints_full/student_final.pt \
                                 --teacher ../../vggt-unified/checkpoints/vggt_unified_fp16.pt \
                                 --images /path/to/test/images/*.jpg \
                                 --max_images 50
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


def compute_feature_variance(features, name="Model"):
    """
    Measure feature variance (discriminative power).
    Higher variance = more discriminative features.
    """
    variances = []
    for i, feat in enumerate(features):
        feat_flat = feat.flatten(0, 2)  # [B*S*Patches, Dim]
        var_per_dim = feat_flat.var(dim=0)  # [Dim]
        mean_var = var_per_dim.mean().item()
        variances.append(mean_var)
    return variances


def compute_activation_sparsity(features):
    """
    Measure activation sparsity (information density).
    Lower sparsity = more information encoded.
    """
    sparsities = []
    for i, feat in enumerate(features):
        feat_flat = feat.flatten()
        near_zero = (feat_flat.abs() < 0.01).float().mean().item()
        mean_abs = feat_flat.abs().mean().item()
        sparsities.append({'near_zero': near_zero, 'mean_abs': mean_abs})
    return sparsities


def compute_feature_statistics(features):
    """Compute basic statistics: mean, std, range."""
    stats = []
    for i, feat in enumerate(features):
        feat_flat = feat.flatten()
        stats.append({
            'mean': feat_flat.mean().item(),
            'std': feat_flat.std().item(),
            'range': (feat_flat.max() - feat_flat.min()).item()
        })
    return stats


def interpret_results(similarities, method="correlation",
                     student_variance=None, teacher_variance=None,
                     student_sparsity=None, teacher_sparsity=None,
                     student_stats=None, teacher_stats=None):
    """Interpret similarity scores with comprehensive metrics."""
    avg_sim = sum(similarities) / len(similarities)

    print(f"\n{'='*70}")
    print(f"FEATURE SIMILARITY RESULTS ({method})")
    print(f"{'='*70}")

    print(f"\nPer-layer similarity:")
    layer_names = ["Early (3/4)", "Mid-Early (8/11)", "Mid-Late (13/17)", "Final (17/23)"]
    for i, (name, sim) in enumerate(zip(layer_names, similarities)):
        status = "✓" if sim > 0.75 else "⚠️" if sim > 0.65 else "✗"
        print(f"  {status} Layer {i+1} [{name}]: {sim:.4f}")

    print(f"\nOverall Average: {avg_sim:.4f}")

    # Comprehensive metrics
    issues = []

    if student_variance and teacher_variance:
        print(f"\n{'='*70}")
        print("FEATURE VARIANCE (Discriminative Power)")
        print(f"{'='*70}")

        avg_s_var = sum(student_variance) / len(student_variance)
        avg_t_var = sum(teacher_variance) / len(teacher_variance)
        var_ratio = avg_s_var / (avg_t_var + 1e-8)

        print(f"\nMean variance per layer:")
        for i, name in enumerate(layer_names):
            s_var = student_variance[i]
            t_var = teacher_variance[i]
            ratio = s_var / (t_var + 1e-8)
            status = "✓" if ratio > 0.7 else "⚠️" if ratio > 0.5 else "✗"
            print(f"  {status} {name:<20}: S={s_var:.6f} | T={t_var:.6f} | Ratio={ratio:.3f}")

        print(f"\nOverall variance ratio (S/T): {var_ratio:.3f}")

        if var_ratio < 0.5:
            print(f"  ⚠️  WARNING: Student features have LOW variance!")
            print(f"      Poor discriminative power for segmentation tasks.")
            issues.append("Low variance")
        elif var_ratio < 0.7:
            print(f"  ⚠️  Student variance is reduced but acceptable.")
            issues.append("Reduced variance")

    if student_sparsity and teacher_sparsity:
        print(f"\n{'='*70}")
        print("ACTIVATION SPARSITY (Information Density)")
        print(f"{'='*70}")

        print(f"\nFraction of near-zero activations (<0.01):")
        for i, name in enumerate(layer_names):
            s_spar = student_sparsity[i]['near_zero']
            t_spar = teacher_sparsity[i]['near_zero']
            s_abs = student_sparsity[i]['mean_abs']
            t_abs = teacher_sparsity[i]['mean_abs']
            print(f"  {name:<20}: S={s_spar:.3f} ({s_abs:.4f}) | T={t_spar:.3f} ({t_abs:.4f})")

        avg_s_spar = sum(s['near_zero'] for s in student_sparsity) / len(student_sparsity)
        avg_t_spar = sum(s['near_zero'] for s in teacher_sparsity) / len(teacher_sparsity)

        if avg_s_spar > avg_t_spar * 1.5:
            print(f"\n  ⚠️  Student is TOO SPARSE (less information density)")
            issues.append("High sparsity")

    if student_stats and teacher_stats:
        print(f"\n{'='*70}")
        print("FEATURE STATISTICS")
        print(f"{'='*70}")

        print(f"\nMean activation values:")
        for i, name in enumerate(layer_names):
            s_mean = student_stats[i]['mean']
            t_mean = teacher_stats[i]['mean']
            print(f"  {name:<20}: S={s_mean:>8.4f} | T={t_mean:>8.4f}")

        print(f"\nStandard deviation:")
        for i, name in enumerate(layer_names):
            s_std = student_stats[i]['std']
            t_std = teacher_stats[i]['std']
            print(f"  {name:<20}: S={s_std:>8.4f} | T={t_std:>8.4f}")

        print(f"\nValue range:")
        for i, name in enumerate(layer_names):
            s_range = student_stats[i]['range']
            t_range = teacher_stats[i]['range']
            print(f"  {name:<20}: S={s_range:>8.4f} | T={t_range:>8.4f}")

    # Final Interpretation
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    if avg_sim < 0.75:
        issues.append("Low correlation")

    if not issues:
        quality = "EXCELLENT"
        emoji = "🎉"
        advice = "Student has learned teacher's feature space very well!"
        action = "✅ Proceed with decoder training confidently."
    elif len(issues) == 1 and "Reduced" in issues[0]:
        quality = "GOOD"
        emoji = "✅"
        advice = "Student captured most of teacher's knowledge with minor quality loss."
        action = "✅ Safe to proceed. Decoder performance may be ~10% lower."
    elif len(issues) <= 2:
        quality = "ACCEPTABLE"
        emoji = "⚠️"
        advice = "Student learned basics but has notable limitations."
        action = "⚠️  Decoder training will work, but expect degraded performance."
    else:
        quality = "POOR"
        emoji = "❌"
        advice = "Student has significant feature quality issues!"
        action = "❌ This likely explains poor decoder performance. Consider:\n" \
                 "      1. Try later checkpoints (epoch 60-80)\n" \
                 "      2. Increase student capacity\n" \
                 "      3. Retrain with adjusted KD hyperparameters"

    print(f"\n{emoji} Quality: {quality} (Correlation: {avg_sim:.4f})")
    print(f"\nIssues found: {', '.join(issues) if issues else 'None'}")
    print(f"\n{advice}")
    print(f"{action}")
    print(f"\n{'='*70}")

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

    # Compute all metrics
    print(f"\n[4] Computing comprehensive metrics...")

    # Cross-correlation
    corr_sims = compute_similarity(student_feats, teacher_feats)

    # Feature variance (discriminative power)
    student_variance = compute_feature_variance(student_feats, "Student")
    teacher_variance = compute_feature_variance(teacher_feats, "Teacher")

    # Activation sparsity (information density)
    student_sparsity = compute_activation_sparsity(student_feats)
    teacher_sparsity = compute_activation_sparsity(teacher_feats)

    # Feature statistics
    student_stats = compute_feature_statistics(student_feats)
    teacher_stats = compute_feature_statistics(teacher_feats)

    print(f"  ✓ All metrics computed")

    # Comprehensive interpretation
    avg_corr = interpret_results(
        corr_sims,
        method="Cross-Correlation",
        student_variance=student_variance,
        teacher_variance=teacher_variance,
        student_sparsity=student_sparsity,
        teacher_sparsity=teacher_sparsity,
        student_stats=student_stats,
        teacher_stats=teacher_stats
    )

    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nCross-Correlation Score: {avg_corr:.4f}")
    print(f"Variance Ratio: {sum(student_variance)/sum(teacher_variance):.3f}")
    print(f"\nTarget: Correlation >0.75, Variance Ratio >0.7")
    print(f"\nCheckpoint evaluated: {args.student}")
    print(f"Tested on {len(valid_paths)} images")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
