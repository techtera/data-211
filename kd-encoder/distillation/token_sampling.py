"""
Token Sampling: Reduce memory by sampling 133/1374 tokens for distillation loss.

VGGT token structure:
    [0]: Camera token
    [1-4]: 4 register tokens
    [5-1373]: 1369 patch tokens (37×37 grid)

Sampling strategy:
    - Keep all 5 special tokens (camera + registers)
    - Sample 128 random patches → 133 total (90% memory reduction)

CRITICAL: Teacher and student must use SAME patch indices for spatial alignment!
    1. sample_tokens(teacher) → returns (features, indices)
    2. sample_tokens_with_indices(student, indices) → uses same indices
"""

import torch
from typing import Tuple


def sample_tokens(
    features: torch.Tensor,
    patch_start_idx: int = 5,
    num_patch_samples: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tokens: keep all special tokens + random patch subset.
    Returns (sampled_features [B,S,133,C], indices [128]) for alignment.
    """
    B, S, P, C = features.shape

    assert P == 1374, f"Expected 1374 tokens, got {P}"

    # Split: special tokens (0-4) and patch tokens (5+)
    special_tokens = features[:, :, :patch_start_idx, :]  # [B, S, 5, C]
    patch_tokens = features[:, :, patch_start_idx:, :]    # [B, S, 1369, C]

    # Randomly sample patches
    num_patches = patch_tokens.shape[2]
    sampled_indices = torch.randperm(num_patches, device=features.device)[:num_patch_samples]
    sampled_patches = patch_tokens[:, :, sampled_indices, :]  # [B, S, 128, C]

    # Concatenate: special + sampled patches
    sampled = torch.cat([special_tokens, sampled_patches], dim=2)  # [B, S, 133, C]

    return sampled, sampled_indices


def sample_tokens_with_indices(
    features: torch.Tensor,
    indices: torch.Tensor,
    patch_start_idx: int = 5
) -> torch.Tensor:
    """
    Sample tokens using provided indices (for student to match teacher).
    Returns sampled_features [B,S,133,C] with SAME spatial locations as teacher.
    """
    B, S, P, C = features.shape

    special_tokens = features[:, :, :patch_start_idx, :]
    patch_tokens = features[:, :, patch_start_idx:, :]

    # Use provided indices (from teacher) instead of random sampling
    sampled_patches = patch_tokens[:, :, indices, :]

    sampled = torch.cat([special_tokens, sampled_patches], dim=2)

    return sampled


def get_sampling_stats(
    original_shape: Tuple[int, ...],
    sampled_shape: Tuple[int, ...],
    patch_start_idx: int = 5,
    num_patch_samples: int = 128
) -> dict:
    """Get sampling statistics for logging (reduction ratio, memory savings, etc)."""
    total_tokens = original_shape[2]
    sampled_tokens = sampled_shape[2]
    num_patches = total_tokens - patch_start_idx

    reduction_ratio = total_tokens / sampled_tokens
    memory_savings = (1 - sampled_tokens / total_tokens) * 100

    return {
        'total_tokens': total_tokens,           # 1374
        'sampled_tokens': sampled_tokens,       # 133
        'special_tokens': patch_start_idx,      # 5
        'total_patches': num_patches,           # 1369
        'sampled_patches': num_patch_samples,   # 128
        'reduction_ratio': reduction_ratio,     # 10.3x
        'memory_savings_pct': memory_savings,   # 90.3%
    }

