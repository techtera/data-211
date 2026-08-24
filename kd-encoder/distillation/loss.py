"""
Distillation Loss: MSE (70%) + Cosine (30%) with progressive layer weighting.

Components:
    - MSE: Magnitude matching (squared distance)
    - Cosine: Direction matching (semantic alignment)
    - Layer weights [1.0, 1.5, 2.0, 2.5]: Later layers weighted more (higher semantics)

Formula per layer:
    1. Project student (1536) → (2048)
    2. Layer loss = 0.7×MSE + 0.3×Cosine
    3. Weighted = layer_weights[i] × layer_loss
    Total = sum(weighted) / sum(layer_weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .projection import MultiLayerProjection


class DistillationLoss(nn.Module):
    """
    Distillation loss with projection: Projects student→teacher, computes MSE+Cosine.
    Returns (total_loss, metrics_dict) for training loop.
    """

    def __init__(
        self,
        student_dim: int = 1536,
        teacher_dim: int = 2048,
        num_layers: int = 4,
        layer_weights: List[float] = None,
        mse_weight: float = 0.7,
        cosine_weight: float = 0.3
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

        # Progressive layer weights (later layers = more important)
        if layer_weights is None:
            layer_weights = [1.0, 1.5, 2.0, 2.5]
        self.layer_weights = layer_weights

        # Projection heads (training-only, discarded after)
        self.projection = MultiLayerProjection(
            num_layers=num_layers,
            student_dim=student_dim,
            teacher_dim=teacher_dim
        )

        self.num_layers = num_layers

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss: project student→teacher, MSE+Cosine per layer, progressive weighting.

        Args:
            student_features: List of [B,S,P,1536], length=4
            teacher_features: List of [B,S,P,2048], length=4

        Returns:
            total_loss: Scalar for backward()
            metrics: Dict with per-layer breakdown
        """
        assert len(student_features) == self.num_layers
        assert len(teacher_features) == self.num_layers

        total_loss = 0.0
        metrics = {}

        for i in range(self.num_layers):
            s_feat = student_features[i]
            t_feat = teacher_features[i]

            # Project student to teacher dimension
            s_feat_proj = self.projection.project_layer(s_feat, i)

            # MSE loss (magnitude matching)
            mse_loss = F.mse_loss(s_feat_proj, t_feat)

            # Cosine loss (direction matching)
            s_flat = s_feat_proj.flatten(1)
            t_flat = t_feat.flatten(1)
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
            cosine_loss = 1.0 - cos_sim

            # Weighted combination
            layer_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
            weighted_loss = self.layer_weights[i] * layer_loss

            total_loss += weighted_loss

            # Store metrics
            metrics[f'layer_{i}_mse'] = mse_loss.item()
            metrics[f'layer_{i}_cosine_sim'] = cos_sim.item()
            metrics[f'layer_{i}_cosine_loss'] = cosine_loss.item()
            metrics[f'layer_{i}_loss'] = layer_loss.item()
            metrics[f'layer_{i}_weighted_loss'] = weighted_loss.item()

        # Normalize by sum of layer weights
        total_loss = total_loss / sum(self.layer_weights)
        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics


class SimplifiedDistillationLoss(nn.Module):
    """Simplified loss WITHOUT projection (for testing when dimensions already match)."""

    def __init__(
        self,
        num_layers: int = 4,
        layer_weights: List[float] = None,
        mse_weight: float = 0.7,
        cosine_weight: float = 0.3
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

        if layer_weights is None:
            layer_weights = [1.0, 1.5, 2.0, 2.5]
        self.layer_weights = layer_weights
        self.num_layers = num_layers

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute loss (no projection, assumes dimensions match)."""
        total_loss = 0.0
        metrics = {}

        for i in range(self.num_layers):
            s_feat = student_features[i]
            t_feat = teacher_features[i]

            # MSE + Cosine
            mse_loss = F.mse_loss(s_feat, t_feat)
            s_flat = s_feat.flatten(1)
            t_flat = t_feat.flatten(1)
            cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1).mean()
            cosine_loss = 1.0 - cos_sim

            layer_loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss
            weighted_loss = self.layer_weights[i] * layer_loss

            total_loss += weighted_loss

            metrics[f'layer_{i}_mse'] = mse_loss.item()
            metrics[f'layer_{i}_cosine_sim'] = cos_sim.item()
            metrics[f'layer_{i}_loss'] = layer_loss.item()

        total_loss = total_loss / sum(self.layer_weights)
        metrics['total_loss'] = total_loss.item()

        return total_loss, metrics
