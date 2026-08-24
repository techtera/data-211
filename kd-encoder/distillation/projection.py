"""
Projection Heads: Align student (1536) to teacher (2048) dimensions for loss computation.

Architecture:
    - ProjectionHead: LayerNorm + Linear (1536 → 2048)
    - MultiLayerProjection: 4 separate heads, one per cached layer (~12.6M params)
    - Separate heads allow layer-specific transformations

IMPORTANT: Training-only! Discarded after distillation completes.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Single projection head: LayerNorm + Linear (1536 → 2048). ~3.15M params."""

    def __init__(self, student_dim: int = 1536, teacher_dim: int = 2048):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(student_dim),
            nn.Linear(student_dim, teacher_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project [B, S, P, 1536] → [B, S, P, 2048]."""
        return self.projection(x)


class MultiLayerProjection(nn.Module):
    """
    4 separate projection heads for 4 cached layers.
    Different layers have different feature distributions → layer-specific projections.
    """

    def __init__(
        self,
        num_layers: int = 4,
        student_dim: int = 1536,
        teacher_dim: int = 2048
    ):
        super().__init__()
        self.num_layers = num_layers
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

        # One head per cached layer
        self.projection_heads = nn.ModuleList([
            ProjectionHead(student_dim, teacher_dim)
            for _ in range(num_layers)
        ])

    def forward(self, student_features: list) -> list:
        """Project all layers: List[[B,S,P,1536]] → List[[B,S,P,2048]]."""
        projected = []
        for i, s_feat in enumerate(student_features):
            proj = self.projection_heads[i](s_feat)
            projected.append(proj)
        return projected

    def project_layer(self, student_feature: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Project single layer: [B,S,P,1536] → [B,S,P,2048]."""
        return self.projection_heads[layer_idx](student_feature)


def count_projection_parameters(projection: nn.Module) -> dict:
    """Count parameters (total, per_head, num_heads) for logging."""
    total = sum(p.numel() for p in projection.parameters())

    if isinstance(projection, MultiLayerProjection):
        per_head = sum(p.numel() for p in projection.projection_heads[0].parameters())
        return {
            'total': total,
            'per_head': per_head,
            'num_heads': projection.num_layers,
        }
    else:
        return {
            'total': total,
            'per_head': total,
            'num_heads': 1,
        }
