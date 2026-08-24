# Distillation module for Phase 1 training

from .token_sampling import (
    sample_tokens,
    sample_tokens_with_indices,
    get_sampling_stats
)
from .projection import (
    ProjectionHead,
    MultiLayerProjection,
    count_projection_parameters
)
from .loss import (
    DistillationLoss,
    SimplifiedDistillationLoss
)

__all__ = [
    # Token sampling
    'sample_tokens',
    'sample_tokens_with_indices',
    'get_sampling_stats',
    # Projection
    'ProjectionHead',
    'MultiLayerProjection',
    'count_projection_parameters',
    # Loss
    'DistillationLoss',
    'SimplifiedDistillationLoss',
]
