# Student encoder module for VGGT knowledge distillation

from .aggregator import StudentAggregator
from .initialization import (
    load_dinov2_vitb14_reg,
    initialize_student_from_dinov2,
    verify_initialization
)

__all__ = [
    'StudentAggregator',
    'load_dinov2_vitb14_reg',
    'initialize_student_from_dinov2',
    'verify_initialization',
]
