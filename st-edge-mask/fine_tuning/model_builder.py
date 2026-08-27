"""Model Builder for Edge Mask Fine-Tuning with Student Encoder."""

import sys
import torch

sys.path.insert(0, "../kd-encoder")

from student import StudentAggregator
from edge_mask.model import StudentEdgeMask
from .config import DEVICE, STUDENT_CHECKPOINT

_MODEL = None


def print_parameter_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print("\n" + "=" * 60)
    print("Parameter Summary")
    print("=" * 60)
    print(f"Total Parameters      : {total:,}")
    print(f"Trainable Parameters  : {trainable:,}")
    print(f"Frozen Parameters     : {frozen:,}")


def print_module_summary(model):
    print("\n" + "=" * 60)
    print("Module-wise Parameter Summary")
    print("=" * 60)

    modules = {
        "Student Aggregator (frozen)": model.feature_extractor.aggregator,
        "Feature Projections": model.feature_extractor.projections,
        "UNet++ Decoder": model.decoder,
        "Edge Refinement": model.refinement,
        "Final Conv": model.final_conv,
    }

    for name, module in modules.items():
        if module is None:
            continue
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:<30} Total: {total:>12,} | Trainable: {trainable:>12,}")


def build_model():
    global _MODEL

    if _MODEL is not None:
        print("\nUsing cached model.\n")
        return _MODEL

    print("=" * 60)
    print("Loading Student Encoder")
    print("=" * 60)
    print(f"Checkpoint : {STUDENT_CHECKPOINT}")

    checkpoint = torch.load(STUDENT_CHECKPOINT, map_location='cpu')
    state_dict = checkpoint.get('student_state_dict', checkpoint.get('model_state_dict', checkpoint))

    student_aggregator = StudentAggregator()
    student_aggregator.load_state_dict(state_dict)
    student_aggregator.eval()
    student_aggregator.requires_grad_(False)

    student_params = sum(p.numel() for p in student_aggregator.parameters())
    print(f"Student encoder loaded: {student_params:,} parameters")

    print("\n" + "=" * 60)
    print("Building StudentEdgeMask")
    print("=" * 60)

    model = StudentEdgeMask(student_aggregator)
    model.to(DEVICE)

    print_parameter_summary(model)
    print_module_summary(model)

    _MODEL = model
    return _MODEL
