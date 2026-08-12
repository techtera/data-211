"""
Model Builder for Edge Mask Fine-Tuning.

Responsibilities
----------------
1. Load pretrained VGGT weights from HuggingFace.
2. Freeze the pretrained encoder (aggregator).
3. Build the VGGTEdgeMask model with trainable decoder.
4. Print a detailed model summary.
"""

import sys
sys.path.insert(0, ".")

from vggt.vggt.models.vggt import VGGT
from edge_mask.model import VGGTEdgeMask

from .config import DEVICE, PRETRAINED_MODEL


# ============================================================
# Global Model Cache
# ============================================================

_MODEL = None


# ============================================================
# Parameter Summary
# ============================================================

def print_parameter_summary(model):
    """
    Print total, trainable and frozen parameters.
    """

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    print("\n" + "=" * 60)
    print("Parameter Summary")
    print("=" * 60)

    print(f"Total Parameters      : {total:,}")
    print(f"Trainable Parameters  : {trainable:,}")
    print(f"Frozen Parameters     : {frozen:,}")


# ============================================================
# Module Summary
# ============================================================

def print_module_summary(model):
    """
    Print parameter statistics for each module.
    """

    print("\n" + "=" * 60)
    print("Module-wise Parameter Summary")
    print("=" * 60)

    modules = {
        "Aggregator (frozen)": model.feature_extractor.aggregator,
        "Feature Projections": model.feature_extractor.projections,
        "UNet++ Decoder": model.decoder,
        "Edge Refinement": model.refinement,
        "Final Conv": model.final_conv,
    }

    for name, module in modules.items():

        if module is None:
            continue

        total = sum(p.numel() for p in module.parameters())

        trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

        print(
            f"{name:<25}"
            f" Total: {total:>12,}"
            f" | Trainable: {trainable:>12,}"
        )


# ============================================================
# Build Model
# ============================================================

def build_model():
    """
    Build the VGGTEdgeMask model for fine-tuning.

    Steps:
        1. Load VGGT from HuggingFace (pretrained)
        2. Pass the aggregator to VGGTEdgeMask
           (encoder is frozen inside VGGTFeatureExtractor)
        3. Move to device
    """

    global _MODEL

    if _MODEL is not None:

        print("\nUsing cached model.\n")

        return _MODEL

    print("=" * 60)
    print("Loading Pretrained VGGT")
    print("=" * 60)

    print(f"Model ID : {PRETRAINED_MODEL}")

    vggt_model = VGGT.from_pretrained(PRETRAINED_MODEL)

    print("VGGT loaded successfully.")

    # --------------------------------------------------------
    # Build Edge Mask Model
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Building VGGTEdgeMask")
    print("=" * 60)

    model = VGGTEdgeMask(vggt_model.aggregator)

    model.to(DEVICE)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_parameter_summary(model)

    print_module_summary(model)

    _MODEL = model

    return _MODEL
