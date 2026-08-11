"""
Model Builder for SegFormer Fine-Tuning.

Responsibilities
----------------
1. Load pretrained VGGT weights.
2. Freeze the pretrained encoder.
3. Enable training only for the segmentation head.
4. Print a detailed model summary.
"""

from vggt.models.vggt_modifying import VGGT

from .config import DEVICE, PRETRAINED_MODEL


# ============================================================
# Global Model Cache
# ============================================================

_MODEL = None


# ============================================================
# Freeze Pretrained Encoder
# ============================================================

def freeze_encoder(model):
    """
    Freeze all pretrained encoder modules.
    """

    print("\nFreezing pretrained modules...\n")

    if model.aggregator is not None:
        model.aggregator.requires_grad_(False)
        print("✓ Aggregator Frozen")

    if model.camera_head is not None:
        model.camera_head.requires_grad_(False)
        print("✓ Camera Head Frozen")


# ============================================================
# Enable Segmentation Head
# ============================================================

def unfreeze_segmentation_head(model):
    """
    Enable training only for the SegFormer-based
    segmentation head.
    """

    print("\nEnabling Segmentation Head...\n")

    if model.depth_head is not None:
        model.depth_head.requires_grad_(True)
        print("✓ Depth Head Trainable")


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
        "Aggregator": model.aggregator,
        "Camera Head": model.camera_head,
        "Depth Head": model.depth_head,
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
            f"{name:<15}"
            f" Total: {total:,}"
            f" | Trainable: {trainable:,}"
        )


# ============================================================
# Trainable Layers
# ============================================================

def print_trainable_layers(model):
    """
    Print all trainable parameters.
    """

    print("\n" + "=" * 60)
    print("Trainable Layers")
    print("=" * 60)

    for name, param in model.named_parameters():

        if param.requires_grad:
            print(name)


# ============================================================
# Build Model
# ============================================================

def build_model():
    """
    Build the modified VGGT model for fine-tuning.

    The pretrained model is loaded only once during the
    lifetime of the current Python process.
    """

    global _MODEL

    if _MODEL is not None:

        print("\nUsing cached VGGT model.\n")

        return _MODEL

    print("=" * 60)
    print("Loading Pretrained VGGT")
    print("=" * 60)

    model = VGGT.from_pretrained(PRETRAINED_MODEL)

    model.to(DEVICE)

    freeze_encoder(model)

    unfreeze_segmentation_head(model)

    print_parameter_summary(model)

    print_module_summary(model)

    print_trainable_layers(model)

    _MODEL = model

    return _MODEL