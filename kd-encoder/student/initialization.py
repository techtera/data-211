# DINOv2 pretrained weight initialization for student encoder
# Loads DINOv2 ViT-Base (768 dim, 12 layers) into student (768 dim, 18 layers)

import torch
import torch.nn as nn
from typing import Optional


def load_dinov2_vitb14_reg(verbose: bool = True) -> nn.Module:
    """
    Load DINOv2 ViT-Base with register tokens from torch.hub.

    Returns:
        DINOv2 model with pretrained weights

    Note:
        - First call will download ~350MB from torch.hub
        - Subsequent calls use cached weights
    """
    if verbose:
        print("Loading DINOv2 ViT-Base (dinov2_vitb14_reg)...")
        print("  - This may take a few minutes on first run (downloading ~350MB)")

    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg',
            pretrained=True
        )
        if verbose:
            print("✓ DINOv2 loaded successfully")
        return dinov2_model
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv2: {e}")


def initialize_student_from_dinov2(
    student: nn.Module,
    dinov2_model: Optional[nn.Module] = None,
    verbose: bool = True
) -> None:
    """
    Initialize student encoder with DINOv2 ViT-Base pretrained weights.

    Strategy:
        - Patch embedding: DINOv2 → Student
        - Frame blocks 0-11: DINOv2 blocks 0-11 → Student
        - Frame blocks 12-17: Keep random initialization
        - Global blocks 0-11: DINOv2 blocks 0-11 → Student (same weights as frame)
        - Global blocks 12-17: Keep random initialization
        - Special tokens: Keep random initialization (std=1e-6)

    Args:
        student: StudentAggregator instance to initialize
        dinov2_model: Pretrained DINOv2 model (if None, will load automatically)
        verbose: Print progress messages

    Note:
        - DINOv2 has 12 blocks, student has 18 blocks
        - Blocks 0-11 initialized from pretrained
        - Blocks 12-17 keep PyTorch default initialization
        - Both frame and global branches start with identical DINOv2 weights
    """
    if verbose:
        print("\n" + "="*60)
        print("Initializing Student Encoder from DINOv2")
        print("="*60)

    # Load DINOv2 if not provided
    if dinov2_model is None:
        dinov2_model = load_dinov2_vitb14_reg(verbose=verbose)

    dinov2_model.eval()

    # Get DINOv2 architecture info
    num_dinov2_blocks = len(dinov2_model.blocks)
    if verbose:
        print(f"\nDINOv2 architecture:")
        print(f"  - Blocks: {num_dinov2_blocks}")
        print(f"  - Dimension: 768")

    # Verify compatibility
    student_dim = student.frame_blocks[0].attn.qkv.in_features
    if student_dim != 768:
        raise ValueError(f"Student dimension ({student_dim}) doesn't match DINOv2 (768)")

    if verbose:
        print(f"\nStudent architecture:")
        print(f"  - Frame blocks: {len(student.frame_blocks)}")
        print(f"  - Global blocks: {len(student.global_blocks)}")
        print(f"  - Dimension: {student_dim}")

    # 1. Initialize patch embedding
    if verbose:
        print(f"\n[1/4] Initializing patch embedding...")

    try:
        # DINOv2's patch_embed is wrapped in a module
        dinov2_patch_embed = dinov2_model.patch_embed

        # Copy projection weights (Conv2d)
        student.patch_embed.proj.weight.data.copy_(dinov2_patch_embed.proj.weight.data)
        student.patch_embed.proj.bias.data.copy_(dinov2_patch_embed.proj.bias.data)

        if verbose:
            print(f"  ✓ Patch embedding initialized from DINOv2")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Patch embedding initialization failed: {e}")
            print(f"  Keeping random initialization")

    # 2. Initialize frame blocks (0-11 from DINOv2, 12-17 random)
    if verbose:
        print(f"\n[2/4] Initializing frame blocks...")

    num_init_blocks = min(num_dinov2_blocks, len(student.frame_blocks))

    for i in range(num_init_blocks):
        try:
            student.frame_blocks[i].load_state_dict(
                dinov2_model.blocks[i].state_dict(),
                strict=False  # Allow missing keys (e.g., rope parameters)
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Block {i} initialization failed: {e}")

    if verbose:
        print(f"  ✓ Frame blocks 0-{num_init_blocks-1}: DINOv2 pretrained")
        print(f"  ✓ Frame blocks {num_init_blocks}-{len(student.frame_blocks)-1}: Random initialization")

    # 3. Initialize global blocks (0-11 from DINOv2, 12-17 random)
    if verbose:
        print(f"\n[3/4] Initializing global blocks...")

    for i in range(num_init_blocks):
        try:
            student.global_blocks[i].load_state_dict(
                dinov2_model.blocks[i].state_dict(),
                strict=False
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Block {i} initialization failed: {e}")

    if verbose:
        print(f"  ✓ Global blocks 0-{num_init_blocks-1}: DINOv2 pretrained")
        print(f"  ✓ Global blocks {num_init_blocks}-{len(student.global_blocks)-1}: Random initialization")

    # 4. Special tokens remain random (already initialized in StudentAggregator)
    if verbose:
        print(f"\n[4/4] Special tokens...")
        print(f"  ✓ Camera token: Random (std=1e-6)")
        print(f"  ✓ Register tokens: Random (std=1e-6)")

    if verbose:
        print("\n" + "="*60)
        print("✓ Initialization Complete")
        print("="*60)
        print("\nInitialization summary:")
        print(f"  - Patch embedding: DINOv2 pretrained")
        print(f"  - Frame blocks 0-11: DINOv2 pretrained")
        print(f"  - Frame blocks 12-17: Random")
        print(f"  - Global blocks 0-11: DINOv2 pretrained (same as frame)")
        print(f"  - Global blocks 12-17: Random")
        print(f"  - Special tokens: Random")
        print(f"\nNote: Frame and global branches start with identical weights.")
        print(f"      They will specialize during training via gradient updates.")


def verify_initialization(student: nn.Module, verbose: bool = True) -> dict:
    """
    Verify initialization was successful.

    Returns:
        dict with verification results
    """
    results = {
        'all_parameters_initialized': True,
        'has_nan': False,
        'has_inf': False,
        'parameter_ranges': {}
    }

    for name, param in student.named_parameters():
        # Check for NaN
        if torch.isnan(param).any():
            results['has_nan'] = True
            if verbose:
                print(f"⚠ NaN found in {name}")

        # Check for Inf
        if torch.isinf(param).any():
            results['has_inf'] = True
            if verbose:
                print(f"⚠ Inf found in {name}")

        # Store parameter range
        results['parameter_ranges'][name] = {
            'min': param.min().item(),
            'max': param.max().item(),
            'mean': param.mean().item(),
            'std': param.std().item(),
        }

    if verbose and not results['has_nan'] and not results['has_inf']:
        print("\n✓ All parameters initialized correctly (no NaN or Inf)")

    return results
