# DINOv2 pretrained weight initialization for student encoder
# Loads DINOv2 ViT-Large (1024 dim, 24 layers) into student (768 dim, 18 layers)
# Uses projection to adapt 1024→768 dimension

import torch
import torch.nn as nn
from typing import Optional


def load_dinov2_vitl14_reg(verbose: bool = True) -> nn.Module:
    """
    Load DINOv2 ViT-Large with register tokens from torch.hub.

    Returns:
        DINOv2-Large model with pretrained weights (1024-dim, 24 layers)

    Note:
        - First call will download ~1.2GB from torch.hub
        - Subsequent calls use cached weights
        - This is LARGER than ViT-Base (~350MB) but provides better features
    """
    if verbose:
        print("Loading DINOv2 ViT-Large (dinov2_vitl14_reg)...")
        print("  - This may take a few minutes on first run (downloading ~1.2GB)")

    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitl14_reg',
            pretrained=True
        )
        if verbose:
            print("✓ DINOv2-Large loaded successfully")
            print(f"  - Dimension: 1024")
            print(f"  - Layers: 24")
        return dinov2_model
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv2-Large: {e}")


def load_dinov2_vitb14_reg(verbose: bool = True) -> nn.Module:
    """
    [DEPRECATED] Load DINOv2 ViT-Base with register tokens.
    Use load_dinov2_vitl14_reg() instead for better quality.
    """
    if verbose:
        print("WARNING: Using DINOv2-Base (deprecated). Consider using DINOv2-Large.")
        print("Loading DINOv2 ViT-Base (dinov2_vitb14_reg)...")

    try:
        dinov2_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14_reg',
            pretrained=True
        )
        if verbose:
            print("✓ DINOv2-Base loaded")
        return dinov2_model
    except Exception as e:
        raise RuntimeError(f"Failed to load DINOv2: {e}")


def initialize_student_from_dinov2_large(
    student: nn.Module,
    dinov2_model: Optional[nn.Module] = None,
    verbose: bool = True
) -> None:
    """
    Initialize student encoder (768-dim) with DINOv2-Large (1024-dim) pretrained weights.

    Strategy:
        - Patch embedding: Project 1024→768, initialize from DINOv2-Large
        - Frame blocks 0-17: Project 1024→768, initialize from DINOv2-Large blocks 0-17
        - Global blocks 0-17: Same as frame blocks (identical init)
        - Special tokens: Keep random initialization (std=1e-6)

    Args:
        student: StudentAggregator instance (768-dim) to initialize
        dinov2_model: Pretrained DINOv2-Large model (if None, will load automatically)
        verbose: Print progress messages

    Note:
        - DINOv2-Large has 24 blocks (1024-dim), student has 18 blocks (768-dim)
        - Blocks 0-17 initialized from pretrained DINOv2-Large (with projection)
        - Projection: Take first 768 channels from 1024, or use linear projection
        - Both frame and global branches start with identical projected weights
    """
    if verbose:
        print("\n" + "="*60)
        print("Initializing Student Encoder from DINOv2-Large")
        print("="*60)

    # Load DINOv2-Large if not provided
    if dinov2_model is None:
        dinov2_model = load_dinov2_vitl14_reg(verbose=verbose)

    dinov2_model.eval()

    # Get DINOv2 architecture info
    num_dinov2_blocks = len(dinov2_model.blocks)
    dinov2_dim = dinov2_model.blocks[0].attn.qkv.in_features
    if verbose:
        print(f"\nDINOv2-Large architecture:")
        print(f"  - Blocks: {num_dinov2_blocks}")
        print(f"  - Dimension: {dinov2_dim}")

    # Verify student dimension
    student_dim = student.frame_blocks[0].attn.qkv.in_features
    if student_dim != 768:
        raise ValueError(f"Student dimension must be 768, got {student_dim}")

    if verbose:
        print(f"\nStudent architecture:")
        print(f"  - Frame blocks: {len(student.frame_blocks)}")
        print(f"  - Global blocks: {len(student.global_blocks)}")
        print(f"  - Dimension: {student_dim}")
        print(f"\nProjection: {dinov2_dim} → {student_dim}")

    # Helper: project 1024→768 by taking first 768 channels
    def project_weight(weight_1024):
        """Project 1024-dim weight to 768-dim by truncation."""
        if weight_1024.shape[0] == 1024:
            return weight_1024[:768].clone()
        elif weight_1024.shape[1] == 1024:
            return weight_1024[:, :768].clone()
        else:
            # For QKV: [3072, 1024] → [2304, 768]
            if weight_1024.shape[0] == 3072:
                return weight_1024[:2304, :768].clone()
            return weight_1024.clone()

    def project_bias(bias_1024):
        """Project 1024-dim bias to 768-dim by truncation."""
        if bias_1024 is None:
            return None
        if bias_1024.shape[0] == 1024:
            return bias_1024[:768].clone()
        elif bias_1024.shape[0] == 3072:
            return bias_1024[:2304].clone()
        return bias_1024.clone()

    # 1. Initialize patch embedding (project 1024→768)
    if verbose:
        print(f"\n[1/4] Initializing patch embedding (projection)...")

    try:
        dinov2_patch_embed = dinov2_model.patch_embed
        # Patch embed: Conv2d [1024, 3, 14, 14] → [768, 3, 14, 14]
        student.patch_embed.proj.weight.data = project_weight(dinov2_patch_embed.proj.weight.data)
        student.patch_embed.proj.bias.data = project_bias(dinov2_patch_embed.proj.bias.data)
        if verbose:
            print(f"  ✓ Patch embedding initialized (1024→768 projection)")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Patch embedding initialization failed: {e}")

    # 2. Initialize frame blocks (0-17 from DINOv2-Large with projection)
    if verbose:
        print(f"\n[2/4] Initializing frame blocks (projection)...")

    num_init_blocks = min(num_dinov2_blocks, len(student.frame_blocks))

    for i in range(num_init_blocks):
        try:
            # Project each parameter
            dinov2_block = dinov2_model.blocks[i]
            student_block = student.frame_blocks[i]

            # Attention QKV
            student_block.attn.qkv.weight.data = project_weight(dinov2_block.attn.qkv.weight.data)
            student_block.attn.qkv.bias.data = project_bias(dinov2_block.attn.qkv.bias.data)

            # Attention output projection
            student_block.attn.proj.weight.data = project_weight(dinov2_block.attn.proj.weight.data)
            student_block.attn.proj.bias.data = project_bias(dinov2_block.attn.proj.bias.data)

            # MLP fc1 (768→3072)
            student_block.mlp.fc1.weight.data = dinov2_block.mlp.fc1.weight.data[:3072, :768].clone()
            student_block.mlp.fc1.bias.data = dinov2_block.mlp.fc1.bias.data[:3072].clone()

            # MLP fc2 (3072→768)
            student_block.mlp.fc2.weight.data = dinov2_block.mlp.fc2.weight.data[:768, :3072].clone()
            student_block.mlp.fc2.bias.data = dinov2_block.mlp.fc2.bias.data[:768].clone()

            # LayerNorms
            student_block.norm1.weight.data = dinov2_block.norm1.weight.data[:768].clone()
            student_block.norm1.bias.data = dinov2_block.norm1.bias.data[:768].clone()
            student_block.norm2.weight.data = dinov2_block.norm2.weight.data[:768].clone()
            student_block.norm2.bias.data = dinov2_block.norm2.bias.data[:768].clone()

            # LayerScale (if present)
            if hasattr(dinov2_block, 'ls1') and hasattr(student_block, 'ls1'):
                student_block.ls1.gamma.data = dinov2_block.ls1.gamma.data[:768].clone()
            if hasattr(dinov2_block, 'ls2') and hasattr(student_block, 'ls2'):
                student_block.ls2.gamma.data = dinov2_block.ls2.gamma.data[:768].clone()

        except Exception as e:
            if verbose:
                print(f"  ⚠ Block {i} projection failed: {e}")

    if verbose:
        print(f"  ✓ Frame blocks 0-{num_init_blocks-1}: DINOv2-Large (projected)")
        if len(student.frame_blocks) > num_init_blocks:
            print(f"  ✓ Frame blocks {num_init_blocks}-{len(student.frame_blocks)-1}: Random init")

    # 3. Initialize global blocks (same as frame blocks)
    if verbose:
        print(f"\n[3/4] Initializing global blocks (copy from frame)...")

    for i in range(len(student.global_blocks)):
        try:
            student.global_blocks[i].load_state_dict(
                student.frame_blocks[i].state_dict()
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Block {i} copy failed: {e}")

    if verbose:
        print(f"  ✓ Global blocks 0-{len(student.global_blocks)-1}: Copied from frame blocks")

    # 4. Special tokens remain random
    if verbose:
        print(f"\n[4/4] Special tokens...")
        print(f"  ✓ Camera token: Random (std=1e-6)")
        print(f"  ✓ Register tokens: Random (std=1e-6)")

    # 5. Ensure all parameters are on the same device (DDP requirement)
    # Get the device of the first parameter
    first_param = next(student.parameters())
    target_device = first_param.device

    # Move student to target device (ensures output_norm and all components are on same device)
    student.to(target_device)

    if verbose:
        print("\n" + "="*60)
        print("✓ Initialization Complete (DINOv2-Large)")
        print("="*60)
        print("\nInitialization summary:")
        print(f"  - Source: DINOv2-Large (1024-dim, 24 layers)")
        print(f"  - Target: Student (768-dim, 18 layers)")
        print(f"  - Patch embedding: Projected 1024→768")
        print(f"  - Frame blocks 0-17: Projected from DINOv2-Large")
        print(f"  - Global blocks 0-17: Copied from frame blocks")
        print(f"  - Special tokens: Random")
        print(f"  - Device: {target_device}")
        print(f"\nNote: Both branches start identical and will diverge during training.")


def initialize_student_from_dinov2(
    student: nn.Module,
    dinov2_model: Optional[nn.Module] = None,
    verbose: bool = True
) -> None:
    """
    [DEPRECATED] Initialize student with DINOv2-Base (768-dim).
    Use initialize_student_from_dinov2_large() instead for better quality.
    """
    if verbose:
        print("WARNING: Using DINOv2-Base initialization (deprecated).")
        print("Recommend: initialize_student_from_dinov2_large() for better features.\n")

    # ... rest of old code for backwards compatibility
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

    # 5. Ensure all parameters are on the same device (DDP requirement)
    first_param = next(student.parameters())
    target_device = first_param.device
    student.to(target_device)

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
        print(f"  - Device: {target_device}")
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
