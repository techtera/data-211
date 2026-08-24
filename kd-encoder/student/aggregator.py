# Copyright (c) Meta Platforms, Inc. and affiliates.
# Student encoder for VGGT knowledge distillation
# Simplified from VGGT teacher for Phase 0A benchmarking

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .layers import Block, PatchEmbed, RotaryPositionEmbedding2D, PositionGetter

# Normalization constants (ImageNet)
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def slice_expand_and_flatten(tensor, B, S):
    """Expand and flatten special tokens for batch processing."""
    # tensor shape: [1, 2, num_tokens, dim]
    # Slice to get first/rest camera tokens
    first_tokens = tensor[:, 0:1, :, :]  # [1, 1, num_tokens, dim]
    rest_tokens = tensor[:, 1:2, :, :]   # [1, 1, num_tokens, dim]

    # Expand for batch
    first_tokens = first_tokens.expand(B, 1, -1, -1)  # [B, 1, num_tokens, dim]
    rest_tokens = rest_tokens.expand(B, S-1, -1, -1) if S > 1 else torch.empty(B, 0, tensor.shape[2], tensor.shape[3], device=tensor.device)

    # Concatenate and flatten
    expanded = torch.cat([first_tokens, rest_tokens], dim=1)  # [B, S, num_tokens, dim]
    return expanded.reshape(B * S, tensor.shape[2], tensor.shape[3])  # [B*S, num_tokens, dim]


class StudentAggregator(nn.Module):
    """
    Student encoder with alternating frame/global attention.

    Architecture:
        - embed_dim: 768 (vs teacher: 1024)
        - depth: 18 layers (vs teacher: 24)
        - num_heads: 12 (vs teacher: 16)
        - cached_layers: [3, 8, 13, 17] (vs teacher: [4, 11, 17, 23])

    Args:
        img_size: Image size in pixels (default: 518)
        patch_size: Patch size (default: 14)
        embed_dim: Token embedding dimension (default: 768)
        depth: Number of transformer layers (default: 18)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        num_register_tokens: Number of register tokens (default: 4)
        qkv_bias: Use bias in QKV projection (default: True)
        proj_bias: Use bias in output projection (default: True)
        ffn_bias: Use bias in FFN (default: True)
        qk_norm: Apply QK normalization (default: True)
        rope_freq: RoPE base frequency, -1 to disable (default: 100)
        init_values: Layer scale init value (default: 0.01)
        cached_layer_indices: Layers to cache for distillation (default: (3, 8, 13, 17))
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=18,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        cached_layer_indices: Tuple[int, ...] = (3, 8, 13, 17),
    ):
        super().__init__()

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        # RoPE (Rotary Position Embedding)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Frame attention blocks
        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        # Global attention blocks
        self.global_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)  # Always cache last layer

        # Special tokens: camera (1) + register (4)
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens  # 5 special tokens

        # Initialize special tokens with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        Forward pass through student encoder.

        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1]
                B = batch size
                S = sequence length (number of frames)
                H, W = height, width (typically 518x518)

        Returns:
            output_list: List of cached layer outputs [B, S, P, 2C]
                         P = number of tokens (1374)
                         2C = 2 * embed_dim (frame + global concatenated)
                         Uncached layers are None
            patch_start_idx: Index where patch tokens start (5)
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape for patch embedding: [B*S, 3, H, W]
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)  # [B*S, P_patches, C]

        _, P_patches, C = patch_tokens.shape

        # Expand special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)  # [B*S, 1, C]
        register_token = slice_expand_and_flatten(self.register_token, B, S)  # [B*S, 4, C]

        # Concatenate: [camera, registers, patches]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)  # [B*S, P, C]
        _, P, C = tokens.shape

        # Get position embeddings if using RoPE
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            # Set position to 0 for special tokens
            pos = pos + 1  # Offset patch positions
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Alternating attention: process frame and global in sequence
        output_list = []

        # Use gradient checkpointing to save memory (trades compute for memory)
        # Only keep activations for cached layers, recompute others during backward
        use_checkpointing = self.training  # Only during training

        for layer_idx in range(self.depth):
            if use_checkpointing and layer_idx not in self.cached_layer_indices:
                # Checkpoint non-cached layers (recompute during backward)
                from torch.utils.checkpoint import checkpoint

                def frame_global_block(tokens_input, pos_input):
                    # Frame attention
                    tokens_f = self.frame_blocks[layer_idx](tokens_input, pos=pos_input)
                    # Global attention
                    tokens_g = tokens_f.view(B, S * P, C)
                    pos_g = pos_input.view(B, S * P, 2) if pos_input is not None else None
                    tokens_g = self.global_blocks[layer_idx](tokens_g, pos=pos_g)
                    return tokens_g.view(B * S, P, C)

                tokens = checkpoint(frame_global_block, tokens, pos, use_reentrant=False)
            else:
                # Normal forward for cached layers (need to cache activations)
                # Frame attention: operates on [B*S, P, C]
                tokens_frame = self.frame_blocks[layer_idx](tokens, pos=pos)

                # Global attention: operates on [B, S*P, C]
                tokens_global = tokens_frame.view(B, S * P, C)
                if pos is not None:
                    pos_global = pos.view(B, S * P, 2)
                else:
                    pos_global = None
                tokens_global = self.global_blocks[layer_idx](tokens_global, pos=pos_global)

                # Reshape back to [B*S, P, C]
                tokens = tokens_global.view(B * S, P, C)

            # Cache outputs if this is a cached layer
            if layer_idx in self.cached_layer_indices:
                # Concatenate frame and global features
                frame_output = tokens_frame.view(B, S, P, C)
                global_output = tokens_global.view(B, S, P, C)
                concat_output = torch.cat([frame_output, global_output], dim=-1)  # [B, S, P, 2C]
                output_list.append(concat_output)
            else:
                output_list.append(None)

        return output_list, self.patch_start_idx
