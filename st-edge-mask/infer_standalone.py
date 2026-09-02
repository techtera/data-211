#!/usr/bin/env python3
"""
Standalone Edge Mask Inference Script
NO external project imports - all architecture code inlined

Usage:
    Single image:  python infer_standalone.py input.jpg output.png --threshold 0.5 --overlay
    Batch mode:    python infer_standalone.py input_dir/ output_dir/ --batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, List
from pathlib import Path
import argparse
import numpy as np
from PIL import Image


# ============================================================================
# CONSTANTS AND HELPER FUNCTIONS
# ============================================================================

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def make_2tuple(x):
    """Convert int to tuple of length 2."""
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


def slice_expand_and_flatten(tensor, B, S):
    """Expand and flatten special tokens for batch processing."""
    # tensor shape: [1, 2, num_tokens, dim]
    first_tokens = tensor[:, 0:1, :, :]  # [1, 1, num_tokens, dim]
    rest_tokens = tensor[:, 1:2, :, :]   # [1, 1, num_tokens, dim]

    # Expand for batch
    first_tokens = first_tokens.expand(B, 1, -1, -1)  # [B, 1, num_tokens, dim]
    rest_tokens = rest_tokens.expand(B, S-1, -1, -1) if S > 1 else torch.empty(
        B, 0, tensor.shape[2], tensor.shape[3], device=tensor.device
    )

    # Concatenate and flatten
    expanded = torch.cat([first_tokens, rest_tokens], dim=1)  # [B, S, num_tokens, dim]
    return expanded.reshape(B * S, tensor.shape[2], tensor.shape[3])  # [B*S, num_tokens, dim]


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


# ============================================================================
# LAYER COMPONENTS (IN DEPENDENCY ORDER)
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    """Layer scale for training stability."""

    def __init__(self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """MLP (Feed-forward network) module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ):
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input height {H} not divisible by patch height {patch_H}"
        assert W % patch_W == 0, f"Input width {W} not divisible by patch width {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


class PositionGetter:
    """Generates and caches 2D spatial positions for patches."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Returns: Tensor (batch_size, height*width, 2) with y,x coordinates."""
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding (RoPE)."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings."""
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation."""
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings."""
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D RoPE to tokens (batch_size, n_heads, n_tokens, dim)."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions: (batch, tokens, 2)"

        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        return torch.cat((vertical_features, horizontal_features), dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_add_residual_stochastic_depth(
    x: torch.Tensor, residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0, pos=None
) -> torch.Tensor:
    """Drop paths with residual addition."""
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    if pos is not None:
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size

    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor, pos=None) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = x + drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = x + drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


# ============================================================================
# STUDENT ENCODER (255M PARAMETERS)
# ============================================================================

class StudentAggregator(nn.Module):
    """Student encoder with alternating frame/global attention."""

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

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        # RoPE
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

        # Special tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize special tokens
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Normalization constants
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        """
        Args:
            images: [B, S, 3, H, W] in range [0, 1]

        Returns:
            output_list: List of cached layer outputs [B, S, P, 2C]
            patch_start_idx: Index where patch tokens start (5)
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 channels, got {C_in}")

        # Normalize
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)  # [B*S, P_patches, C]

        _, P_patches, C = patch_tokens.shape

        # Expand special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, P, C = tokens.shape

        # Position embeddings
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Alternating attention
        output_list = []
        use_checkpointing = self.training

        for layer_idx in range(self.depth):
            if use_checkpointing and layer_idx not in self.cached_layer_indices:
                from torch.utils.checkpoint import checkpoint

                def frame_global_block(tokens_input, pos_input):
                    tokens_f = self.frame_blocks[layer_idx](tokens_input, pos=pos_input)
                    tokens_g = tokens_f.view(B, S * P, C)
                    pos_g = pos_input.view(B, S * P, 2) if pos_input is not None else None
                    tokens_g = self.global_blocks[layer_idx](tokens_g, pos=pos_g)
                    return tokens_g.view(B * S, P, C)

                tokens = checkpoint(frame_global_block, tokens, pos, use_reentrant=False)
            else:
                # Frame attention
                tokens_frame = self.frame_blocks[layer_idx](tokens, pos=pos)

                # Global attention
                tokens_global = tokens_frame.view(B, S * P, C)
                if pos is not None:
                    pos_global = pos.view(B, S * P, 2)
                else:
                    pos_global = None
                tokens_global = self.global_blocks[layer_idx](tokens_global, pos=pos_global)

                # Reshape back
                tokens = tokens_global.view(B * S, P, C)

            # Cache outputs
            if layer_idx in self.cached_layer_indices:
                frame_output = tokens_frame.view(B, S, P, C)
                global_output = tokens_global.view(B, S, P, C)
                concat_output = torch.cat([frame_output, global_output], dim=-1)
                output_list.append(concat_output)
            else:
                output_list.append(None)

        return output_list, self.patch_start_idx


# ============================================================================
# EDGE MASK DECODER COMPONENTS
# ============================================================================

class ConvBlock(nn.Module):
    """Convolution block with GroupNorm and SiLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    """Upsample block with interpolation and convolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class DeepSupervisionHead(nn.Module):
    """Deep supervision head for intermediate outputs."""

    def __init__(self, in_ch=64, output_size=518):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.head(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)
        return x


class UNetPPDecoder(nn.Module):
    """UNet++ decoder architecture."""

    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = channels

        # Upsample blocks
        self.up_3_0 = Upsample(c3, c2)
        self.up_2_0 = Upsample(c2, c1)
        self.up_2_1 = Upsample(c2, c1)
        self.up_1_0 = Upsample(c1, c0)
        self.up_1_1 = Upsample(c1, c0)
        self.up_1_2 = Upsample(c1, c0)

        # ConvBlocks
        self.conv_2_1 = ConvBlock(c2 + c2, c2)
        self.conv_1_1 = ConvBlock(c1 + c1, c1)
        self.conv_1_2 = ConvBlock(c1 * 3, c1)
        self.conv_0_1 = ConvBlock(c0 + c0, c0)
        self.conv_0_2 = ConvBlock(c0 * 3, c0)
        self.conv_0_3 = ConvBlock(c0 * 4, c0)

        # Deep supervision
        self.ds1 = DeepSupervisionHead(c0)
        self.ds2 = DeepSupervisionHead(c0)

    def forward(self, features):
        x_0_0, x_1_0, x_2_0, x_3_0 = features

        size_0 = x_0_0.shape[2:]
        size_1 = x_1_0.shape[2:]
        size_2 = x_2_0.shape[2:]

        # Build UNet++ structure
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up_3_0(x_3_0, size_2)], dim=1))
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.up_2_0(x_2_0, size_1)], dim=1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, self.up_2_1(x_2_1, size_1)], dim=1))
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, self.up_1_0(x_1_0, size_0)], dim=1))
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, self.up_1_1(x_1_1, size_0)], dim=1))
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up_1_2(x_1_2, size_0)], dim=1))

        # Deep supervision
        ds1_out = self.ds1(x_0_1)
        ds2_out = self.ds2(x_0_2)

        return x_0_3, ds1_out, ds2_out


class EdgeRefinement(nn.Module):
    """Edge refinement with residual connection."""

    def __init__(self, ch=64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.refine(x)


class FeatureProjection(nn.Module):
    """Project encoder features to decoder dimensions."""

    def __init__(self, in_ch=1536, out_ch=64, target_size=None, downsample=False):
        super().__init__()
        self.target_size = target_size
        self.downsample = downsample

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

        if target_size is not None:
            self.resize = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            )
        elif downsample:
            self.resize = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            )
        else:
            self.resize = None

    def forward(self, x):
        x = self.proj(x)
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
            x = self.resize(x)
        elif self.resize is not None:
            x = self.resize(x)
        return x


class StudentFeatureExtractor(nn.Module):
    """Feature extractor for Student encoder."""

    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        self.layer_indices = [3, 8, 13, 17]
        self.patch_start_idx = 5

        self.projections = nn.ModuleList([
            FeatureProjection(1536, 64, target_size=(148, 148)),
            FeatureProjection(1536, 128, target_size=(74, 74)),
            FeatureProjection(1536, 256),
            FeatureProjection(1536, 512, downsample=True),
        ])

    def forward(self, images):
        B, S = images.shape[:2]

        with torch.no_grad():
            aggregated_tokens_list, _ = self.aggregator(images)

        features = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Layer {layer_idx} returned None")
            x = x[:, :, self.patch_start_idx:]
            x = x.reshape(B * S, x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1)
            x = x.reshape(B * S, 1536, 37, 37)
            x = x.detach()
            x = self.projections[i](x)
            features.append(x)

        return features


# ============================================================================
# COMPLETE EDGE MASK MODEL
# ============================================================================

class StudentEdgeMask(nn.Module):
    """Edge mask prediction using Student encoder."""

    def __init__(self, student_aggregator):
        super().__init__()
        self.feature_extractor = StudentFeatureExtractor(student_aggregator)
        self.decoder = UNetPPDecoder(channels=(64, 128, 256, 512))
        self.refinement = EdgeRefinement(ch=64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, images):
        """
        Args:
            images: [B, 3, 518, 518] or [B, S, 3, 518, 518] in [0, 1]

        Returns:
            training: logits, ds1_logits, ds2_logits
            eval: sigmoid(logits)
        """
        # Handle 4D and 5D inputs
        if images.ndim == 4:
            images = images.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, S = images.shape[:2]

        features = self.feature_extractor(images)
        x_0_3, ds1_logits, ds2_logits = self.decoder(features)
        x = self.refinement(x_0_3)

        logits = self.final_conv(x)
        logits = F.interpolate(logits, size=(518, 518), mode="bilinear", align_corners=False)

        logits = logits.view(B, S, 1, 518, 518)
        ds1_logits = ds1_logits.view(B, S, 1, 518, 518)
        ds2_logits = ds2_logits.view(B, S, 1, 518, 518)

        if squeeze_output:
            logits = logits.squeeze(1)
            ds1_logits = ds1_logits.squeeze(1)
            ds2_logits = ds2_logits.squeeze(1)

        if self.training:
            return logits, ds1_logits, ds2_logits
        else:
            return torch.sigmoid(logits)


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def load_model(encoder_path: Path, decoder_path: Path, device: str = "cuda") -> StudentEdgeMask:
    """Load student encoder and edge mask decoder."""
    print(f"Loading student encoder from: {encoder_path}")
    encoder_ckpt = torch.load(encoder_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in encoder_ckpt:
        encoder_state = encoder_ckpt["model_state_dict"]
    elif "model" in encoder_ckpt:
        encoder_state = encoder_ckpt["model"]
    else:
        encoder_state = encoder_ckpt

    # Create encoder
    encoder = StudentAggregator(
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
        cached_layer_indices=(3, 8, 13, 17),
    )
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()

    print(f"Loading edge mask decoder from: {decoder_path}")
    decoder_ckpt = torch.load(decoder_path, map_location=device)

    # Create full model
    model = StudentEdgeMask(encoder)

    # Load decoder weights
    if "model_state_dict" in decoder_ckpt:
        decoder_state = decoder_ckpt["model_state_dict"]
    elif "model" in decoder_ckpt:
        decoder_state = decoder_ckpt["model"]
    else:
        decoder_state = decoder_ckpt

    # Load only decoder weights (encoder is already loaded)
    model_dict = model.state_dict()
    decoder_only = {k: v for k, v in decoder_state.items() if not k.startswith("feature_extractor.aggregator")}
    model_dict.update(decoder_only)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def load_image(image_path: Path, size: int = 518) -> torch.Tensor:
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")

    # Resize to 518x518
    img = img.resize((size, size), Image.BILINEAR)

    # Convert to tensor [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]

    return img_tensor


def save_mask(mask: torch.Tensor, output_path: Path, threshold: float = 0.5):
    """Save binary mask as PNG."""
    # Apply threshold
    binary_mask = (mask > threshold).float()

    # Convert to numpy
    mask_np = (binary_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Save
    Image.fromarray(mask_np).save(output_path)


def save_overlay(image_path: Path, mask: torch.Tensor, output_path: Path, threshold: float = 0.5):
    """Save image with mask overlay."""
    # Load original image
    img = Image.open(image_path).convert("RGB").resize((518, 518))
    img_array = np.array(img)

    # Create binary mask
    binary_mask = (mask > threshold).squeeze().cpu().numpy()

    # Create colored overlay (red edges)
    overlay = img_array.copy()
    overlay[binary_mask > 0.5] = [255, 0, 0]

    # Blend
    blended = (0.6 * img_array + 0.4 * overlay).astype(np.uint8)

    # Save
    Image.fromarray(blended).save(output_path)


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Standalone edge mask inference")
    parser.add_argument("input", type=str, help="Input image or directory")
    parser.add_argument("output", type=str, help="Output path or directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Edge threshold (0-1)")
    parser.add_argument("--batch", action="store_true", help="Batch processing mode")
    parser.add_argument("--overlay", action="store_true", help="Save overlay image")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default="../kd-encoder/checkpoints/student_final.pt",
        help="Path to student encoder checkpoint"
    )
    parser.add_argument(
        "--decoder-checkpoint",
        type=str,
        default="checkpoints/checkpoint_best.pt",
        help="Path to edge mask decoder checkpoint"
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    encoder_path = Path(args.encoder_checkpoint)
    decoder_path = Path(args.decoder_checkpoint)

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    model = load_model(encoder_path, decoder_path, device)

    # Process images
    if args.batch:
        # Batch mode
        if not input_path.is_dir():
            raise ValueError("Input must be a directory in batch mode")

        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        print(f"Processing {len(image_files)} images...")

        for img_file in image_files:
            print(f"Processing: {img_file.name}")

            # Load and predict
            img_tensor = load_image(img_file).unsqueeze(0).to(device)

            with torch.no_grad():
                mask_pred = model(img_tensor)

            # Save results
            output_file = output_path / f"{img_file.stem}_edge.png"
            save_mask(mask_pred, output_file, args.threshold)

            if args.overlay:
                overlay_file = output_path / f"{img_file.stem}_overlay.png"
                save_overlay(img_file, mask_pred, overlay_file, args.threshold)

        print(f"Done! Saved to {output_path}")

    else:
        # Single image mode
        if not input_path.is_file():
            raise ValueError("Input must be an image file")

        print(f"Processing: {input_path}")

        # Load and predict
        img_tensor = load_image(input_path).unsqueeze(0).to(device)

        with torch.no_grad():
            mask_pred = model(img_tensor)

        # Save results
        save_mask(mask_pred, output_path, args.threshold)
        print(f"Saved mask to: {output_path}")

        if args.overlay:
            overlay_path = output_path.parent / f"{output_path.stem}_overlay.png"
            save_overlay(input_path, mask_pred, overlay_path, args.threshold)
            print(f"Saved overlay to: {overlay_path}")


if __name__ == "__main__":
    main()
