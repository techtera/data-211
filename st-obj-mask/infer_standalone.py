#!/usr/bin/env python3
"""
Standalone Object Mask Inference - No External Dependencies

All model architectures are inlined in this file for easy deployment.

Usage:
    # Single image
    python infer_standalone.py input.jpg output.png
    python infer_standalone.py input.jpg output.png --overlay

    # Batch processing
    python infer_standalone.py input_dir/ output_dir/ --batch
    python infer_standalone.py input_dir/ output_dir/ --batch --overlay

Checkpoints:
    - Encoder: ../kd-encoder/checkpoints_full/student_final.pt
    - Decoder: checkpoints/checkpoint_best.pt
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================================
# Constants
# ============================================================================

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Helper Functions
# ============================================================================

def slice_expand_and_flatten(tensor, B, S):
    """Expand and flatten special tokens for batch processing."""
    # tensor shape: [1, 2, num_tokens, dim]
    first_tokens = tensor[:, 0:1, :, :]  # [1, 1, num_tokens, dim]
    rest_tokens = tensor[:, 1:2, :, :]   # [1, 1, num_tokens, dim]

    first_tokens = first_tokens.expand(B, 1, -1, -1)
    rest_tokens = rest_tokens.expand(B, S-1, -1, -1) if S > 1 else torch.empty(
        B, 0, tensor.shape[2], tensor.shape[3], device=tensor.device
    )

    expanded = torch.cat([first_tokens, rest_tokens], dim=1)
    return expanded.reshape(B * S, tensor.shape[2], tensor.shape[3])


def make_2tuple(x):
    """Convert int to tuple or validate existing tuple."""
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """Create normalized UV grid of shape (width, height, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """Generate 1D sinusoidal positional embedding."""
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(
        embed_dim // 2,
        dtype=torch.float32 if device.type == "mps" else torch.double,
        device=device
    )
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)."""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


# ============================================================================
# Layer Components
# ============================================================================

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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
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
    """MLP (Feed-Forward Network) block."""
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

        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


class PositionGetter:
    """Generates and caches 2D spatial positions for patches."""
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generate spatial positions for a batch of patches."""
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
        self,
        dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frequency components for rotary embeddings."""
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
        """Rotate features for RoPE."""
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Apply 1D rotary position embeddings."""
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply 2D rotary position embeddings."""
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2

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
    """Multi-head self-attention."""
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
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
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
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
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
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None
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

    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    """Transformer block."""
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
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias
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
# Student Encoder
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

        # Register normalization constants
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False
            )

    def forward(self, images: torch.Tensor) -> Tuple[List[Optional[torch.Tensor]], int]:
        """Forward pass through student encoder."""
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        _, P_patches, C = patch_tokens.shape

        # Expand special tokens
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, P, C = tokens.shape

        # Get position embeddings
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1  # Offset patch positions
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Alternating attention
        output_list = []

        for layer_idx in range(self.depth):
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
# SegFormer Decoder Components
# ============================================================================

class MLPSegFormer(nn.Module):
    """Linear embedding for SegFormer decoder."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerDecoder(nn.Module):
    """SegFormer decoder for semantic segmentation."""
    def __init__(
        self,
        in_channels=[256, 512, 1024, 1024],
        embedding_dim=256,
        num_classes=2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Linear embeddings
        self.linear_c1 = MLPSegFormer(in_channels[0], embedding_dim)
        self.linear_c2 = MLPSegFormer(in_channels[1], embedding_dim)
        self.linear_c3 = MLPSegFormer(in_channels[2], embedding_dim)
        self.linear_c4 = MLPSegFormer(in_channels[3], embedding_dim)

        # Fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        B = c1.shape[0]

        # Process C4
        _c4 = self.linear_c4(c4)
        _c4 = _c4.permute(0, 2, 1).reshape(B, self.embedding_dim, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        # Process C3
        _c3 = self.linear_c3(c3)
        _c3 = _c3.permute(0, 2, 1).reshape(B, self.embedding_dim, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        # Process C2
        _c2 = self.linear_c2(c2)
        _c2 = _c2.permute(0, 2, 1).reshape(B, self.embedding_dim, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        # Process C1
        _c1 = self.linear_c1(c1)
        _c1 = _c1.permute(0, 2, 1).reshape(B, self.embedding_dim, c1.shape[2], c1.shape[3])

        # Fuse
        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x


# ============================================================================
# DPT Head (Object Mask Decoder)
# ============================================================================

class DPTHead(nn.Module):
    """DPT-style segmentation head with SegFormer decoder."""
    def __init__(
        self,
        dim_in: int = 1536,
        patch_size: int = 14,
        output_dim: int = 2,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [3, 8, 13, 17],
        pos_embed: bool = True,
    ):
        super(DPTHead, self).__init__()

        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx

        # Layer normalization
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        # SegFormer decoder
        self.segformer_decoder = SegFormerDecoder(
            in_channels=[256, 512, 1024, 1024],
            embedding_dim=256,
            num_classes=output_dim,
        )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ):
        """Forward pass of segmentation head."""
        B, S, _, H, W = images.shape

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Chunked inference
        assert frames_chunk_size > 0
        outputs = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_output = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
            )
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """Internal implementation of segmentation forward pass."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)

            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1

        mask_logits = self.segformer_decoder(out)
        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        mask_logits = mask_logits.view(B, S, -1, H, W)

        return mask_logits

    def _apply_pos_embed(
        self,
        x: torch.Tensor,
        W: int,
        H: int,
        ratio: float = 0.1,
    ) -> torch.Tensor:
        """Add 2D positional embedding to feature map."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]

        pos_embed = create_uv_grid(
            patch_w,
            patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )

        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = (
            pos_embed
            .permute(2, 0, 1)
            [None]
            .expand(x.shape[0], -1, -1, -1)
        )

        return x + pos_embed


# ============================================================================
# Complete Model Wrapper
# ============================================================================

class StudentObjMask(nn.Module):
    """Complete object mask model: encoder + decoder."""
    def __init__(self, encoder: StudentAggregator):
        super().__init__()
        self.encoder = encoder
        self.decoder = DPTHead(
            dim_in=1536,  # 768 frame + 768 global
            patch_size=14,
            output_dim=2,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=[3, 8, 13, 17],
            pos_embed=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] or [B, S, 3, H, W]

        Returns:
            logits: [B, num_classes, H, W] or [B, S, num_classes, H, W]
        """
        # Handle single image input
        if images.ndim == 4:
            images = images.unsqueeze(1)  # Add sequence dimension

        # Encode
        features, patch_start_idx = self.encoder(images)

        # Decode
        logits = self.decoder(features, images, patch_start_idx)

        # Remove sequence dimension if single image
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)

        return logits


# ============================================================================
# Inference Functions
# ============================================================================

def load_checkpoint(
    encoder_path: str = '../kd-encoder/checkpoints_full/student_final.pt',
    decoder_path: str = 'checkpoints/checkpoint_best.pt',
    device: str = 'cuda'
) -> Tuple[StudentObjMask, torch.device]:
    """Load complete model from checkpoints."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print(f"Loading encoder: {encoder_path}")
    student_ckpt = torch.load(encoder_path, map_location='cpu')
    state_dict = student_ckpt.get('student_state_dict', student_ckpt.get('model_state_dict', student_ckpt))

    encoder = StudentAggregator()
    encoder.load_state_dict(state_dict)
    encoder.eval()
    encoder.requires_grad_(False)

    print(f"Loading decoder: {decoder_path}")
    decoder_ckpt = torch.load(decoder_path, map_location='cpu')

    model = StudentObjMask(encoder).to(device)
    model.load_state_dict(decoder_ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded on {device}\n")
    return model, device


def preprocess_image(image_path: str, size: int = 518) -> Tuple[torch.Tensor, tuple, Image.Image]:
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    img_resized = img.resize((size, size), Image.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, original_size, img_resized


def run_inference(
    model: StudentObjMask,
    image_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[np.ndarray, float]:
    """Run inference on preprocessed image."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        start = time.time()
        logits = model(image_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latency = (time.time() - start) * 1000  # ms

        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return mask, latency


def save_mask(mask: np.ndarray, output_path: Path, original_size: tuple = None):
    """Save binary mask as image."""
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode='L')

    if original_size:
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)

    mask_pil.save(output_path)


def create_overlay(img_resized: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Create visualization with mask overlay."""
    overlay = np.array(img_resized).copy()
    red_mask = np.zeros_like(overlay)
    red_mask[mask == 1] = [255, 0, 0]

    result = (overlay * (1 - alpha) + red_mask * alpha).astype(np.uint8)
    return Image.fromarray(result)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone Object Mask Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python infer_standalone.py input.jpg output.png
    python infer_standalone.py input.jpg output.png --overlay

  Batch processing:
    python infer_standalone.py input_dir/ output_dir/ --batch
    python infer_standalone.py input_dir/ output_dir/ --batch --overlay
        """
    )
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('output', help='Output mask path or directory')
    parser.add_argument('--encoder', default='../kd-encoder/checkpoints_full/student_final.pt',
                        help='Path to encoder checkpoint')
    parser.add_argument('--decoder', default='checkpoints/checkpoint_best.pt',
                        help='Path to decoder checkpoint')
    parser.add_argument('--overlay', action='store_true',
                        help='Save visualization overlay')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode (process directory)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Load model
    model, device = load_checkpoint(args.encoder, args.decoder, args.device)

    # Single image mode
    if not args.batch:
        print(f"Processing: {args.input}")
        img_tensor, orig_size, img_resized = preprocess_image(args.input)
        mask, latency = run_inference(model, img_tensor, device)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_mask(mask, output_path, orig_size)

        obj_ratio = (mask == 1).sum() / mask.size * 100
        print(f"  Latency: {latency:.1f}ms")
        print(f"  Object coverage: {obj_ratio:.1f}%")
        print(f"  Saved: {args.output}")

        if args.overlay:
            overlay_path = str(output_path).replace('.png', '_overlay.png')
            overlay_img = create_overlay(img_resized, mask)
            overlay_img.save(overlay_path)
            print(f"  Overlay: {overlay_path}")

    # Batch mode
    else:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(image_files)} images...\n")

        total_latency = 0
        for i, img_path in enumerate(image_files, 1):
            img_tensor, orig_size, img_resized = preprocess_image(img_path)
            mask, latency = run_inference(model, img_tensor, device)
            total_latency += latency

            out_path = output_dir / f"{img_path.stem}_mask.png"
            save_mask(mask, out_path, orig_size)

            if args.overlay:
                overlay_path = output_dir / f"{img_path.stem}_overlay.png"
                overlay_img = create_overlay(img_resized, mask)
                overlay_img.save(overlay_path)

            if i % 10 == 0 or i == len(image_files):
                print(f"  [{i}/{len(image_files)}] {latency:.1f}ms")

        avg_latency = total_latency / len(image_files)
        print(f"\nComplete!")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Throughput: {1000/avg_latency:.1f} images/sec")
        print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
