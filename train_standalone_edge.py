"""
VGGT Edge Mask — Standalone Training Script (no external file dependencies).

Usage:
    python train_standalone_edge.py --data_dir data/
    python train_standalone_edge.py --data_dir data/ --epochs 100
    python train_standalone_edge.py --data_dir data/ --resume checkpoints/latest_model.pt
    python train_standalone_edge.py --data_dir data/ --model_id facebook/VGGT-1B

Dataset structure:
    data/
    ├── rgb/
    │   ├── abc.png
    │   ├── def.png
    │   └── ...
    └── masks/
        ├── abc_mask.png
        ├── def_mask.png
        └── ...

Requires: local vggt/ directory (HuggingFace model loaded via VGGT.from_pretrained)
"""

import argparse
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.amp import autocast, GradScaler
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 518
BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_MAX_NORM = 1.0
WARMUP_FRACTION = 0.05
VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42
LOG_EVERY = 10
PATIENCE = 15
CHECKPOINT_DIR = "checkpoints"

# Loss config
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
POS_WEIGHT_CLAMP = (5, 25)
DS1_WEIGHT = 0.1
DS2_WEIGHT = 0.2
FINAL_WEIGHT = 1.0


# ============================================================
# Layers: DropPath
# ============================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================================
# Layers: LayerScale
# ============================================================

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ============================================================
# Layers: Mlp
# ============================================================

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: Callable[..., nn.Module] = nn.GELU,
                 drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# Layers: SwiGLU
# ============================================================

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=None,
                 drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=None,
                 drop: float = 0.0, bias: bool = True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


# ============================================================
# Layers: RoPE
# ============================================================

class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions
        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., :feature_dim // 2], x[..., feature_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)


# ============================================================
# Layers: PatchEmbed
# ============================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten_embedding=True):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ============================================================
# Layers: Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 proj_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 norm_layer: nn.Module = nn.LayerNorm, qk_norm: bool = False,
                 fused_attn: bool = True, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
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


# ============================================================
# Layers: Block
# ============================================================

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, proj_bias: bool = True, ffn_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, init_values=None,
                 drop_path: float = 0.0, act_layer: Callable[..., nn.Module] = nn.GELU,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 attn_class: Callable[..., nn.Module] = Attention,
                 ffn_layer: Callable[..., nn.Module] = Mlp,
                 qk_norm: bool = False, fused_attn: bool = True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop,
                               qk_norm=qk_norm, fused_attn=fused_attn, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual_func(x, pos=None):
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(x, pos=pos, residual_func=attn_residual_func,
                                                     sample_drop_ratio=self.sample_drop_ratio)
            x = _drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func,
                                                     sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0, pos=None):
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


# ============================================================
# DinoVisionTransformer
# ============================================================

class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


def _named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        _named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True,
                 proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False,
                 init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU,
                 block_fn=None, ffn_layer="mlp", block_chunks=1,
                 num_register_tokens=0, interpolate_antialias=False,
                 interpolate_offset=0.1, qk_norm=False):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if block_fn is None:
            block_fn = Block

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer_cls = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_layer_cls, init_values=init_values, qk_norm=qk_norm)
            for i in range(depth)
        ]

        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        _named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, **kwargs,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            if self.training:
                x = grad_checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def _vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        block_fn=partial(Block, attn_class=Attention),
        num_register_tokens=num_register_tokens, **kwargs,
    )
    return model


# ============================================================
# Aggregator
# ============================================================

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def _slice_expand_and_flatten(token_tensor, B, S):
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


class Aggregator(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24,
                 num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
                 block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True,
                 patch_embed="dinov2_vitl14_reg", aa_order=None, aa_block_size=1,
                 qk_norm=True, rope_freq=100, init_values=0.01,
                 cached_layer_indices=(4, 11, 17, 23)):
        super().__init__()

        if aa_order is None:
            aa_order = ["frame", "global"]

        self._build_patch_embed(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     init_values=init_values, qk_norm=qk_norm, rope=self.rope)
            for _ in range(depth)
        ])

        self.global_blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     init_values=init_values, qk_norm=qk_norm, rope=self.rope)
            for _ in range(depth)
        ])

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices.add(depth - 1)
        self.aa_block_num = self.depth // self.aa_block_size

        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        self.register_buffer("_resnet_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_resnet_std", torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False)
        self.use_reentrant = False

    def _build_patch_embed(self, patch_embed, img_size, patch_size, num_register_tokens, embed_dim=1024):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            self.patch_embed = _vit_large(
                img_size=img_size, patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=True, interpolate_offset=0.0,
                block_chunks=0, init_values=1.0,
            )
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor):
        B, S, C_in, H, W = images.shape
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        _, P, C = patch_tokens.shape

        camera_token = _slice_expand_and_flatten(self.camera_token, B, S)
        register_token = _slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)

            for i in range(len(frame_intermediates)):
                layer_idx = len(output_list)
                if layer_idx in self.cached_layer_indices:
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                else:
                    output_list.append(None)

        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = grad_checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = grad_checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates


# ============================================================
# Feature Extractor
# ============================================================

class FeatureProjection(nn.Module):
    def __init__(self, in_ch=2048, out_ch=64, target_size=None, downsample=False):
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


class VGGTFeatureExtractor(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)
        self.layer_indices = [4, 11, 17, 23]
        self.patch_start_idx = 5
        self.projections = nn.ModuleList([
            FeatureProjection(2048, 64, target_size=(148, 148)),
            FeatureProjection(2048, 128, target_size=(74, 74)),
            FeatureProjection(2048, 256),
            FeatureProjection(2048, 512, downsample=True),
        ])

    def forward(self, images):
        B, S = images.shape[:2]
        with torch.no_grad():
            aggregated_tokens_list, _ = self.aggregator(images)
        features = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx]
            x = x[:, :, self.patch_start_idx:]
            x = x.reshape(B * S, x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1)
            x = x.reshape(B * S, 2048, 37, 37)
            x = x.detach()
            x = self.projections[i](x)
            features.append(x)
        return features


# ============================================================
# UNet++ Decoder
# ============================================================

class ConvBlock(nn.Module):
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
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return x


class UNetPPDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = channels
        self.up_3_0 = Upsample(c3, c2)
        self.up_2_0 = Upsample(c2, c1)
        self.up_2_1 = Upsample(c2, c1)
        self.up_1_0 = Upsample(c1, c0)
        self.up_1_1 = Upsample(c1, c0)
        self.up_1_2 = Upsample(c1, c0)
        self.conv_2_1 = ConvBlock(c2 + c2, c2)
        self.conv_1_1 = ConvBlock(c1 + c1, c1)
        self.conv_1_2 = ConvBlock(c1 * 3, c1)
        self.conv_0_1 = ConvBlock(c0 + c0, c0)
        self.conv_0_2 = ConvBlock(c0 * 3, c0)
        self.conv_0_3 = ConvBlock(c0 * 4, c0)
        self.ds1 = DeepSupervisionHead(c0)
        self.ds2 = DeepSupervisionHead(c0)

    def forward(self, features):
        x_0_0, x_1_0, x_2_0, x_3_0 = features
        size_0 = x_0_0.shape[2:]
        size_1 = x_1_0.shape[2:]
        size_2 = x_2_0.shape[2:]
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up_3_0(x_3_0, size_2)], dim=1))
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.up_2_0(x_2_0, size_1)], dim=1))
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, self.up_2_1(x_2_1, size_1)], dim=1))
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, self.up_1_0(x_1_0, size_0)], dim=1))
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, self.up_1_1(x_1_1, size_0)], dim=1))
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up_1_2(x_1_2, size_0)], dim=1))
        ds1_out = self.ds1(x_0_1)
        ds2_out = self.ds2(x_0_2)
        return x_0_3, ds1_out, ds2_out


# ============================================================
# Edge Refinement
# ============================================================

class EdgeRefinement(nn.Module):
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


# ============================================================
# Full Model: VGGTEdgeMask
# ============================================================

class VGGTEdgeMask(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.feature_extractor = VGGTFeatureExtractor(aggregator)
        self.decoder = UNetPPDecoder(channels=(64, 128, 256, 512))
        self.refinement = EdgeRefinement(ch=64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, images):
        B, S = images.shape[:2]
        features = self.feature_extractor(images)
        x_0_3, ds1_logits, ds2_logits = self.decoder(features)
        x = self.refinement(x_0_3)
        logits = self.final_conv(x)
        logits = F.interpolate(logits, size=(518, 518), mode="bilinear", align_corners=False)
        logits = logits.view(B, S, 1, 518, 518)
        ds1_logits = ds1_logits.view(B, S, 1, 518, 518)
        ds2_logits = ds2_logits.view(B, S, 1, 518, 518)
        if self.training:
            return logits, ds1_logits, ds2_logits
        else:
            return torch.sigmoid(logits)


# ============================================================
# Dataset
# ============================================================

class EdgeMaskDataset(Dataset):
    """
    Loads RGB images and their corresponding binary edge masks.

    Expected structure:
        data/
        ├── rgb/   (png/jpg)
        └── masks/ (abc_mask.png for each abc.png in rgb/)
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.mask_dir = self.data_dir / "masks"

        rgb_files = os.listdir(self.rgb_dir)
        mask_files = os.listdir(self.mask_dir)

        mask_stems = set()
        for f in mask_files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                stem = Path(f).stem
                if stem.endswith("_mask"):
                    mask_stems.add(stem[:-5])

        all_images = sorted([
            f for f in rgb_files
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.image_names = [
            f for f in all_images
            if Path(f).stem in mask_stems
        ]

        print(f"  Valid pairs: {len(self.image_names)}")

        self.rgb_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        stem = Path(img_name).stem
        suffix = Path(img_name).suffix

        rgb_path = self.rgb_dir / img_name
        mask_path = self.mask_dir / f"{stem}_mask{suffix}"

        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        rgb = self.rgb_transform(rgb)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        rgb = rgb.unsqueeze(0)    # [1, 3, 518, 518]
        mask = mask.unsqueeze(0)  # [1, 1, 518, 518]

        return rgb, mask


# ============================================================
# Loss Functions
# ============================================================

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_weight = BCE_WEIGHT
        self.dice_weight = DICE_WEIGHT
        self.pos_weight_min = POS_WEIGHT_CLAMP[0]
        self.pos_weight_max = POS_WEIGHT_CLAMP[1]

    def forward(self, pred_logits, target):
        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(self.pos_weight_min, self.pos_weight_max)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target,
            pos_weight=pos_weight.view(1, 1, 1, 1).expand_as(pred_logits),
        )

        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6))

        return self.bce_weight * bce + self.dice_weight * dice


def compute_total_loss(final_logits, ds1_logits, ds2_logits, target, loss_fn):
    loss_final = loss_fn(final_logits, target)
    loss_ds1 = loss_fn(ds1_logits, target)
    loss_ds2 = loss_fn(ds2_logits, target)
    return FINAL_WEIGHT * loss_final + DS2_WEIGHT * loss_ds2 + DS1_WEIGHT * loss_ds1


# ============================================================
# Learning Rate Scheduler
# ============================================================

def build_scheduler(optimizer, total_steps):
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler, warmup_steps


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.train()
    model.feature_extractor.aggregator.eval()

    total_loss = 0.0
    total_samples = 0
    total_edge_ratio = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)
        total_samples += batch_size

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits, ds1_logits, ds2_logits = model(images)
            loss = compute_total_loss(logits, ds1_logits, ds2_logits, masks, criterion)

        preds = torch.sigmoid(logits)
        edge_ratio = (preds > 0.5).float().mean().item()

        total_loss += loss.item() * batch_size
        total_edge_ratio += edge_ratio * batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_edge_ratio = total_edge_ratio / max(total_samples, 1)

    return {"loss": avg_loss, "edge_ratio": avg_edge_ratio}


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, filepath):
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"  Resumed from epoch {epoch}, loss={loss:.4f}")
    return epoch, loss


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, epoch, device):
    model.train()
    model.feature_extractor.aggregator.eval()

    running_loss = 0.0
    max_grad_norm = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits, ds1_logits, ds2_logits = model(images)
            loss = compute_total_loss(logits, ds1_logits, ds2_logits, masks, criterion)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        max_grad_norm = max(max_grad_norm, grad_norm.item())

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Batch [{batch_idx + 1:04d}/{len(dataloader):04d}] "
                  f"Loss: {loss.item():.4f}  LR: {lr:.2e}  Grad: {grad_norm.item():.2f}")

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, max_grad_norm


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, num_epochs, start_epoch=0):
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch + 1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        train_loss, max_grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch, device,
        )

        val_results = validate(model, val_loader, criterion, device)
        val_loss = val_results["loss"]
        edge_ratio = val_results["edge_ratio"]

        # Collapse detection
        if epoch >= 10 and edge_ratio < 1e-4:
            print("  [WARNING] Possible all-zero collapse! edge_ratio ~ 0")

        # Save latest
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss,
                        checkpoint_dir / "latest_model.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss,
                            checkpoint_dir / "best_model.pt")
            print(f"  New best val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Epoch summary
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n  Train Loss       : {train_loss:.4f}")
        print(f"  Val Loss         : {val_loss:.4f}")
        print(f"  Edge Ratio       : {edge_ratio:.4f}")
        print(f"  Learning Rate    : {lr:.2e}")
        print(f"  Max Grad Norm    : {max_grad_norm:.2f}")
        print(f"  Best Val Loss    : {best_val_loss:.4f}")
        print(f"  Patience         : {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Verify encoder stayed frozen
    encoder_has_grad = any(p.grad is not None for p in model.feature_extractor.aggregator.parameters())
    print(f"\n{'=' * 60}")
    print("Training Complete")
    print(f"{'=' * 60}")
    print(f"  Best Val Loss    : {best_val_loss:.4f}")
    print(f"  Encoder Frozen   : {'YES' if not encoder_has_grad else 'NO - ERROR!'}")

    return best_val_loss


# ============================================================
# Model Loading
# ============================================================

def build_model_from_hf(model_id, device):
    """
    Load pretrained VGGT encoder from HuggingFace Hub, build the
    VGGTEdgeMask model, and return it (encoder frozen, decoder trainable).
    """
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vggt_path = os.path.join(script_dir, "vggt")
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)
    from vggt.models.vggt import VGGT

    print(f"Loading pretrained VGGT from HuggingFace: {model_id}...")
    pretrained = VGGT.from_pretrained(model_id)
    print("  VGGT loaded successfully.")

    print("Building VGGTEdgeMask model...")
    model = VGGTEdgeMask(pretrained.aggregator)
    del pretrained

    model.to(device)
    return model


def build_model_from_checkpoint(checkpoint_path, device):
    """
    Load full model from a previously fine-tuned checkpoint (best_model.pt).
    Creates fresh optimizer/scheduler — only model weights are restored.
    """
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vggt_path = os.path.join(script_dir, "vggt")
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)
    from vggt.models.vggt import VGGT

    print(f"Loading base VGGT for aggregator architecture...")
    pretrained = VGGT.from_pretrained("facebook/VGGT-1B")

    print("Building VGGTEdgeMask model...")
    model = VGGTEdgeMask(pretrained.aggregator)
    del pretrained

    print(f"Loading fine-tuned checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", "?")
    print(f"  Checkpoint epoch: {epoch}, loss: {loss}")

    model.to(device)
    return model


def print_param_summary(model):
    """Print parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"  Total Parameters      : {total:,}")
    print(f"  Trainable Parameters  : {trainable:,}")
    print(f"  Frozen Parameters     : {frozen:,}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VGGT Edge Mask — Standalone Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (with rgb/ and masks/)")
    parser.add_argument("--model_id", type=str, default="facebook/VGGT-1B",
                        help="HuggingFace model ID (default: facebook/VGGT-1B)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a fine-tuned checkpoint to start from (skips fresh init)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to training checkpoint to resume (restores optimizer/scheduler too)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Build model (3 modes)
    if args.checkpoint:
        # Mode 2: Fine-tune from a previous checkpoint (fresh optimizer)
        model = build_model_from_checkpoint(args.checkpoint, device)
    else:
        # Mode 1: Fresh from HuggingFace
        model = build_model_from_hf(args.model_id, device)

    print_param_summary(model)

    # Build dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = EdgeMaskDataset(args.data_dir)

    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * VALIDATION_SPLIT))
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    print(f"  Train samples: {train_size}, Val samples: {val_size}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build optimizer, scheduler, scaler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps = args.epochs * len(train_loader)
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"  Optimizer: AdamW (lr={args.lr}, wd={WEIGHT_DECAY})")
    print(f"  Scheduler: Cosine + {warmup_steps} warmup steps / {total_steps} total")

    # Build loss
    criterion = EdgeLoss().to(device)

    # Mode 3: Resume full training state (optimizer + scheduler + scaler)
    start_epoch = 0
    if args.resume:
        print(f"\nResuming full training state from {args.resume}...")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, scaler, args.resume, device)

    # Train
    print(f"\nStarting training for {args.epochs} epochs...\n")
    best_val_loss = train(
        model, train_loader, val_loader, criterion,
        optimizer, scheduler, scaler, device, args.epochs, start_epoch,
    )

    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
