"""
VGGT Edge Mask — Standalone Inference Script (no external file dependencies).

Usage:
    python inference_standalone_edge.py --image path/to/image.png --checkpoint best_model.pt
    python inference_standalone_edge.py --image img1.png img2.png --checkpoint best_model.pt
    python inference_standalone_edge.py --image_dir path/to/images/ --checkpoint best_model.pt
    python inference_standalone_edge.py --image_dir imgs/ --checkpoint best_model.pt --output_dir results/ --threshold 0.5
"""

import argparse
import math
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
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision import transforms


# ============================================================
# Constants
# ============================================================

IMAGE_SIZE = 518
PRETRAINED_MODEL_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
}


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
# DinoVisionTransformer (used as patch_embed in Aggregator)
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
# Feature Extractor (projections from encoder to decoder)
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
# Model Loading
# ============================================================

def _build_model(device):
    aggregator = Aggregator(
        img_size=518, patch_size=14, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
        block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True,
        patch_embed="dinov2_vitl14_reg", aa_order=["frame", "global"],
        aa_block_size=1, qk_norm=True, rope_freq=100, init_values=0.01,
        cached_layer_indices=(4, 11, 17, 23),
    )
    model = VGGTEdgeMask(aggregator)
    return model


def load_model(checkpoint_path, device):
    print("Building model architecture...")
    model = _build_model(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    loss = checkpoint.get("loss", "?")
    print(f"Checkpoint epoch: {epoch}, loss: {loss}")
    return model


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tensor = transform(img)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def predict(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device, non_blocking=True)
    with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        prob_map = model(image_tensor)
    prob_map = F.interpolate(
        prob_map.view(1, 1, 518, 518),
        size=(720, 1280),
        mode="bilinear",
        align_corners=False,
    )
    prob_map = prob_map.squeeze().cpu()
    binary_mask = (prob_map >= threshold).float()
    return binary_mask.numpy()


# ============================================================
# Save Output
# ============================================================

def save_output(binary_mask, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    mask_img.save(output_path)
    print(f"  Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VGGT Edge Mask — Standalone Inference")
    parser.add_argument("--image", type=str, nargs="+", default=None,
                        help="Path(s) to input image(s)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of input images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")

    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide --image or --image_dir")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    if args.image:
        image_paths = [Path(p) for p in args.image]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

    print(f"\nRunning inference on {len(image_paths)} image(s)...\n")

    output_dir = Path(args.output_dir)
    latencies = []

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        image_tensor = preprocess_image(img_path)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        binary_mask = predict(model, image_tensor, device, args.threshold)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        print(f"  Latency: {latency_ms:.1f} ms")

        output_path = output_dir / f"{img_path.stem}_edge_mask.png"
        save_output(binary_mask, output_path)

    print(f"\nDone. {len(image_paths)} image(s) processed.")
    if latencies:
        print(f"Latency — avg: {sum(latencies)/len(latencies):.1f} ms | "
              f"min: {min(latencies):.1f} ms | max: {max(latencies):.1f} ms")


if __name__ == "__main__":
    main()
