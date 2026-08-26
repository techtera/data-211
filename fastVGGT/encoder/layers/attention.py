# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
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
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
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

    def forward(self, x: Tensor, pos=None, merge_info=None) -> Tensor:
        """
        Forward pass with optional token merging.

        Args:
            x: Input tokens [B, N, C]
            pos: Position embeddings [B, N, 2] for RoPE
            merge_info: Dict with token merging information (optional)
                - If provided: Apply RoPE BEFORE merging (RoPE-First)
                - If None: Standard attention
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # RoPE-First: Apply RoPE BEFORE merging to preserve both src and dst position info
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Token merging: merge q, k, v separately after RoPE encoding
        unmerge_func = None
        if merge_info is not None:
            q, k, v, unmerge_func = self._merge_qkv(q, k, v, merge_info)

        # Attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, -1, C)  # -1 handles merged token count
        x = self.proj(x)
        x = self.proj_drop(x)

        # Unmerge: restore original token count
        if unmerge_func is not None:
            x = unmerge_func(x)

        return x

    def _merge_qkv(self, q, k, v, merge_info):
        """
        Merge q, k, v tensors separately after RoPE has been applied.
        This preserves position information from both src and dst tokens.

        Args:
            q: [B, num_heads, N, head_dim] - already has RoPE applied
            k: [B, num_heads, N, head_dim] - already has RoPE applied
            v: [B, num_heads, N, head_dim]
            merge_info: Dict with masks and mapping

        Returns:
            q_merged, k_merged, v_merged, unmerge_func
        """
        B, num_heads, N, head_dim = q.shape

        # Extract masks
        dst_mask = merge_info['dst_mask']  # [B, N]
        src_mask = merge_info['src_mask']  # [B, N]
        salient_mask = merge_info['salient_mask']  # [B, N]
        src_to_dst_idx = merge_info['src_to_dst_mapping']  # [B, N_src]

        # Reshape to [B, N, num_heads * head_dim] for easier indexing
        q_flat = q.permute(0, 2, 1, 3).reshape(B, N, -1)
        k_flat = k.permute(0, 2, 1, 3).reshape(B, N, -1)
        v_flat = v.permute(0, 2, 1, 3).reshape(B, N, -1)

        # Extract dst, src, salient tokens
        q_dst = q_flat[dst_mask].view(B, -1, num_heads * head_dim)
        k_dst = k_flat[dst_mask].view(B, -1, num_heads * head_dim)
        v_dst = v_flat[dst_mask].view(B, -1, num_heads * head_dim)

        q_src = q_flat[src_mask].view(B, -1, num_heads * head_dim)
        k_src = k_flat[src_mask].view(B, -1, num_heads * head_dim)
        v_src = v_flat[src_mask].view(B, -1, num_heads * head_dim)

        q_salient = q_flat[salient_mask].view(B, -1, num_heads * head_dim)
        k_salient = k_flat[salient_mask].view(B, -1, num_heads * head_dim)
        v_salient = v_flat[salient_mask].view(B, -1, num_heads * head_dim)

        # Merge src into dst using scatter_add (average)
        N_dst = q_dst.shape[1]
        N_src = q_src.shape[1]
        C_head = num_heads * head_dim

        # Count how many src tokens merge to each dst
        dst_count = torch.ones(B, N_dst, 1, device=q.device)
        src_count = torch.ones(B, N_src, 1, device=q.device)
        dst_count = dst_count.scatter_add(1, src_to_dst_idx.unsqueeze(-1).expand(-1, -1, 1), src_count)

        # Merge by averaging
        q_dst = q_dst.scatter_add(1, src_to_dst_idx.unsqueeze(-1).expand(-1, -1, C_head), q_src) / dst_count
        k_dst = k_dst.scatter_add(1, src_to_dst_idx.unsqueeze(-1).expand(-1, -1, C_head), k_src) / dst_count
        v_dst = v_dst.scatter_add(1, src_to_dst_idx.unsqueeze(-1).expand(-1, -1, C_head), v_src) / dst_count

        # Concatenate: [dst, salient]
        q_merged = torch.cat([q_dst, q_salient], dim=1)
        k_merged = torch.cat([k_dst, k_salient], dim=1)
        v_merged = torch.cat([v_dst, v_salient], dim=1)

        # Reshape back to [B, num_heads, N_merged, head_dim]
        q_merged = q_merged.view(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        k_merged = k_merged.view(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        v_merged = v_merged.view(B, -1, num_heads, head_dim).permute(0, 2, 1, 3)

        # Create unmerge function
        def unmerge(x_merged):
            """Unmerge attention output back to original token count"""
            B, N_merged, C = x_merged.shape
            N_dst = dst_mask.sum(dim=1)[0].item()
            N_salient = salient_mask.sum(dim=1)[0].item()

            # Split back to dst and salient
            x_dst = x_merged[:, :N_dst, :]
            x_salient = x_merged[:, N_dst:, :]

            # For src tokens, copy from their dst token
            x_src = torch.gather(x_dst, 1, src_to_dst_idx.unsqueeze(-1).expand(-1, -1, C))

            # Reconstruct original order
            x_full = torch.zeros(B, N, C, device=x_merged.device, dtype=x_merged.dtype)
            x_full[dst_mask] = x_dst.flatten(0, 1)
            x_full[src_mask] = x_src.flatten(0, 1)
            x_full[salient_mask] = x_salient.flatten(0, 1)

            return x_full

        return q_merged, k_merged, v_merged, unmerge


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
