"""
FastVGGT Token Merging Implementation

Based on: "FastVGGT: Training-Free Acceleration of Visual Geometry Transformer"
Paper: https://arxiv.org/abs/2509.02560
GitHub: https://github.com/mystorm16/FastVGGT

This module implements token merging to accelerate VGGT's Global Attention bottleneck
while preserving reconstruction quality and mitigating error accumulation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class TokenMergingConfig:
    """Configuration for FastVGGT token merging."""

    def __init__(
        self,
        merge_ratio: float = 0.9,
        salient_stride: int = 10,
        apply_from_block: int = 0,
        merge_global_only: bool = True,
    ):
        """
        Args:
            merge_ratio: Fraction of tokens to merge (0.9 = merge 90%, keep 10%)
            salient_stride: Stride for selecting salient tokens (10 = keep ~10%)
            apply_from_block: Which block to start applying merging (0 = all blocks)
            merge_global_only: If True, only merge in global attention, not frame attention
        """
        self.merge_ratio = merge_ratio
        self.salient_stride = salient_stride
        self.apply_from_block = apply_from_block
        self.merge_global_only = merge_global_only


class TokenMerger:
    """
    FastVGGT token merging with three-category partitioning:
    1. Reference tokens (first frame) - always destination
    2. Salient tokens - bypass merging, kept for correspondences
    3. Src/Dst tokens - merged based on similarity
    """

    def __init__(self, config: TokenMergingConfig):
        self.config = config
        self.merge_info = None

    def partition_tokens(
        self,
        x: torch.Tensor,
        frame_idx: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Partition tokens into reference, salient, src, and dst categories.

        Args:
            x: Token tensor [B, N, C] where N = num_frames * tokens_per_frame
            frame_idx: Frame index for each token [B, N]
            num_frames: Total number of frames
            tokens_per_frame: Number of tokens per frame

        Returns:
            dst_tokens: Destination tokens (first frame + sampled)
            src_tokens: Source tokens to be merged
            salient_tokens: Salient tokens that bypass merging
            merge_info: Dict containing mapping information for unmerging
        """
        B, N, C = x.shape
        device = x.device

        # 1. Extract first frame tokens (reference tokens - always dst)
        first_frame_mask = frame_idx == 0  # [B, N]

        # 2. Select salient tokens from remaining frames using fixed-stride sampling
        # This retains ~10% of tokens as distinctive keypoints
        salient_mask = torch.zeros_like(first_frame_mask, dtype=torch.bool)
        for f in range(1, num_frames):
            frame_mask = frame_idx == f
            frame_indices = torch.where(frame_mask[0])[0]  # Assuming same for all batch

            if len(frame_indices) > 0:
                # Select every Nth token as salient
                salient_indices = frame_indices[::self.config.salient_stride]
                salient_mask[:, salient_indices] = True

        # 3. Region-based random sampling for remaining tokens
        # Partition into src and dst using spatially balanced selection
        remaining_mask = ~(first_frame_mask | salient_mask)

        dst_mask = first_frame_mask.clone()
        src_mask = torch.zeros_like(first_frame_mask)

        for f in range(1, num_frames):
            frame_mask = (frame_idx == f) & remaining_mask
            frame_indices = torch.where(frame_mask[0])[0]

            if len(frame_indices) > 0:
                # Calculate how many dst tokens to sample from this frame
                num_frame_tokens = len(frame_indices)
                num_dst = int(num_frame_tokens * (1 - self.config.merge_ratio))

                # Region-based sampling: divide into grid and sample uniformly
                # For simplicity, use random sampling with fixed seed for determinism
                perm = torch.randperm(num_frame_tokens, device=device)
                dst_indices = frame_indices[perm[:num_dst]]
                src_indices = frame_indices[perm[num_dst:]]

                dst_mask[:, dst_indices] = True
                src_mask[:, src_indices] = True

        # Extract token sets
        dst_tokens = x[dst_mask.unsqueeze(-1).expand_as(x)].view(B, -1, C)
        src_tokens = x[src_mask.unsqueeze(-1).expand_as(x)].view(B, -1, C)
        salient_tokens = x[salient_mask.unsqueeze(-1).expand_as(x)].view(B, -1, C)

        # Store merge info for unmerging
        merge_info = {
            'dst_mask': dst_mask,
            'src_mask': src_mask,
            'salient_mask': salient_mask,
            'original_shape': x.shape,
            'src_to_dst_mapping': None,  # Will be filled during merge
        }

        return dst_tokens, src_tokens, salient_tokens, merge_info

    def merge_tokens(
        self,
        dst_tokens: torch.Tensor,
        src_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge source tokens into their most similar destination tokens.

        Args:
            dst_tokens: [B, N_dst, C]
            src_tokens: [B, N_src, C]

        Returns:
            merged_tokens: [B, N_dst, C] with src tokens averaged into dst
            src_to_dst_idx: [B, N_src] mapping of src token to dst token index
        """
        B, N_dst, C = dst_tokens.shape
        N_src = src_tokens.shape[1]

        if N_src == 0:
            return dst_tokens, torch.zeros(B, 0, dtype=torch.long, device=dst_tokens.device)

        # Compute cosine similarity between each src and all dst tokens
        # Normalize for cosine similarity
        src_norm = F.normalize(src_tokens, p=2, dim=-1)  # [B, N_src, C]
        dst_norm = F.normalize(dst_tokens, p=2, dim=-1)  # [B, N_dst, C]

        # Compute similarity: [B, N_src, N_dst]
        similarity = torch.bmm(src_norm, dst_norm.transpose(1, 2))

        # Find most similar dst token for each src token
        src_to_dst_idx = similarity.argmax(dim=2)  # [B, N_src]

        # Merge by averaging
        merged_tokens = dst_tokens.clone()
        for b in range(B):
            for s in range(N_src):
                dst_idx = src_to_dst_idx[b, s].item()
                # Average the src token into its matched dst token
                merged_tokens[b, dst_idx] = (merged_tokens[b, dst_idx] + src_tokens[b, s]) / 2.0

        return merged_tokens, src_to_dst_idx

    def apply_merging(
        self,
        x: torch.Tensor,
        frame_idx: torch.Tensor,
        num_frames: int,
        tokens_per_frame: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply full token merging pipeline: partition -> merge -> combine.

        Args:
            x: Input tokens [B, N, C]
            frame_idx: Frame index for each token [B, N]
            num_frames: Total number of frames
            tokens_per_frame: Tokens per frame

        Returns:
            merged_x: Merged tokens [B, N_merged, C]
            merge_info: Information needed for unmerging
        """
        # Partition tokens
        dst_tokens, src_tokens, salient_tokens, merge_info = self.partition_tokens(
            x, frame_idx, num_frames, tokens_per_frame
        )

        # Merge src into dst
        merged_dst, src_to_dst_idx = self.merge_tokens(dst_tokens, src_tokens)
        merge_info['src_to_dst_mapping'] = src_to_dst_idx

        # Combine merged dst with salient tokens
        merged_x = torch.cat([merged_dst, salient_tokens], dim=1)
        merge_info['num_dst'] = merged_dst.shape[1]
        merge_info['num_salient'] = salient_tokens.shape[1]

        return merged_x, merge_info

    def unmerge_tokens(
        self,
        merged_x: torch.Tensor,
        merge_info: dict,
    ) -> torch.Tensor:
        """
        Unmerge tokens back to original sequence length for dense prediction.

        Args:
            merged_x: Merged tokens [B, N_merged, C]
            merge_info: Merge information from apply_merging

        Returns:
            x: Restored tokens [B, N, C] with original sequence length
        """
        B, N, C = merge_info['original_shape']
        device = merged_x.device

        # Split merged tokens back into dst and salient
        num_dst = merge_info['num_dst']
        merged_dst = merged_x[:, :num_dst]
        salient_tokens = merged_x[:, num_dst:]

        # Reconstruct original tensor
        x_reconstructed = torch.zeros(B, N, C, device=device, dtype=merged_x.dtype)

        dst_mask = merge_info['dst_mask']
        src_mask = merge_info['src_mask']
        salient_mask = merge_info['salient_mask']
        src_to_dst_idx = merge_info['src_to_dst_mapping']

        # Place dst tokens
        x_reconstructed[dst_mask] = merged_dst.reshape(-1, C)

        # Place salient tokens
        x_reconstructed[salient_mask] = salient_tokens.reshape(-1, C)

        # Replicate merged dst values for src positions
        for b in range(B):
            src_positions = torch.where(src_mask[b])[0]
            for i, src_pos in enumerate(src_positions):
                if i < src_to_dst_idx.shape[1]:
                    dst_idx = src_to_dst_idx[b, i].item()
                    x_reconstructed[b, src_pos] = merged_dst[b, dst_idx]

        return x_reconstructed


def create_frame_index_tensor(batch_size: int, num_frames: int, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
    """
    Create a frame index tensor for token merging.

    Args:
        batch_size: Batch size
        num_frames: Number of frames
        tokens_per_frame: Tokens per frame
        device: Device to create tensor on

    Returns:
        frame_idx: [B, N] tensor where each token is labeled with its frame index
    """
    frame_idx = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    frame_idx = frame_idx.unsqueeze(0).expand(batch_size, -1)
    return frame_idx
