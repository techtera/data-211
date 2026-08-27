import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .head_utils import create_uv_grid, position_grid_to_embed
from .segformer_decoder import SegFormerDecoder

class DPTHead(nn.Module):
    """
    VGGT Segmentation Head using a SegFormer Decoder.

    This module keeps the feature extraction pipeline from the original
    DPT head and replaces only the DPT refinement network with a
    SegFormer decoder.

    Overall Pipeline
    ----------------
    Input Images
            │
            ▼
    VGGT Aggregator
            │
            ▼
    Transformer Tokens
            │
            ▼
    LayerNorm
            │
            ▼
    1×1 Projection Layers
            │
            ▼
    Multi-scale Feature Pyramid
            │
            ▼
    SegFormer Decoder
            │
            ▼
    Segmentation Mask Logits

    Args:
        dim_in:
            Channel dimension of the transformer tokens.

        patch_size:
            Patch size used by the VGGT backbone.

        output_dim:
            Number of segmentation classes.

        out_channels:
            Number of output channels for each pyramid level.

        intermediate_layer_idx:
            Transformer layers used to construct the feature pyramid.

        pos_embed:
            Whether positional embeddings should be added to the
            generated feature maps.
    """

    def __init__(
        self,
        dim_in: int = 1536,  # Student encoder output dim (768 frame + 768 global)
        patch_size: int = 14,
        output_dim: int = 2,  # Background + object
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [3, 8, 13, 17],  # Student cached layers
        pos_embed: bool = True,
    ) -> None:

        super(DPTHead, self).__init__()

        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx

        ####################################################################
        # Layer Normalization
        #
        # Transformer token embeddings are normalized before being converted
        # into spatial feature maps.
        ####################################################################

        self.norm = nn.LayerNorm(dim_in)

        ####################################################################
        # Projection Layers
        #
        # Each selected transformer layer is projected from the transformer
        # embedding dimension (dim_in) into the channel dimensions expected
        # by the SegFormer decoder.
        #
        # Output channels:
        #
        #   Level 1 → 256
        #   Level 2 → 512
        #   Level 3 → 1024
        #   Level 4 → 1024
        ####################################################################

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

        ####################################################################
        # Resize Layers
        #
        # The projected features are resized to form a four-level feature
        # pyramid compatible with the SegFormer decoder.
        #
        # Pyramid Levels
        #
        #   c1 : 84 × 148   (Stride 4)
        #   c2 : 42 × 74    (Stride 8)
        #   c3 : 21 × 37    (Stride 16)
        #   c4 : 11 × 19    (Stride 32)
        ####################################################################

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

        ####################################################################
        # SegFormer Decoder
        #
        # Receives the four-level feature pyramid produced above.
        #
        # Input Features
        #
        #   c1 : [B*S, 256, 84,148]
        #   c2 : [B*S, 512, 42,74]
        #   c3 : [B*S,1024,21,37]
        #   c4 : [B*S,1024,11,19]
        #
        # Output
        #
        #   mask_logits :
        #
        #   [B*S, num_classes, 84,148]
        #
        # These logits are later upsampled to the original image resolution.
        ####################################################################
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
        """
        Forward pass of the VGGT Segmentation Head.

        This function supports processing long image sequences by optionally
        splitting them into smaller chunks. Chunked inference reduces GPU
        memory usage while producing the same final output.

        Inputs
        ------
        aggregated_tokens_list:
            List of intermediate transformer outputs from the VGGT
            Aggregator.

        images:
            Input image sequence of shape

                [B, S, 3, H, W]

            where

                B : Batch size
                S : Number of frames

        patch_start_idx:
            Index indicating where patch tokens begin inside each
            transformer token sequence.

        frames_chunk_size:
            Number of frames processed simultaneously.
            If None or greater than the sequence length, all frames
            are processed together.

        Returns
        -------
        mask_logits:

            Tensor of shape

                [B, S, num_classes, H, W]

            containing segmentation logits for every frame.
        """

        B, S, _, H, W = images.shape

        ####################################################################
        # Fast Path
        #
        # If chunking is disabled (or unnecessary), process the entire
        # sequence in a single forward pass.
        ####################################################################

        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
            )

        ####################################################################
        # Chunked Inference
        #
        # Long sequences can consume a large amount of GPU memory.
        #
        # Instead of processing every frame together, split the sequence
        # into smaller chunks and concatenate the predictions afterwards.
        ####################################################################

        assert frames_chunk_size > 0

        outputs = []

        ####################################################################
        # Process one chunk of frames at a time.
        ####################################################################

        for frames_start_idx in range(0, S, frames_chunk_size):

            frames_end_idx = min(
                frames_start_idx + frames_chunk_size,
                S,
            )

            chunk_output = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_start_idx,
                frames_start_idx,
                frames_end_idx,
            )

            outputs.append(chunk_output)

        ####################################################################
        # Reconstruct the complete sequence by concatenating all chunk
        # predictions along the temporal (sequence) dimension.
        #
        # Shape:
        #
        # Before:
        #     List of [B, chunk_size, C, H, W]
        #
        # After:
        #     [B, S, C, H, W]
        ####################################################################

        return torch.cat(outputs, dim=1)
    
    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> torch.Tensor:
        """
        Internal implementation of the segmentation forward pass.

        This function converts intermediate VGGT transformer features into a
        four-level feature pyramid before passing them through the SegFormer
        decoder.

        Pipeline
        --------

        Transformer Tokens
                │
                ▼
        Remove Special Tokens
                │
                ▼
        Layer Normalization
                │
                ▼
        Token → Feature Map Projection
                │
                ▼
        Multi-scale Feature Pyramid
                │
                ▼
        SegFormer Decoder
                │
                ▼
        Mask Logits
                │
                ▼
        Upsample to Original Resolution

        Args
        ----
        aggregated_tokens_list:
            Intermediate transformer outputs from selected VGGT layers.

        images:
            Input images of shape

                [B, S, 3, H, W]

        patch_start_idx:
            Index where image patch tokens begin.

        frames_start_idx:
            First frame index when chunked inference is enabled.

        frames_end_idx:
            Last frame index (exclusive) when chunked inference is enabled.

        Returns
        -------
        mask_logits:

            Tensor of shape

                [B, S, num_classes, H, W]
        """

        ####################################################################
        # Select only the current chunk of frames (if chunking is enabled).
        ####################################################################

        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        ####################################################################
        # Number of image patches along each spatial dimension.
        ####################################################################

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        ####################################################################
        # Multi-scale feature pyramid that will be passed to SegFormer.
        ####################################################################

        out = []
        dpt_idx = 0

        ####################################################################
        # Build the feature pyramid using transformer features from the
        # selected intermediate layers.
        ####################################################################

        for layer_idx in self.intermediate_layer_idx:

            ################################################################
            # Extract only the image patch tokens.
            #
            # Original:
            #
            #   [B, S, TotalTokens, C]
            #
            # After removing special tokens:
            #
            #   [B, S, PatchTokens, C]
            ################################################################

            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            ################################################################
            # Keep only the current frame chunk when chunked inference
            # is enabled.
            ################################################################

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            ################################################################
            # Merge batch and temporal dimensions.
            #
            # Before:
            #
            #   [B, S, PatchTokens, C]
            #
            # After:
            #
            #   [B*S, PatchTokens, C]
            ################################################################

            x = x.reshape(B * S, -1, x.shape[-1])

            ################################################################
            # Normalize transformer embeddings.
            ################################################################

            x = self.norm(x)

            ################################################################
            # Convert token sequence back into a spatial feature map.
            #
            # Before:
            #
            #   [B*S, PatchTokens, C]
            #
            # After:
            #
            #   [B*S, C, patch_h, patch_w]
            ################################################################

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w)
            )

            ################################################################
            # Project transformer features to the channel dimensions expected
            # by the SegFormer decoder.
            ################################################################

            x = self.projects[dpt_idx](x)

            ################################################################
            # Inject positional information into the feature map.
            ################################################################

            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            ################################################################
            # Resize the feature map to the appropriate pyramid level.
            ################################################################

            x = self.resize_layers[dpt_idx](x)

            ################################################################
            # Store the feature map.
            #
            # Resulting pyramid:
            #
            # c1 : [B*S,256,84,148]
            # c2 : [B*S,512,42,74]
            # c3 : [B*S,1024,21,37]
            # c4 : [B*S,1024,11,19]
            ################################################################

            out.append(x)
            # print(f"Pyramid Level {dpt_idx}")
            # print(x.shape)

            dpt_idx += 1

        ####################################################################
        # Decode the multi-scale feature pyramid using the SegFormer decoder.
        #
        # Input:
        #
        #   [c1, c2, c3, c4]
        #
        # Output:
        #
        #   [B*S, num_classes, 84,148]
        ####################################################################

        mask_logits = self.segformer_decoder(out)
        # print("\nDecoder Output Shape")
        # print(mask_logits.shape)

        ####################################################################
        # SegFormer predicts masks at the highest pyramid resolution
        # (1/4 of the original image).
        #
        # Upsample them back to the original image resolution.
        ####################################################################

        mask_logits = F.interpolate(
            mask_logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        # print("\nUpsampled Output Shape")
        # print(mask_logits.shape)
        ####################################################################
        #   [B*S, C, H, W] -> [B, S, C, H, W]
        ####################################################################

        # Reshape to restore batch dimension for chunked inference
        mask_logits = mask_logits.view(B, S, -1, H, W)

        return mask_logits
    
    def _apply_pos_embed(
        self,
        x: torch.Tensor,
        W: int,
        H: int,
        ratio: float = 0.1,
    ) -> torch.Tensor:
        """
        Adds a 2D positional embedding to a feature map.

        After transformer tokens are reshaped into a spatial feature map,
        they no longer explicitly encode their pixel locations. This function
        generates a UV-based positional encoding and adds it to the feature map
        so that the decoder can better preserve spatial information.

        Args
        ----
        x:
            Feature map of shape

                [B, C, H', W']

        W:
            Original image width.

        H:
            Original image height.

        ratio:
            Scaling factor controlling the magnitude of the positional
            embedding before it is added to the feature map.

        Returns
        -------
        Tensor

            Feature map with positional embeddings added.

            Shape:

                [B, C, H', W']
        """

        ####################################################################
        # Current spatial resolution of the feature map.
        ####################################################################

        patch_w = x.shape[-1]
        patch_h = x.shape[-2]

        ####################################################################
        # Create a UV coordinate grid corresponding to the current feature
        # map resolution.
        ####################################################################

        pos_embed = create_uv_grid(
            patch_w,
            patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )

        ####################################################################
        # Convert the UV grid into a positional embedding with the same
        # channel dimension as the feature map.
        ####################################################################

        pos_embed = position_grid_to_embed(
            pos_embed,
            x.shape[1],
        )

        ####################################################################
        # Scale the positional embedding before adding it to the features.
        ####################################################################

        pos_embed = pos_embed * ratio

        ####################################################################
        # Rearrange dimensions from
        #
        #     [H, W, C]
        #
        # to
        #
        #     [1, C, H, W]
        #
        # and replicate across the batch dimension.
        ####################################################################

        pos_embed = (
            pos_embed
            .permute(2, 0, 1)
            [None]
            .expand(x.shape[0], -1, -1, -1)
        )

        ####################################################################
        # Inject positional information into the feature map.
        ####################################################################

        return x + pos_embed