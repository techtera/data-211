"""Student encoder + Object mask decoder wrapper."""

import torch
import torch.nn as nn

from obj_mask import ObjMaskDecoder


class StudentObjMask(nn.Module):
    """Object segmentation using Student encoder (255M params)."""

    def __init__(self, student_aggregator):
        super().__init__()
        self.aggregator = student_aggregator
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        self.decoder = ObjMaskDecoder(
            dim_in=1536,  # Student output dim
            output_dim=2,  # Background + object
            patch_size=14,
            intermediate_layer_idx=[3, 8, 13, 17],  # Student cached layers
        )

    def forward(self, images):
        """
        Args:
            images: [B, 3, 518, 518] or [B, S, 3, 518, 518] in range [0, 1]

        Returns:
            mask_logits: [B, 2, 518, 518] or [B, S, 2, 518, 518]
        """
        # Handle both 4D [B, C, H, W] and 5D [B, S, C, H, W] inputs
        if images.ndim == 4:
            images = images.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        mask_logits = self.decoder(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )

        # Squeeze back to [B, 2, H, W] if input was 4D
        if squeeze_output:
            mask_logits = mask_logits.squeeze(1)

        return mask_logits
