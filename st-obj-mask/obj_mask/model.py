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
            images: [B, S, 3, 518, 518] in range [0, 1]

        Returns:
            mask_logits: [B, S, 2, 518, 518]
        """
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        mask_logits = self.decoder(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )

        return mask_logits
