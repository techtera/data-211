import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor import StudentFeatureExtractor
from .decoder import UNetPPDecoder
from .refinement import EdgeRefinement


class StudentEdgeMask(nn.Module):
    """Edge mask prediction using Student encoder (255M params)."""

    def __init__(self, student_aggregator):
        super().__init__()
        self.feature_extractor = StudentFeatureExtractor(student_aggregator)
        self.decoder = UNetPPDecoder(channels=(64, 128, 256, 512))
        self.refinement = EdgeRefinement(ch=64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, images):
        """
        Args:
            images: [B, 3, 518, 518] or [B, S, 3, 518, 518] in range [0, 1]

        Returns:
            training mode: logits, ds1_logits, ds2_logits (each [B, 1, 518, 518] or [B, S, 1, 518, 518])
            eval mode: sigmoid(logits)
        """
        # Handle both 4D [B, C, H, W] and 5D [B, S, C, H, W] inputs
        if images.ndim == 4:
            images = images.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
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

        # Squeeze back to [B, 1, H, W] if input was 4D
        if squeeze_output:
            logits = logits.squeeze(1)
            ds1_logits = ds1_logits.squeeze(1)
            ds2_logits = ds2_logits.squeeze(1)

        if self.training:
            return logits, ds1_logits, ds2_logits
        else:
            return torch.sigmoid(logits)
