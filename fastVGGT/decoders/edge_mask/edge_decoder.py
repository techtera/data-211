import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_projections import FeatureProjections
from .unetpp_decoder import UNetPPDecoder
from .refinement import EdgeRefinement


class EdgeMaskDecoder(nn.Module):
    """Edge mask decoder: FeatureProjections → UNet++ → EdgeRefinement → 1-ch logits."""

    def __init__(self):
        super().__init__()
        self.feature_projections = FeatureProjections()
        self.decoder = UNetPPDecoder(channels=(64, 128, 256, 512))
        self.refinement = EdgeRefinement(ch=64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, aggregated_tokens_list, B, S):
        features = self.feature_projections(aggregated_tokens_list, B, S)

        x_0_3, ds1_logits, ds2_logits = self.decoder(features)

        x = self.refinement(x_0_3)

        logits = self.final_conv(x)
        logits = F.interpolate(logits, size=(518, 518),
                               mode="bilinear", align_corners=False)

        logits = logits.view(B, S, 1, 518, 518)

        return torch.sigmoid(logits)
