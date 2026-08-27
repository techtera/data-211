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
