import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeatureProjections(nn.Module):
    """Projects pre-computed aggregator tokens into multi-scale feature maps."""

    def __init__(self):
        super().__init__()
        self.layer_indices = [4, 11, 17, 23]
        self.patch_start_idx = 5

        self.projections = nn.ModuleList([
            FeatureProjection(2048, 64, target_size=(148, 148)),
            FeatureProjection(2048, 128, target_size=(74, 74)),
            FeatureProjection(2048, 256),
            FeatureProjection(2048, 512, downsample=True),
        ])

    def forward(self, aggregated_tokens_list, B, S):
        features = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx]
            x = x[:, :, self.patch_start_idx:]
            x = x.reshape(B * S, x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1)
            x = x.reshape(B * S, 2048, 37, 37)
            x = self.projections[i](x)
            features.append(x)

        return features
