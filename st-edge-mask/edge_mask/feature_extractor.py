import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    def __init__(self, in_ch=1536, out_ch=64, target_size=None, downsample=False):
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


class StudentFeatureExtractor(nn.Module):
    """
    Feature extractor for Student encoder (1536 dim output).
    """
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        self.layer_indices = [3, 8, 13, 17]
        self.patch_start_idx = 5

        # CRITICAL FIX: Normalize student features to match teacher scale
        # Student outputs unnormalized features (mean=-1076, std=27k)
        # Teacher outputs normalized features (mean~0, std~2)
        # LayerNorm normalizes immediately + learns optimal scale during training
        self.input_norms = nn.ModuleList([
            nn.LayerNorm(1536) for _ in range(4)
        ])

        self.projections = nn.ModuleList([
            FeatureProjection(1536, 64, target_size=(148, 148)),
            FeatureProjection(1536, 128, target_size=(74, 74)),
            FeatureProjection(1536, 256),
            FeatureProjection(1536, 512, downsample=True),
        ])

    def forward(self, images):
        B, S = images.shape[:2]

        with torch.no_grad():
            aggregated_tokens_list, _ = self.aggregator(images)

        features = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Layer {layer_idx} returned None - check student caching")
            x = x[:, :, self.patch_start_idx:]
            x = x.reshape(B * S, x.shape[2], x.shape[3])

            # CRITICAL FIX: Normalize BEFORE projection
            # Detach BEFORE normalization: freeze encoder, but train LayerNorm
            # x shape: [B*S, Patches, 1536]
            x = x.detach()  # Stop gradients to encoder
            x_norm = self.input_norms[i](x)  # Normalize (gradients flow to LayerNorm)

            x_norm = x_norm.permute(0, 2, 1)  # [B*S, 1536, Patches]
            x_norm = x_norm.reshape(B * S, 1536, 37, 37)
            x_proj = self.projections[i](x_norm)
            features.append(x_proj)

        return features
