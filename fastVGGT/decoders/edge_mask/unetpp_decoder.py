import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_ch=64, output_size=518):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.head(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)
        return x


class UNetPPDecoder(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3 = channels

        # Upsample blocks (lower level → upper level)
        self.up_3_0 = Upsample(c3, c2)
        self.up_2_0 = Upsample(c2, c1)
        self.up_2_1 = Upsample(c2, c1)
        self.up_1_0 = Upsample(c1, c0)
        self.up_1_1 = Upsample(c1, c0)
        self.up_1_2 = Upsample(c1, c0)

        # ConvBlocks for intermediate nodes
        self.conv_2_1 = ConvBlock(c2 + c2, c2)        # 512 → 256
        self.conv_1_1 = ConvBlock(c1 + c1, c1)        # 256 → 128
        self.conv_1_2 = ConvBlock(c1 * 3, c1)         # 384 → 128
        self.conv_0_1 = ConvBlock(c0 + c0, c0)        # 128 → 64
        self.conv_0_2 = ConvBlock(c0 * 3, c0)         # 192 → 64
        self.conv_0_3 = ConvBlock(c0 * 4, c0)         # 256 → 64

        # Deep supervision heads
        self.ds1 = DeepSupervisionHead(c0)
        self.ds2 = DeepSupervisionHead(c0)

    def forward(self, features):
        x_0_0, x_1_0, x_2_0, x_3_0 = features

        size_0 = x_0_0.shape[2:]  # (148, 148)
        size_1 = x_1_0.shape[2:]  # (74, 74)
        size_2 = x_2_0.shape[2:]  # (37, 37)

        # Level 2, column 1
        x_2_1 = self.conv_2_1(torch.cat([x_2_0, self.up_3_0(x_3_0, size_2)], dim=1))

        # Level 1, column 1
        x_1_1 = self.conv_1_1(torch.cat([x_1_0, self.up_2_0(x_2_0, size_1)], dim=1))

        # Level 1, column 2
        x_1_2 = self.conv_1_2(torch.cat([x_1_0, x_1_1, self.up_2_1(x_2_1, size_1)], dim=1))

        # Level 0, column 1
        x_0_1 = self.conv_0_1(torch.cat([x_0_0, self.up_1_0(x_1_0, size_0)], dim=1))

        # Level 0, column 2
        x_0_2 = self.conv_0_2(torch.cat([x_0_0, x_0_1, self.up_1_1(x_1_1, size_0)], dim=1))

        # Level 0, column 3
        x_0_3 = self.conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up_1_2(x_1_2, size_0)], dim=1))

        # Deep supervision
        ds1_out = self.ds1(x_0_1)
        ds2_out = self.ds2(x_0_2)

        return x_0_3, ds1_out, ds2_out
