import torch.nn as nn


class EdgeRefinement(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.refine(x)
