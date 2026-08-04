import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear embedding used in the SegFormer decoder.

    Input:
        [B, C, H, W]

    Output:
        [B, H*W, embed_dim]
    """

    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):

        B, C, H, W = x.shape

        # [B,C,H,W]
        # ->
        # [B,HW,C]
        x = x.flatten(2).transpose(1, 2)

        # Linear projection
        x = self.proj(x)

        return x
    
class SegFormerDecoder(nn.Module):
    """
    Pure PyTorch implementation of the SegFormer decoder.

    Input:
        c1 : [B,256,H/4,W/4]
        c2 : [B,512,H/8,W/8]
        c3 : [B,1024,H/16,W/16]
        c4 : [B,1024,H/32,W/32]

    Output:
        mask logits

        [B,num_classes,H/4,W/4]
    """

    def __init__(
        self,
        in_channels=[256, 512, 1024, 1024],
        embedding_dim=256,
        num_classes=1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        ####################################################
        # Linear embedding for every feature level
        ####################################################

        self.linear_c1 = MLP(in_channels[0], embedding_dim)
        self.linear_c2 = MLP(in_channels[1], embedding_dim)
        self.linear_c3 = MLP(in_channels[2], embedding_dim)
        self.linear_c4 = MLP(in_channels[3], embedding_dim)

        ####################################################
        # Fuse all four embedded feature maps
        ####################################################

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                embedding_dim * 4,
                embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        ####################################################
        # Final segmentation layer
        ####################################################

        self.dropout = nn.Dropout2d(0.1)

        self.linear_pred = nn.Conv2d(
            embedding_dim,
            num_classes,
            kernel_size=1,
        )
        
    def forward(self, features):

        c1, c2, c3, c4 = features

        B = c1.shape[0]

        ####################################################
        # C4
        ####################################################

        _c4 = self.linear_c4(c4)
        _c4 = _c4.permute(0, 2, 1).reshape(
            B,
            self.embedding_dim,
            c4.shape[2],
            c4.shape[3],
        )

        _c4 = F.interpolate(
            _c4,
            size=c1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        ####################################################
        # C3
        ####################################################

        _c3 = self.linear_c3(c3)
        _c3 = _c3.permute(0, 2, 1).reshape(
            B,
            self.embedding_dim,
            c3.shape[2],
            c3.shape[3],
        )

        _c3 = F.interpolate(
            _c3,
            size=c1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        ####################################################
        # C2
        ####################################################

        _c2 = self.linear_c2(c2)
        _c2 = _c2.permute(0, 2, 1).reshape(
            B,
            self.embedding_dim,
            c2.shape[2],
            c2.shape[3],
        )

        _c2 = F.interpolate(
            _c2,
            size=c1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        ####################################################
        # C1
        ####################################################

        _c1 = self.linear_c1(c1)
        _c1 = _c1.permute(0, 2, 1).reshape(
            B,
            self.embedding_dim,
            c1.shape[2],
            c1.shape[3],
        )

        ####################################################
        # Fuse
        ####################################################

        x = torch.cat(
            [_c4, _c3, _c2, _c1],
            dim=1,
        )

        x = self.linear_fuse(x)

        x = self.dropout(x)

        x = self.linear_pred(x)

        return x