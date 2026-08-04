# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.segformer_head_for_dpt import DPTHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        ########################################################
        # Heads
        ########################################################

        self.camera_head = (
            CameraHead(dim_in=2 * embed_dim)
            if enable_camera
            else None
        )

        # Disabled for now
        self.point_head = None

        # SegFormer-based segmentation head
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
            )
            if enable_depth
            else None
        )

        self.track_head = None

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
    ):

        ########################################################
        # Add batch dimension if required
        ########################################################

        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        ########################################################
        # VGGT Encoder
        ########################################################

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        ########################################################
        # Heads
        ########################################################

        with torch.cuda.amp.autocast(enabled=False):

            ########################
            # Camera
            ########################

            if self.camera_head is not None:

                pose_enc_list = self.camera_head(
                    aggregated_tokens_list
                )

                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            ########################
            # Segmentation
            ########################

            if self.depth_head is not None:

                mask_logits = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )

                predictions["mask_logits"] = mask_logits
                
        ########################################################
        # Save input image during inference
        ########################################################

        if not self.training:
            predictions["images"] = images

        return predictions