import time

import torch
import torch.nn as nn
from typing import List, Optional

from huggingface_hub import hf_hub_download

from encoder import Aggregator
from decoders.obj_mask import ObjMaskDecoder
from decoders.edge_mask import EdgeMaskDecoder

HF_MODEL_ID = "facebook/VGGT-1B"


class VGGTUnified(nn.Module):
    """
    Shared VGGT Aggregator encoder with two task-specific decoders.

    Encoder weights are loaded from HuggingFace (facebook/VGGT-1B).
    Forward pass is selectable via task='both'|'obj'|'edge'.
    """

    def __init__(self, load_encoder: bool = True):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
        )

        self.obj_decoder = ObjMaskDecoder(
            dim_in=2048,
            output_dim=2,
        )

        self.edge_decoder = EdgeMaskDecoder()

        if load_encoder:
            self._load_encoder_from_hf()

    def _load_encoder_from_hf(self):
        """Download VGGT-1B from HuggingFace and load only the aggregator weights."""
        print(f"Loading encoder from HuggingFace: {HF_MODEL_ID}")
        ckpt_path = hf_hub_download(repo_id=HF_MODEL_ID, filename="model.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        agg_state = {}
        for k, v in state_dict.items():
            if k.startswith("aggregator."):
                agg_state[k[len("aggregator."):]] = v

        self.aggregator.load_state_dict(agg_state)
        self.aggregator.eval()
        self.aggregator.requires_grad_(False)
        print(f"  Encoder loaded and frozen ({sum(p.numel() for p in self.aggregator.parameters())/1e6:.0f}M params)")

    def forward(self, images: torch.Tensor, task: str = "both"):
        """
        Args:
            images: [B, S, 3, 518, 518] in range [0, 1]
            task: 'both' | 'obj' | 'edge' | 'cascade'

        Returns:
            dict with keys depending on task:
                'obj_mask': [B, S, 2, 518, 518] logits (when task='obj' or 'both')
                'edge_mask': [B, S, 1, 518, 518] sigmoid probabilities (when task='edge' or 'both')

            For task='cascade':
                'obj_mask': [B, S, 2, 518, 518] logits
                'edge_mask': [B, S, 1, 518, 518] edge probs masked to ROI only (background=0)
                'roi_bbox': list of (x_min, y_min, x_max, y_max) per frame
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, S = images.shape[:2]
        latency = {}

        if images.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        if images.is_cuda:
            torch.cuda.synchronize()
        latency["encoder"] = time.perf_counter() - t0

        results = {}

        if task in ("obj", "both"):
            t1 = time.perf_counter()
            obj_logits = self.obj_decoder(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            if obj_logits.dim() == 4:
                C, H, W = obj_logits.shape[1:]
                obj_logits = obj_logits.view(B, S, C, H, W)
            if images.is_cuda:
                torch.cuda.synchronize()
            latency["obj_decoder"] = time.perf_counter() - t1
            results["obj_mask"] = obj_logits

        if task in ("edge", "both"):
            t1 = time.perf_counter()
            edge_probs = self.edge_decoder(aggregated_tokens_list, B, S)
            if images.is_cuda:
                torch.cuda.synchronize()
            latency["edge_decoder"] = time.perf_counter() - t1
            results["edge_mask"] = edge_probs

        if task == "cascade":
            t1 = time.perf_counter()
            obj_logits = self.obj_decoder(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            if obj_logits.dim() == 4:
                C, H, W = obj_logits.shape[1:]
                obj_logits = obj_logits.view(B, S, C, H, W)
            if images.is_cuda:
                torch.cuda.synchronize()
            latency["obj_decoder"] = time.perf_counter() - t1
            results["obj_mask"] = obj_logits

            t2 = time.perf_counter()
            roi_mask = obj_logits.argmax(dim=2, keepdim=True).float()
            roi_bboxes = self._extract_roi_bboxes(roi_mask)
            latency["roi_extraction"] = time.perf_counter() - t2
            results["roi_bbox"] = roi_bboxes

            t3 = time.perf_counter()
            edge_probs = self.edge_decoder(aggregated_tokens_list, B, S)
            edge_probs = edge_probs * roi_mask
            if images.is_cuda:
                torch.cuda.synchronize()
            latency["edge_decoder"] = time.perf_counter() - t3
            results["edge_mask"] = edge_probs

        latency["total"] = sum(latency.values())
        results["latency"] = latency

        return results

    @staticmethod
    def _extract_roi_bboxes(roi_mask: torch.Tensor):
        """
        Extract bounding boxes from binary ROI mask.

        Args:
            roi_mask: [B, S, 1, H, W] binary mask (1=object, 0=background)

        Returns:
            List of lists of (x_min, y_min, x_max, y_max) per batch per frame.
            Returns None for frames with no object detected.
        """
        B, S, _, H, W = roi_mask.shape
        bboxes = []
        for b in range(B):
            batch_bboxes = []
            for s in range(S):
                mask_2d = roi_mask[b, s, 0]
                ys, xs = torch.where(mask_2d > 0.5)
                if len(ys) == 0:
                    batch_bboxes.append(None)
                else:
                    batch_bboxes.append((
                        xs.min().item(), ys.min().item(),
                        xs.max().item(), ys.max().item()
                    ))
            bboxes.append(batch_bboxes)
        return bboxes

    def load_unified_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """Load the full model (encoder + both decoders) from a single checkpoint."""
        print(f"Loading unified checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"]

        fp16 = ckpt.get("config", {}).get("fp16", False)
        if fp16:
            state_dict = {k: v.float() for k, v in state_dict.items()}

        # Load with strict=False to allow truncated models (ignoring extra blocks)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        if unexpected_keys:
            # Count how many blocks are being ignored
            ignored_frame_blocks = set()
            ignored_global_blocks = set()
            for key in unexpected_keys:
                if 'frame_blocks.' in key:
                    block_num = int(key.split('frame_blocks.')[1].split('.')[0])
                    ignored_frame_blocks.add(block_num)
                elif 'global_blocks.' in key:
                    block_num = int(key.split('global_blocks.')[1].split('.')[0])
                    ignored_global_blocks.add(block_num)

            if ignored_frame_blocks or ignored_global_blocks:
                print(f"  Ignored {len(ignored_frame_blocks)} frame blocks and {len(ignored_global_blocks)} global blocks from checkpoint")

        self.aggregator.eval()
        self.aggregator.requires_grad_(False)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Unified model loaded ({total_params / 1e6:.0f}M params, fp16_stored={fp16})")

    def load_decoder_checkpoint(self, task: str, checkpoint_path: str, device: str = "cpu"):
        """Load a pretrained decoder checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        if task == "obj":
            decoder_prefix = "depth_head."
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith(decoder_prefix):
                    new_state[k[len(decoder_prefix):]] = v
            self.obj_decoder.load_state_dict(new_state)
        elif task == "edge":
            edge_prefixes = ("feature_extractor.", "decoder.", "refinement.", "final_conv.")
            new_state = {}
            for k, v in state_dict.items():
                for prefix in edge_prefixes:
                    if k.startswith(prefix):
                        mapped_key = k
                        if k.startswith("feature_extractor.projections."):
                            mapped_key = k.replace("feature_extractor.projections.",
                                                   "feature_projections.projections.")
                        elif k.startswith("feature_extractor."):
                            continue
                        new_state[mapped_key] = v
                        break
            self.edge_decoder.load_state_dict(new_state, strict=False)
