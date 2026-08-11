# Implementation Specification

## File Structure

```
edge_mask/
├── __init__.py
├── model.py              # VGGTEdgeMask (full pipeline)
├── feature_extractor.py  # Extract + project VGGT features
├── decoder.py            # UNet++ decoder (ConvBlock, Upsample, grid)
├── refinement.py         # EdgeRefinement block
├── losses.py             # WeightedBCE + Dice composite loss
├── train.py              # Training loop
├── evaluate.py           # BF1, ODS, Dice metrics
└── config.yaml           # All hyperparameters
```

---

## Module 1: feature_extractor.py

### Class: FeatureProjection

One per level. Projects 2048-dim features to target channels and resizes spatially.

```python
class FeatureProjection(nn.Module):
    def __init__(self, in_ch=2048, out_ch, target_size=None, downsample=False):
        # 1x1 channel projection
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        
        # Spatial resize (varies per level)
        if target_size is not None:  # Levels 0, 1: upsample
            self.resize = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            )
            self.target_size = target_size
        elif downsample:  # Level 3: stride-2 conv
            self.resize = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            )
            self.target_size = None
        else:  # Level 2: identity
            self.resize = None
            self.target_size = None

    def forward(self, x):
        x = self.proj(x)
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
            x = self.resize(x)
        elif self.resize is not None:
            x = self.resize(x)
        return x
```

### Class: VGGTFeatureExtractor

Extracts features from frozen VGGT and projects to 4-level pyramid.

```python
class VGGTFeatureExtractor(nn.Module):
    def __init__(self, vggt_model):
        self.aggregator = vggt_model.aggregator
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)
        
        self.layer_indices = [4, 11, 17, 23]
        self.patch_start_idx = 5  # 1 camera + 4 register
        
        self.projections = nn.ModuleList([
            FeatureProjection(2048, 64, target_size=(148, 148)),    # Level 0
            FeatureProjection(2048, 128, target_size=(74, 74)),     # Level 1
            FeatureProjection(2048, 256),                           # Level 2 (identity)
            FeatureProjection(2048, 512, downsample=True),          # Level 3
        ])

    def forward(self, images):
        # images: [B, S, 3, 518, 518]
        B, S = images.shape[:2]
        
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        features = []
        for i, layer_idx in enumerate(self.layer_indices):
            x = aggregated_tokens_list[layer_idx]  # [B, S, 1374, 2048]
            x = x[:, :, self.patch_start_idx:]     # [B, S, 1369, 2048]
            x = x.reshape(B * S, 1369, 2048)       # [B*S, 1369, 2048]
            x = x.permute(0, 2, 1)                 # [B*S, 2048, 1369]
            x = x.reshape(B * S, 2048, 37, 37)     # [B*S, 2048, 37, 37]
            x = x.detach()
            x = self.projections[i](x)
            features.append(x)
        
        return features  # [level0, level1, level2, level3]
```

---

## Module 2: decoder.py

### Class: ConvBlock

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
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
```

### Class: Upsample

```python
class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    
    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.conv(x)
```

### Class: DeepSupervisionHead

```python
class DeepSupervisionHead(nn.Module):
    def __init__(self, in_ch=64, output_size=518):
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
        return x  # logits
```

### Class: UNetPPDecoder

```python
class UNetPPDecoder(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        # channels[0]=64, channels[1]=128, channels[2]=256, channels[3]=512
        
        # Upsample blocks (from lower level to upper level)
        # Up from level 3→2
        self.up_3_0 = Upsample(channels[3], channels[2])  # 512→256
        # Up from level 2→1
        self.up_2_0 = Upsample(channels[2], channels[1])  # 256→128
        self.up_2_1 = Upsample(channels[2], channels[1])  # 256→128
        # Up from level 1→0
        self.up_1_0 = Upsample(channels[1], channels[0])  # 128→64
        self.up_1_1 = Upsample(channels[1], channels[0])  # 128→64
        self.up_1_2 = Upsample(channels[1], channels[0])  # 128→64
        
        # ConvBlocks for each intermediate node
        self.conv_2_1 = ConvBlock(channels[2] + channels[2], channels[2])  # 512→256
        self.conv_1_1 = ConvBlock(channels[1] + channels[1], channels[1])  # 256→128
        self.conv_1_2 = ConvBlock(channels[1] * 3, channels[1])            # 384→128
        self.conv_0_1 = ConvBlock(channels[0] + channels[0], channels[0])  # 128→64
        self.conv_0_2 = ConvBlock(channels[0] * 3, channels[0])            # 192→64
        self.conv_0_3 = ConvBlock(channels[0] * 4, channels[0])            # 256→64
        
        # Deep supervision heads
        self.ds1 = DeepSupervisionHead(channels[0])  # from X(0,1)
        self.ds2 = DeepSupervisionHead(channels[0])  # from X(0,2)
    
    def forward(self, features):
        # features = [level0, level1, level2, level3]
        x_0_0, x_1_0, x_2_0, x_3_0 = features
        
        # Target sizes for each level
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
        
        # Deep supervision outputs (logits)
        ds1_out = self.ds1(x_0_1)
        ds2_out = self.ds2(x_0_2)
        
        return x_0_3, ds1_out, ds2_out
```

---

## Module 3: refinement.py

### Class: EdgeRefinement

```python
class EdgeRefinement(nn.Module):
    def __init__(self, ch=64):
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
```

---

## Module 4: losses.py

### Class: EdgeLoss

```python
class EdgeLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight_clamp=(5, 25)):
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight_clamp = pos_weight_clamp
    
    def forward(self, pred_logits, target):
        # Dynamic pos_weight (per batch)
        pos = target.sum()
        neg = target.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(*self.pos_weight_clamp)
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target,
            pos_weight=pos_weight.expand_as(pred_logits)
        )
        
        # Dice Loss
        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        
        return self.bce_weight * bce + self.dice_weight * dice
```

### Total Loss Computation

```python
def compute_total_loss(final_logits, ds1_logits, ds2_logits, target, loss_fn):
    loss_final = loss_fn(final_logits, target)
    loss_ds1 = loss_fn(ds1_logits, target)
    loss_ds2 = loss_fn(ds2_logits, target)
    return 1.0 * loss_final + 0.2 * loss_ds2 + 0.1 * loss_ds1
```

---

## Module 5: model.py

### Class: VGGTEdgeMask

```python
class VGGTEdgeMask(nn.Module):
    def __init__(self, vggt_model):
        self.feature_extractor = VGGTFeatureExtractor(vggt_model)
        self.decoder = UNetPPDecoder(channels=[64, 128, 256, 512])
        self.refinement = EdgeRefinement(ch=64)
        self.final_conv = nn.Conv2d(64, 1, 1)
    
    def forward(self, images):
        # images: [B, S, 3, 518, 518]
        B, S = images.shape[:2]
        
        # Extract multi-scale features
        features = self.feature_extractor(images)
        
        # UNet++ decode
        x_0_3, ds1_logits, ds2_logits = self.decoder(features)
        
        # Edge refinement
        x = self.refinement(x_0_3)
        
        # Final prediction
        logits = self.final_conv(x)  # [B*S, 1, 148, 148]
        logits = F.interpolate(logits, size=(518, 518),
                               mode="bilinear", align_corners=False)
        
        # Reshape to [B, S, 1, 518, 518]
        logits = logits.view(B, S, 1, 518, 518)
        ds1_logits = ds1_logits.view(B, S, 1, 518, 518)
        ds2_logits = ds2_logits.view(B, S, 1, 518, 518)
        
        if self.training:
            return logits, ds1_logits, ds2_logits
        else:
            return torch.sigmoid(logits)
```

---

## Complete Tensor Flow (Forward Pass)

```
# Input
images: [B, S, 3, 518, 518]

# Encoder (frozen, no_grad)
aggregated_tokens_list[4]:  [B, S, 1374, 2048]
aggregated_tokens_list[11]: [B, S, 1374, 2048]
aggregated_tokens_list[17]: [B, S, 1374, 2048]
aggregated_tokens_list[23]: [B, S, 1374, 2048]

# Slice patch tokens (remove camera + register)
layer_4_patches:  [B, S, 1369, 2048]
layer_11_patches: [B, S, 1369, 2048]
layer_17_patches: [B, S, 1369, 2048]
layer_23_patches: [B, S, 1369, 2048]

# Flatten B*S and reshape to spatial
layer_4_spatial:  [B*S, 2048, 37, 37]
layer_11_spatial: [B*S, 2048, 37, 37]
layer_17_spatial: [B*S, 2048, 37, 37]
layer_23_spatial: [B*S, 2048, 37, 37]

# Feature projections
level_0: [B*S, 64, 148, 148]    (from layer 4)
level_1: [B*S, 128, 74, 74]     (from layer 11)
level_2: [B*S, 256, 37, 37]     (from layer 17)
level_3: [B*S, 512, 19, 19]     (from layer 23)

# UNet++ decoder nodes
X(3,0): [B*S, 512, 19, 19]
X(2,0): [B*S, 256, 37, 37]
X(2,1): [B*S, 256, 37, 37]      ConvBlock(cat[256, 256]=512 → 256)
X(1,0): [B*S, 128, 74, 74]
X(1,1): [B*S, 128, 74, 74]      ConvBlock(cat[128, 128]=256 → 128)
X(1,2): [B*S, 128, 74, 74]      ConvBlock(cat[128, 128, 128]=384 → 128)
X(0,0): [B*S, 64, 148, 148]
X(0,1): [B*S, 64, 148, 148]     ConvBlock(cat[64, 64]=128 → 64)
X(0,2): [B*S, 64, 148, 148]     ConvBlock(cat[64, 64, 64]=192 → 64)
X(0,3): [B*S, 64, 148, 148]     ConvBlock(cat[64, 64, 64, 64]=256 → 64)

# Deep supervision (training only)
ds1: [B*S, 1, 518, 518]         from X(0,1)
ds2: [B*S, 1, 518, 518]         from X(0,2)

# Edge refinement
refined: [B*S, 64, 148, 148]    residual block on X(0,3)

# Final output
logits: [B*S, 1, 148, 148]      Conv1x1
logits: [B*S, 1, 518, 518]      bilinear upsample
logits: [B, S, 1, 518, 518]     reshape

# Inference
output: [B, S, 1, 518, 518]     sigmoid applied
```

---

## Parameter Count Breakdown

| Module | Parameters |
|--------|-----------|
| Projection Level 0 (2048→64 + smooth) | ~168K |
| Projection Level 1 (2048→128 + smooth) | ~410K |
| Projection Level 2 (2048→256) | ~525K |
| Projection Level 3 (2048→512 + down) | ~3.4M |
| Upsample blocks (6 total) | ~2.0M |
| ConvBlocks (6 total) | ~3.2M |
| Deep supervision heads (2) | ~37K |
| Edge refinement | ~74K |
| Final Conv1x1 | ~65 |
| **Total trainable** | **~9.9M** |

---

## GPU Memory Estimates

Assumes: encoder fp16, decoder mixed precision, S=1.

| Batch Size | Total Estimate |
|-----------|----------------|
| 1 | ~2.7 GB |
| 2 | ~3.8 GB |
| 4 | ~6.0 GB |
| 8 | ~10.4 GB |
