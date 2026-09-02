# Flow

## High-Level Flow

### Training Flow

```
Input Images (518x518 JPG/PNG)
    ↓
ImageDataset (resize, to tensor)
    ↓
DataLoader (DistributedSampler for DDP)
    ↓
Teacher Forward (frozen, no_grad, FP16)
    ↓
Token Sampling (teacher): 1374→133 tokens + save indices
    ↓
Delete teacher features (free memory)
    ↓
Student Forward (with grad)
    ↓
Token Sampling (student): use teacher's indices
    ↓
Projection: 1536→2048 per layer (4 heads)
    ↓
Loss: MSE + Cosine per layer, weighted sum
    ↓
Backward + Optimizer Step + Scheduler Step
    ↓
Checkpoint: last, best, periodic
```

### Inference Flow (Post-Distillation)

```
Input Images [B, S, 3, 518, 518]
    ↓
StudentAggregator.forward()
    ↓
Patch Embedding → [B*S, 1369, 768]
    ↓
Special Tokens → [B*S, 1374, 768]
    ↓
18× Frame+Global Attention
    ↓
Cached Features at [3, 8, 13, 17] → [B, S, 1374, 1536]
    ↓
Downstream Decoder (obj-mask or edge-mask)
```

### Evaluation Flow

```
Load Student Checkpoint
    ↓
Load Teacher (real VGGT)
    ↓
Load Test Images
    ↓
Extract Features (student layers [3,8,13,17], teacher layers [4,11,17,23])
    ↓
Compute: Cross-Correlation, Variance, Sparsity, Statistics
    ↓
Quality Assessment: EXCELLENT/GOOD/ACCEPTABLE/POOR
```

## Detailed Flow

### Entry Point: `train_ddp.py`

1. Parse CLI args (image_dir, epochs, batch_size, etc.)
2. Get WORLD_SIZE and RANK from environment (set by `torchrun`)
3. Call `train_ddp(rank, world_size, args)`

### DDP Training Setup (per GPU)

1. `setup_ddp(rank, world_size)` — init process group
2. Load teacher: `load_real_teacher()` → `VGGTUnified.aggregator`
   - Set `cached_layer_indices = {4, 11, 17, 23}`
   - Freeze all params
   - Wrap with `FeaturesOnlyWrapper` (drops `patch_start_idx` return)
3. Initialize student: `StudentAggregator(embed_dim=768, depth=18)`
   - `initialize_student_from_dinov2_large()` — project DINOv2-L weights
   - Wrap with `FeaturesOnlyWrapper` then `DDP`
4. Create dataset + `DistributedSampler` + `DataLoader`
5. Create `DistillationLoss`, optimizer (AdamW), scheduler (cosine+warmup)
6. Optionally load checkpoint for resume

### Training Loop (`train_epoch_ddp`)

Per step:
1. `images.to(device)` — [B, S, C, H, W]
2. Teacher forward (no_grad):
   - `teacher(images)` → list of features (None for uncached layers)
   - Filter non-None features → 4 tensors
   - `sample_tokens()` each → 133 tokens + indices
   - Delete teacher features + `empty_cache()`
3. Student forward:
   - `student(images)` → list of features
   - Filter non-None → 4 tensors
   - `sample_tokens_with_indices()` using teacher's indices
4. Loss computation:
   - Per layer: project 1536→2048, MSE + cosine, weighted
   - Divide by accumulation_steps
5. `loss.backward()` — accumulate gradients
6. Every `accumulation_steps`: `optimizer.step()`, `zero_grad()`, `scheduler.step()`

### Checkpoint Saving

- Main process only (rank 0)
- Unwrap: `student.module.model` (DDP → FeaturesOnlyWrapper → StudentAggregator)
- Save: student_state_dict, optimizer_state_dict, scheduler_step, epoch, loss, projection_state_dict
- Final: `save_student_only()` — just state_dict, no optimizer

### Failure Points

- OOM: Large batch sizes with gradient accumulation
- Teacher checkpoint not found at `../../vggt-unified/checkpoints/`
- DINOv2 download fails (first run)
- NCCL timeout during DDP init (network issues)
- Token sampling assertion fails if token count != 1374
