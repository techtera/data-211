# Flow

## High-Level Flow

### Training Flow

```
Input Images (518x518 PNG/JPG)
    ↓
EdgeMaskDataset (resize, to tensor, binarize mask at 0.5)
    ↓
DataLoader (DistributedSampler for DDP)
    ↓
StudentEdgeMask.forward() [training mode]
    ↓
StudentFeatureExtractor (frozen aggregator, no_grad)
    ↓
4× FeatureProjection → multi-scale feature pyramid
    ↓
UNet++ Decoder (nested skip connections)
    ↓
(logits, ds1_logits, ds2_logits)
    ↓
EdgeLoss: BCE(0.5) + Dice(0.5) × [ds1=0.1, ds2=0.2, final=1.0]
    ↓
Backward + Gradient Clipping (max_norm=1.0)
    ↓
Optimizer Step (AdamW) + Scheduler Step (cosine)
    ↓
Validation (F1, Precision, Recall, IoU)
    ↓
Checkpoint: last, best (F1), best_loss
    ↓
Early Stopping (patience=15 on F1)
```

### Inference Flow

```
Input Images [B, 3, 518, 518]
    ↓
StudentEdgeMask.forward() [eval mode]
    ↓
StudentFeatureExtractor → 4 feature maps
    ↓
UNet++ Decoder → x_0_3 (deep supervision heads skipped)
    ↓
EdgeRefinement → Final Conv
    ↓
Interpolate to 518x518
    ↓
sigmoid(logits) → Edge Probability Map [B, 1, 518, 518]
    ↓
Threshold (default 0.5) → Binary Edge Mask
```

### Evaluation Flow

```
Load Best Checkpoint
    ↓
Run Validation (model.train() mode for 3 outputs)
    ↓
Compute Per-Batch Metrics (Precision, Recall, F1, IoU)
    ↓
Compute Final Metrics:
    - BF1 (Boundary F1) at threshold 0.5
    - ODS (Optimal Dataset Scale) — best F1 across thresholds
    - Dice Score
    - Confusion Matrix (TP, FP, FN, TN)
```

## Detailed Flow

### Entry Point: `train_ddp.py`

1. Parse CLI args (--epochs)
2. Get WORLD_SIZE and RANK from environment (set by `torchrun`)
3. Call `train_ddp(rank, world_size, args)`

### DDP Training Setup (per GPU)

1. `setup_ddp(rank, world_size)` — init NCCL process group
2. Load student encoder checkpoint from `../kd-encoder/checkpoints/student_final.pt`
   - Extract `student_state_dict` from checkpoint
   - Create `StudentAggregator()`, load weights, freeze, eval mode
3. Build `StudentEdgeMask(student_aggregator)` → `.to(device)`
4. Wrap with `DDP(model, device_ids=[rank], find_unused_parameters=False)`
5. Create `EdgeMaskDataset(data/)` → `random_split` (90/10) → `DistributedSampler` → `DataLoader`
6. Build `EdgeLoss` (BCE+Dice with deep supervision weights)
7. Build optimizer: `AdamW(trainable_params, lr=3e-4, weight_decay=0.01)`
8. Build scheduler: `LambdaLR` with 5% linear warmup + cosine decay
9. Initialize early stopping state: `patience_counter=0, best_val_f1=0`

### Training Loop (`train_epoch_ddp`)

Per step:
1. `images.to(device)`, `masks.to(device)` — [B, 1, 3, 518, 518], [B, 1, 1, 518, 518]
2. Forward: `model(images)` → `(logits, ds1_logits, ds2_logits)`
3. Loss: `criterion(logits, ds1_logits, ds2_logits, masks)` — weighted sum of BCE+Dice across all 3 outputs
4. `optimizer.zero_grad()` → `loss.backward()` → `clip_grad_norm_(1.0)` → `optimizer.step()` → `scheduler.step()`
5. Accumulate epoch loss

### Validation (`validate_ddp`)

- Model stays in **training mode** (to get 3 outputs for deep supervision loss)
- `torch.no_grad()` prevents gradient computation
- Compute per-batch: Precision, Recall, F1, IoU at threshold 0.5
- Average across batches

### Checkpoint Saving (rank 0 only)

- Save last: `checkpoints/checkpoint_last.pt`
- Save best F1: `checkpoints/checkpoint_best.pt` (resets patience counter)
- Save best loss: `checkpoints/checkpoint_best_loss.pt`
- Early stopping: if `patience_counter >= 15`, break training loop

### Final Evaluation (rank 0 only)

1. Load `checkpoint_best.pt`
2. Run all val batches, collect logits and targets
3. Compute: Dice, BF1 (precision/recall/f1), ODS (best F1 across thresholds), confusion matrix

### Failure Points

- Student checkpoint not found at `../kd-encoder/checkpoints/student_final.pt`
- `data/` directory missing or empty
- Mask naming mismatch (must be `{stem}_mask.{ext}`)
- OOM: frozen encoder + decoder + gradients can be tight at batch_size=4
- NCCL timeout during DDP init
- Corrupt image/mask files (handled by dataset validation, but skipped silently)
