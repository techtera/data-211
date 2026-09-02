# Flow

## High-Level Flow

### Training Flow

```
Input Images (518x518 JPG/PNG)
    ↓
SegmentationDataset (YOLO polygon → binary mask, resize 518x518)
    ↓
DataLoader (DistributedSampler for DDP)
    ↓
StudentObjMask.forward():
    ↓
    StudentAggregator (frozen, no_grad)
        ↓
    Cached Features at [3, 8, 13, 17]
        ↓
    ObjMaskDecoder (DPTHead):
        Strip Special Tokens → LayerNorm → Project → Pyramid → SegFormer → Upsample
        ↓
    Logits [B, 2, 518, 518]
    ↓
Loss: CrossEntropy + Dice
    ↓
Backward → Grad Clip (1.0) → Optimizer Step → Scheduler Step
    ↓
Validation: mIoU, Dice, Pixel Accuracy
    ↓
Checkpoint: last, best (mIoU), best (loss)
    ↓
Early Stopping (patience=15 on val mIoU)
```

### Inference Flow

```
Input Images [B, 3, 518, 518]
    ↓
StudentObjMask.forward()
    ↓
Logits [B, 2, 518, 518]
    ↓
argmax(dim=1)
    ↓
Binary Mask [B, 518, 518]
```

## Detailed Flow

### Entry Point: `train_ddp.py`

1. Parse CLI args (epochs)
2. Get WORLD_SIZE and RANK from environment (set by `torchrun`)
3. Call `train_ddp(rank, world_size, args)`

### DDP Training Setup (per GPU)

1. `setup_ddp(rank, world_size)` — init NCCL process group
2. Load student encoder from `../kd-encoder/checkpoints_v2/student_final.pt`
   - Create `StudentAggregator()`, load state_dict
   - Set eval mode, freeze all parameters
3. Build `StudentObjMask(student_aggregator)` → move to device
4. Wrap with `DDP(model, device_ids=[rank])`
5. Create `SegmentationDataset` from `data/` directory
   - Split 90/10 train/val with `random_split` (seed=42)
   - Create `DistributedSampler` for each split
   - Create `DataLoader` (batch_size=2, num_workers=4, pin_memory, drop_last for train)
6. Build loss (`SegmentationLoss`: CE + Dice)
7. Build optimizer (`AdamW`, LR=1e-4, weight_decay=1e-2, trainable params only)
8. Build scheduler (warmup 5% + cosine decay to 0)

### Training Loop

Per epoch:
1. `train_sampler.set_epoch(epoch)` — reshuffle
2. `train_epoch_ddp()`:
   - Per batch: forward → loss → backward → grad_clip → optimizer.step → scheduler.step
   - Log every 10 steps (loss, LR, ETA)
3. `validate_ddp()`:
   - model.eval(), torch.no_grad()
   - Compute val_loss, mIoU, Dice, Pixel Accuracy
4. Save checkpoints (rank 0 only):
   - checkpoint_last.pt (always)
   - checkpoint_best.pt (if val_miou improved — resets patience)
   - checkpoint_best_loss.pt (if val_loss improved)
5. Early stopping: break if patience_counter >= 15

### Final Evaluation

After training completes:
1. Load `checkpoint_best.pt`
2. Run validation set through model
3. Compute full metrics + confusion matrix
4. Print summary

### Failure Points

- Student checkpoint not found at `../kd-encoder/checkpoints_v2/student_final.pt`
- Dataset `data/` directory missing or empty
- OOM with batch_size > 2 on smaller GPUs
- NCCL timeout during DDP init
- Label files with invalid polygon format (< 3 vertices)
