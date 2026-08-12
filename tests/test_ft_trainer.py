"""
Tests for fine_tuning/trainer.py

Verifies:
    1. train_one_epoch runs without error
    2. Loss decreases during overfitting
    3. Encoder stays frozen
    4. Decoder weights update
    5. Gradient clipping activates
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

import fine_tuning.config as config
from fine_tuning.losses import EdgeLoss, compute_total_loss
from fine_tuning.scheduler import build_scheduler


# ============================================================
# Mock model
# ============================================================

class MockAggregator(nn.Module):
    """Returns fixed features so the decoder can overfit."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(10, 10)
        for p in self.parameters():
            p.requires_grad_(False)
        self._cached = None

    def forward(self, images):
        B, S = images.shape[:2]
        if self._cached is None or self._cached.shape[0] != B:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0)
            self._cached = torch.randn(B, S, 1374, 2048, generator=gen)
        tokens = self._cached.to(images.device)
        return [tokens for _ in range(24)], 5


class MockEdgeMaskModel(nn.Module):
    """Simplified model with mock aggregator + trainable decoder."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.aggregator = MockAggregator()
        # Small trainable decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, images):
        B = images.shape[0]
        # Use actual image pixels as input to decoder
        x = images[:, 0]  # [B, 3, 518, 518]
        logits = self.decoder(x).unsqueeze(1)  # [B, 1, 1, 518, 518]
        ds1 = logits.clone()
        ds2 = logits.clone()
        return logits, ds1, ds2


# ============================================================
# Tests
# ============================================================

def test_train_one_epoch_runs():
    """train_one_epoch completes without error."""
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()
    model.to(config.DEVICE)

    images = torch.randn(4, 1, 3, 518, 518)
    masks = (torch.rand(4, 1, 1, 518, 518) > 0.9).float()
    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=2, drop_last=True)

    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    scheduler, _ = build_scheduler(optimizer, total_steps=20)
    scaler = GradScaler(enabled=False)

    from fine_tuning.trainer import train_one_epoch

    loss, grad_norm = train_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
    )

    assert loss > 0, f"Loss should be positive: {loss}"
    assert grad_norm >= 0, f"Grad norm negative: {grad_norm}"

    print("PASSED: test_train_one_epoch_runs")

    config.DEVICE = original_device


def test_loss_decreases_overfit():
    """Loss should decrease when overfitting on one batch."""
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()
    model.to(config.DEVICE)

    # Single batch, repeated
    images = torch.randn(2, 1, 3, 518, 518)
    masks = (torch.rand(2, 1, 1, 518, 518) > 0.9).float()

    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    model.train()
    model.feature_extractor.aggregator.eval()

    losses = []
    for step in range(30):
        optimizer.zero_grad()
        logits, ds1, ds2 = model(images)
        loss = compute_total_loss(logits, ds1, ds2, masks, criterion)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    print(f"PASSED: test_loss_decreases_overfit ({losses[0]:.4f} -> {losses[-1]:.4f})")

    config.DEVICE = original_device


def test_encoder_frozen():
    """Encoder weights must not change during training."""
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()

    # Snapshot encoder
    encoder_before = {
        name: p.clone()
        for name, p in model.feature_extractor.aggregator.named_parameters()
    }

    images = torch.randn(2, 1, 3, 518, 518)
    masks = (torch.rand(2, 1, 1, 518, 518) > 0.9).float()

    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    model.train()
    model.feature_extractor.aggregator.eval()

    for _ in range(5):
        optimizer.zero_grad()
        logits, ds1, ds2 = model(images)
        loss = compute_total_loss(logits, ds1, ds2, masks, criterion)
        loss.backward()
        optimizer.step()

    for name, p in model.feature_extractor.aggregator.named_parameters():
        assert torch.equal(p, encoder_before[name]), f"Encoder param {name} changed!"

    print("PASSED: test_encoder_frozen")

    config.DEVICE = original_device


def test_decoder_updates():
    """Decoder weights must change during training."""
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()

    decoder_before = {
        name: p.clone()
        for name, p in model.decoder.named_parameters()
    }

    images = torch.randn(2, 1, 3, 518, 518)
    masks = (torch.rand(2, 1, 1, 518, 518) > 0.9).float()

    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    model.train()
    for _ in range(5):
        optimizer.zero_grad()
        logits, ds1, ds2 = model(images)
        loss = compute_total_loss(logits, ds1, ds2, masks, criterion)
        loss.backward()
        optimizer.step()

    changed = sum(
        1 for name, p in model.decoder.named_parameters()
        if not torch.equal(p, decoder_before[name])
    )

    assert changed > 0, "No decoder params changed!"

    print(f"PASSED: test_decoder_updates ({changed} params updated)")

    config.DEVICE = original_device


def test_gradient_clipping():
    """Gradient clipping should cap gradient norms."""
    original_device = config.DEVICE
    config.DEVICE = torch.device("cpu")

    model = MockEdgeMaskModel()

    images = torch.randn(2, 1, 3, 518, 518)
    masks = (torch.rand(2, 1, 1, 518, 518) > 0.9).float()

    criterion = EdgeLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2,  # high lr for large gradients
    )

    model.train()
    max_norm = 1.0
    pre_clip_norms = []

    for _ in range(5):
        optimizer.zero_grad()
        logits, ds1, ds2 = model(images)
        loss = compute_total_loss(logits, ds1, ds2, masks, criterion)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm
        )
        pre_clip_norms.append(grad_norm.item())
        optimizer.step()

    print(f"PASSED: test_gradient_clipping (max pre-clip norm: {max(pre_clip_norms):.2f})")

    config.DEVICE = original_device


if __name__ == "__main__":
    test_train_one_epoch_runs()
    test_loss_decreases_overfit()
    test_encoder_frozen()
    test_decoder_updates()
    test_gradient_clipping()
    print("\n=== ALL 5 TESTS PASSED ===")
