#!/usr/bin/env python3
"""
Training Monitor — run in a separate tmux pane.

Polls a checkpoint file, detects new epochs, logs to CSV,
and displays a live ASCII loss graph + table in the terminal.

Usage:
    # Monitor encoder training
    python monitor.py --checkpoint kd-encoder/checkpoints/checkpoint_last.pt

    # Monitor obj-mask decoder
    python monitor.py --checkpoint st-obj-mask/checkpoints/checkpoint_last.pt

    # Monitor edge-mask decoder
    python monitor.py --checkpoint st-edge-mask/checkpoints/checkpoint_last.pt

    # Custom poll interval (default 30s)
    python monitor.py --checkpoint kd-encoder/checkpoints/checkpoint_last.pt --interval 60

    # Seed with known history (epoch:loss pairs)
    python monitor.py --checkpoint kd-encoder/checkpoints/checkpoint_last.pt \
        --seed "11:0.114457,12:0.119473,13:0.118361,14:0.114246,15:0.113431,16:0.111445,17:0.113190"
"""

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


def load_epoch_and_loss(checkpoint_path):
    import torch
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", None)
        loss = ckpt.get("loss", None)
        return epoch, loss
    except Exception:
        return None, None


def load_existing_log(csv_path):
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "timestamp": row["timestamp"],
                })
    return records


def append_to_csv(csv_path, epoch, loss, timestamp):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "loss", "timestamp"])
        writer.writerow([epoch, f"{loss:.6f}", timestamp])


def draw_graph(records, graph_width=60, graph_height=18):
    """Draw an ASCII loss curve in the terminal."""
    if len(records) < 2:
        return ""

    losses = [r["loss"] for r in records]
    epochs = [r["epoch"] for r in records]

    lo = min(losses)
    hi = max(losses)
    margin = (hi - lo) * 0.1 or 0.001
    lo -= margin
    hi += margin

    y_label_width = 10
    lines = []

    lines.append("")
    lines.append("  Loss")

    canvas = [[" "] * graph_width for _ in range(graph_height)]

    for i, loss in enumerate(losses):
        x = int((i / (len(losses) - 1)) * (graph_width - 1)) if len(losses) > 1 else 0
        y = int(((loss - lo) / (hi - lo)) * (graph_height - 1))
        y = graph_height - 1 - y
        y = max(0, min(graph_height - 1, y))
        canvas[y][x] = "●"

        if i > 0:
            prev_x = int(((i - 1) / (len(losses) - 1)) * (graph_width - 1))
            prev_y = int(((losses[i - 1] - lo) / (hi - lo)) * (graph_height - 1))
            prev_y = graph_height - 1 - prev_y
            prev_y = max(0, min(graph_height - 1, prev_y))

            dx = x - prev_x
            if dx > 1:
                for step in range(1, dx):
                    frac = step / dx
                    ix = prev_x + step
                    iy = prev_y + int(frac * (y - prev_y))
                    iy = max(0, min(graph_height - 1, iy))
                    if canvas[iy][ix] == " ":
                        canvas[iy][ix] = "·"

    for row_idx in range(graph_height):
        val = hi - (row_idx / (graph_height - 1)) * (hi - lo)
        if row_idx == 0 or row_idx == graph_height - 1 or row_idx == graph_height // 2:
            label = f"{val:.5f}"
            prefix = f"  {label:>{y_label_width}} │"
        else:
            prefix = f"  {'':>{y_label_width}} │"
        lines.append(prefix + "".join(canvas[row_idx]))

    lines.append(f"  {'':>{y_label_width}} └" + "─" * graph_width)

    ep_start = str(epochs[0])
    ep_end = str(epochs[-1])
    ep_mid = str(epochs[len(epochs) // 2])
    axis_line = f"  {'':>{y_label_width}}  {ep_start}"
    mid_pos = graph_width // 2 - len(ep_mid) // 2
    end_pos = graph_width - len(ep_end)
    padded = list(" " * graph_width)
    for ci, ch in enumerate(ep_start):
        if ci < graph_width:
            padded[ci] = ch
    for ci, ch in enumerate(ep_mid):
        pos = mid_pos + ci
        if 0 <= pos < graph_width:
            padded[pos] = ch
    for ci, ch in enumerate(ep_end):
        pos = end_pos + ci
        if 0 <= pos < graph_width:
            padded[pos] = ch
    lines.append(f"  {'':>{y_label_width}}  " + "".join(padded))
    lines.append(f"  {'':>{y_label_width}}  " + " " * (graph_width // 2 - 2) + "Epoch")

    return "\n".join(lines)


def print_display(records):
    os.system("clear" if os.name != "nt" else "cls")

    if not records:
        print("No epochs logged yet. Waiting for checkpoint...\n")
        return

    term_width = shutil.get_terminal_size((80, 40)).columns
    graph_width = max(30, min(term_width - 16, 80))

    graph = draw_graph(records, graph_width=graph_width)
    if graph:
        print(graph)
        print()

    print("=" * 58)
    print(f"  Training Monitor — {len(records)} epochs logged")
    print("=" * 58)

    display_records = records[-20:] if len(records) > 20 else records
    if len(records) > 20:
        print(f"  (showing last 20 of {len(records)} epochs)")
        print()

    print(f"  {'Epoch':>6}  {'Loss':>12}  {'Delta':>10}  {'Trend':>5}")
    print("-" * 58)

    best_loss = min(r["loss"] for r in records)
    best_epoch = min((r for r in records if r["loss"] == best_loss), key=lambda r: r["epoch"])["epoch"]

    all_sorted = sorted(records, key=lambda r: r["epoch"])
    loss_by_epoch = {r["epoch"]: r["loss"] for r in all_sorted}

    for i, rec in enumerate(display_records):
        epoch = rec["epoch"]
        loss = rec["loss"]

        idx_in_all = next(j for j, r in enumerate(all_sorted) if r["epoch"] == epoch)

        if idx_in_all == 0:
            delta_str = ""
            trend = ""
        else:
            prev_loss = all_sorted[idx_in_all - 1]["loss"]
            delta = loss - prev_loss
            delta_str = f"{delta:+.6f}"
            trend = "↓" if delta < 0 else "↑" if delta > 0 else "="

        marker = " ★" if epoch == best_epoch else ""
        print(f"  {epoch:>6}  {loss:>12.6f}  {delta_str:>10}  {trend:>5}{marker}")

    print("-" * 58)
    print(f"  Best: Epoch {best_epoch}, Loss {best_loss:.6f}")

    if len(records) >= 2:
        total_drop = all_sorted[0]["loss"] - all_sorted[-1]["loss"]
        avg_drop = total_drop / (len(all_sorted) - 1)
        print(f"  Total Δ: {total_drop:+.6f}  |  Avg Δ/epoch: {avg_drop:+.6f}")

    print("=" * 58)
    print(f"  Last check: {datetime.now().strftime('%H:%M:%S')}")
    print()


def parse_seed(seed_str):
    """Parse seed string like '11:0.114,12:0.119' into records."""
    records = []
    for pair in seed_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        ep_str, loss_str = pair.split(":", 1)
        records.append({
            "epoch": int(ep_str.strip()),
            "loss": float(loss_str.strip()),
            "timestamp": "seed",
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Monitor training loss from checkpoint files")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_last.pt")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("--log", default=None, help="CSV log path (default: <checkpoint_dir>/training_log.csv)")
    parser.add_argument("--seed", default=None, help="Seed with known history: '11:0.114,12:0.119,...'")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    csv_path = args.log or str(Path(checkpoint_path).parent / "training_log.csv")

    print(f"Monitoring: {checkpoint_path}")
    print(f"Logging to: {csv_path}")
    print(f"Poll interval: {args.interval}s")
    print(f"Press Ctrl+C to stop.\n")

    records = load_existing_log(csv_path)
    seen_epochs = {r["epoch"] for r in records}

    if args.seed:
        seed_records = parse_seed(args.seed)
        for sr in seed_records:
            if sr["epoch"] not in seen_epochs:
                records.append(sr)
                seen_epochs.add(sr["epoch"])
                append_to_csv(csv_path, sr["epoch"], sr["loss"], sr["timestamp"])

    records.sort(key=lambda r: r["epoch"])

    if os.path.exists(checkpoint_path):
        epoch, loss = load_epoch_and_loss(checkpoint_path)
        if epoch is not None and loss is not None and epoch not in seen_epochs:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            records.append({"epoch": epoch, "loss": loss, "timestamp": timestamp})
            seen_epochs.add(epoch)
            append_to_csv(csv_path, epoch, loss, timestamp)
            records.sort(key=lambda r: r["epoch"])

    last_mtime = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0

    print_display(records)

    try:
        while True:
            time.sleep(args.interval)

            if not os.path.exists(checkpoint_path):
                continue

            mtime = os.path.getmtime(checkpoint_path)
            if mtime == last_mtime:
                continue
            last_mtime = mtime

            epoch, loss = load_epoch_and_loss(checkpoint_path)
            if epoch is None or loss is None:
                continue

            if epoch in seen_epochs:
                continue

            seen_epochs.add(epoch)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            records.append({"epoch": epoch, "loss": loss, "timestamp": timestamp})
            records.sort(key=lambda r: r["epoch"])

            append_to_csv(csv_path, epoch, loss, timestamp)
            print_display(records)

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        print(f"Log saved: {csv_path}")


if __name__ == "__main__":
    main()
