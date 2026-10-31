"""Train the 671-basin LPU-Stream point model behind the MC-dropout method.

Restored from experiments/baseline/train_model.py (removed by the library
restructure): same architecture, sequence length, optimizer, and
minimum-validation-MSE checkpoint selection that produced the archived
results/checkpoints/lpu_stream_best.pt. Retraining is deterministic in seed
but not bit-identical across torch/CUDA versions, so the archived checkpoint
plus its SHA-256 in the frozen tables remains the reference; this script
exists so the construction itself is reproducible.

Writes results/checkpoints/lpu_stream_best.pt (or lpu_stream_seed{N}_best.pt
with --seed) and results/tables/full_checkpoint_training/point_seed{N}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders, get_basin_list  # noqa: E402
from src.models.lpu_stream import LPUStreamModel  # noqa: E402
from src.utils import set_seed, sha256_file  # noqa: E402

CONFIG = {
    "seq_len": 365,
    "n_dynamic": 15,
    "n_static": 13,
    "hidden_size": 128,
    "embed_dim": 32,
    "dropout": 0.3,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "epochs": 30,
    "seed": 42,
}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_valid = 0.0, 0
    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(dynamic, static)
        loss = (criterion(pred, target) * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n_valid += mask.sum().item()
    return total_loss / max(n_valid, 1.0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_valid = 0.0, 0
    all_preds, all_targets, all_masks = [], [], []
    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)
        pred = model(dynamic, static)
        total_loss += (criterion(pred, target) * mask).sum().item()
        n_valid += mask.sum().item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    return total_loss / max(n_valid, 1.0), preds, targets, masks


def compute_nse(preds, targets, masks):
    valid = masks.flatten() > 0
    if valid.sum() == 0:
        return 0.0
    p, t = preds.flatten()[valid], targets.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    seed_tag = f"_seed{CONFIG['seed']}" if args.seed is not None else ""

    ckpt_dir = ROOT / "results" / "checkpoints"
    best_path = ckpt_dir / f"lpu_stream{seed_tag}_best.pt"
    results_path = (
        ROOT / "results" / "tables" / "full_checkpoint_training" / f"point_seed{CONFIG['seed']}.json"
    )
    if best_path.exists() and results_path.exists():
        try:
            recorded = json.loads(results_path.read_text(encoding="utf-8"))
            if recorded.get("checkpoint_sha256") == sha256_file(best_path):
                print(f"SKIP completed {best_path.name}", flush=True)
                return
        except (OSError, ValueError, KeyError):
            pass

    print(f"lpu_stream point training - config: {json.dumps(CONFIG, indent=2)}", flush=True)
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, val_loader, _ = create_dataloaders(
        seq_len=CONFIG["seq_len"], batch_size=CONFIG["batch_size"],
        basin_list=get_basin_list(), num_workers=4,
    )
    model = LPUStreamModel(
        CONFIG["n_dynamic"], CONFIG["n_static"], CONFIG["hidden_size"],
        CONFIG["embed_dim"], CONFIG["dropout"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss, best_epoch = float("inf"), 0
    patience, patience_counter = 5, 0
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, vp, vt, vm = evaluate(model, val_loader, criterion, device)
        val_nse = compute_nse(vp, vt, vm)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']} | Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | NSE: {val_nse:.4f} | {time.time() - t0:.0f}s",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "val_nse": val_nse,
                "checkpoint_selection": "minimum_validation_mse",
                "config": CONFIG,
            }, best_path)
            print(f"  -> Saved best (val_loss={val_loss:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    results = {
        "model": "lpu_stream_point",
        "model_seed": CONFIG["seed"],
        "config": CONFIG,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_val_loss),
        "checkpoint_selection": "minimum_validation_mse",
        "training_run_complete": True,
        "checkpoint": best_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(best_path),
        "timestamp": datetime.now().isoformat(),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Checkpoint metadata saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
