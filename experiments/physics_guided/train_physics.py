"""
train_physics.py — Train LPU-Stream with physics constraints.

Usage:
    python experiments/physics_guided/train_physics.py
    python experiments/physics_guided/train_physics.py --no-mono --no-wb  # ablation

Paper reference: Section 10 (Physics constraints), Section 16.3 (Phase 3 plan)
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.physics_loss import PhysicsLoss
from src.utils import set_seed

# ─── Config ────────────────────────────────────────────────────────────────
CONFIG = {
    # Model
    "n_dynamic": 5, "n_static": 13, "hidden_size": 128,
    "embed_dim": 32, "dropout": 0.3,
    # Training
    "seq_len": 365, "batch_size": 1024, "learning_rate": 1e-3,
    "epochs": 30, "seed": 42,
    # Physics
    "prcp_std": 7.58, "delta_raw": 2.0,
    "lambda_nonneg": 0.05, "lambda_mono": 0.1, "lambda_wb": 0.1,
    "extreme_alpha": 2.0,
}


def train_epoch(model, loader, optimizer, physics_loss, device):
    model.train()
    total_loss, n_valid = 0.0, 0
    epoch_mono, epoch_wb, epoch_nonneg, n_batches = 0.0, 0.0, 0.0, 0

    for dynamic, static, target, mask, basin_idx, year_idx in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)
        basin_idx = basin_idx.to(device)
        year_idx = year_idx.to(device)

        optimizer.zero_grad()
        pred = model(dynamic, static)

        losses = physics_loss(
            model, dynamic, static, pred, target, mask,
            basin_idx, year_idx,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_valid = mask.sum().item()
        total_loss += losses["mse"].item() * batch_valid
        epoch_mono += losses["mono"].item()
        epoch_wb += losses["wb"].item()
        epoch_nonneg += losses["nonneg"].item()
        n_valid += batch_valid
        n_batches += 1

    return {
        "mse": total_loss / max(n_valid, 1.0),
        "mono": epoch_mono / max(n_batches, 1),
        "wb": epoch_wb / max(n_batches, 1),
        "nonneg": epoch_nonneg / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    total_loss, n_valid = 0.0, 0
    criterion = nn.MSELoss(reduction="none")

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)
        pred = model(dynamic, static)
        total_loss += (criterion(pred, target) * mask).sum().item()
        n_valid += mask.sum().item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    avg_loss = total_loss / max(n_valid, 1.0)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)

    valid = masks.flatten() > 0
    p, t = preds.flatten()[valid], targets.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return avg_loss, nse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nonneg", action="store_true")
    parser.add_argument("--no-mono", action="store_true")
    parser.add_argument("--no-wb", action="store_true")
    parser.add_argument("--no-extreme", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = {**CONFIG}
    if args.epochs:
        cfg["epochs"] = args.epochs

    print("=" * 60)
    print("LPU-Stream + Physics Constraints")
    print("=" * 60)
    flags = []
    if args.no_nonneg: flags.append("no-nonneg")
    if args.no_mono: flags.append("no-mono")
    if args.no_wb: flags.append("no-wb")
    if args.no_extreme: flags.append("no-extreme")
    print(f"Ablation flags: {flags if flags else 'none (full physics)'}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
        basin_list=get_basin_list(),
    )

    # Compute Q95 from training data
    train_ds = train_loader.dataset
    q95 = train_ds.compute_q95()
    print(f"Q95 threshold (normalized log1p): {q95:.4f}")

    # Model
    model = LPUStreamModel(
        n_dynamic=cfg["n_dynamic"], n_static=cfg["n_static"],
        hidden_size=cfg["hidden_size"], embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # Physics loss
    physics_loss = PhysicsLoss(
        q95=q95,
        prcp_std=cfg["prcp_std"],
        delta_raw=cfg["delta_raw"],
        lambda_nonneg=cfg["lambda_nonneg"],
        lambda_mono=cfg["lambda_mono"],
        lambda_wb=cfg["lambda_wb"],
        extreme_alpha=cfg["extreme_alpha"],
        use_nonneg=not args.no_nonneg,
        use_mono=not args.no_mono,
        use_wb=not args.no_wb,
        use_extreme=not args.no_extreme,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Training
    print(f"\nTraining for {cfg['epochs']} epochs...")
    print("-" * 80)

    best_val_loss, best_epoch = float("inf"), 0
    patience, patience_counter = 5, 0
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_physics"
    if flags:
        suffix += "_" + "_".join(flags)
    best_path = ckpt_dir / f"lpu_stream{suffix}_best.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, physics_loss, device)
        val_loss, val_nse = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"MSE: {train_metrics['mse']:.4f} | "
              f"NonNeg: {train_metrics['nonneg']:.4f} | "
              f"Mono: {train_metrics['mono']:.4f} | "
              f"WB: {train_metrics['wb']:.4f} | "
              f"Val NSE: {val_nse:.4f} | "
              f"LR: {lr:.1e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "val_nse": val_nse, "config": cfg,
                "q95": q95, "flags": flags,
            }, best_path)
            print(f"  -> Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_nse = evaluate(model, test_loader, device)

    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test NSE: {test_nse:.4f}")

    # Save results
    results = {
        "model": f"lpu_stream{suffix}",
        "config": cfg, "flags": flags,
        "q95": q95,
        "best_epoch": best_epoch, "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss), "test_nse": float(test_nse),
        "n_params": n_params, "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / f"lpu_stream{suffix}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
