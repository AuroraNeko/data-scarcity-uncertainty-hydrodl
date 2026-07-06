"""
train_quantile.py — Train LPU-Stream with quantile regression + CQR calibration.

Usage:
    python experiments/uncertainty/train_quantile.py

Paper reference: Section 11 (Uncertainty Calibration)
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.pinball_loss import PinballLoss
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.utils import set_seed

# ─── Config ────────────────────────────────────────────────────────────────
# Training hyperparameters. Early stopping / LR scheduling monitor the
# *training* loss (patience 5, max 30 epochs); batch size 1024; dropout 0.3.
QUANTILES = [0.05, 0.5, 0.95]
CONFIG = {
    "n_dynamic": 15, "n_static": 13, "hidden_size": 128,
    "embed_dim": 32, "dropout": 0.3, "quantiles": QUANTILES,
    "seq_len": 365, "batch_size": 1024, "learning_rate": 1e-3,
    "epochs": 30, "seed": 42, "alpha": 0.1,
}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_valid = 0.0, 0

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(dynamic, static)  # (batch, n_quantiles)

        loss = criterion(pred, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_valid = mask.sum().item()
        total_loss += loss.item() * batch_valid
        n_valid += batch_valid

    return total_loss / max(n_valid, 1.0)


@torch.no_grad()
def predict(model, loader, device):
    """Return predictions for all quantiles + targets + masks."""
    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        pred = model(dynamic, static)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    preds = np.concatenate(all_preds)      # (N, n_quantiles)
    targets = np.concatenate(all_targets)   # (N, 1)
    masks = np.concatenate(all_masks)       # (N, 1)
    return preds, targets, masks


def compute_nse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute NSE using median (index 1 = Q_0.5)."""
    median = pred[:, 1:2]
    valid = mask.flatten() > 0
    p, t = median.flatten()[valid], target.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    cfg = {**CONFIG}
    print("=" * 60)
    print("LPU-Stream Quantile Regression + CQR Calibration")
    print("=" * 60)
    print(f"Quantiles: {QUANTILES}")
    print(f"Target coverage: {1 - cfg['alpha']:.0%}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
        basin_list=get_basin_list(),
    )

    # Model
    model = LPUStreamModel(
        n_dynamic=cfg["n_dynamic"], n_static=cfg["n_static"],
        hidden_size=cfg["hidden_size"], embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"], quantiles=QUANTILES,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # Loss
    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Training
    print(f"\nTraining for {cfg['epochs']} epochs...")
    print("-" * 60)

    best_val_loss, best_epoch = float("inf"), 0
    patience, patience_counter = 5, 0
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lpu_stream_quantile_best.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation: predict + compute NSE from median
        val_preds, val_targets, val_masks = predict(model, val_loader, device)
        val_nse = compute_nse(val_preds, val_targets, val_masks)
        # Early stopping / LR scheduling use the *training* loss below (a full
        # validation pinball pass is comparatively expensive). val_nse is
        # reported for monitoring only.
        val_loss = train_loss
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | "
              f"Val NSE: {val_nse:.4f} | "
              f"LR: {lr:.1e} | {elapsed:.0f}s")

        if train_loss < best_val_loss:
            best_val_loss, best_epoch = train_loss, epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_nse": val_nse, "config": cfg,
            }, best_path)
            print(f"  -> Saved best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ─── CQR Calibration + Evaluation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("CQR Calibration & Uncertainty Evaluation")
    print("=" * 60)

    # Load best model
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Predict on validation (calibration) set
    val_preds, val_targets, val_masks = predict(model, val_loader, device)
    valid_val = val_masks.flatten() > 0
    val_q_lower = val_preds[valid_val, 0]  # Q_0.05
    val_q_upper = val_preds[valid_val, 2]  # Q_0.95
    val_y = val_targets.flatten()[valid_val]

    # Fit CQR on validation set
    calibrator = CQRCalibrator(alpha=cfg["alpha"])
    q_cal = calibrator.fit(val_q_lower, val_q_upper, val_y)
    print(f"CQR calibration quantile: {q_cal:.4f}")

    # ── Uncalibrated metrics (validation) ──
    val_metrics_raw = compute_uncertainty_metrics(
        val_q_lower, val_q_upper, val_y, alpha=cfg["alpha"],
    )
    print(f"\nValidation (uncalibrated):")
    print(f"  PICP: {val_metrics_raw['picp']:.4f} (target: {1-cfg['alpha']:.2f})")
    print(f"  MPIW: {val_metrics_raw['mpiw']:.4f}")
    print(f"  Coverage low/normal/high: "
          f"{val_metrics_raw['coverage_low_flow']:.4f}/"
          f"{val_metrics_raw['coverage_normal_flow']:.4f}/"
          f"{val_metrics_raw['coverage_high_flow']:.4f}")

    # ── Calibrated metrics (validation) ──
    cal_lower, cal_upper = calibrator.calibrate(val_q_lower, val_q_upper)
    val_metrics_cal = compute_uncertainty_metrics(
        cal_lower, cal_upper, val_y, alpha=cfg["alpha"],
    )
    print(f"\nValidation (CQR calibrated):")
    print(f"  PICP: {val_metrics_cal['picp']:.4f} (target: {1-cfg['alpha']:.2f})")
    print(f"  MPIW: {val_metrics_cal['mpiw']:.4f}")
    print(f"  Coverage low/normal/high: "
          f"{val_metrics_cal['coverage_low_flow']:.4f}/"
          f"{val_metrics_cal['coverage_normal_flow']:.4f}/"
          f"{val_metrics_cal['coverage_high_flow']:.4f}")

    # ── Test set evaluation ──
    test_preds, test_targets, test_masks = predict(model, test_loader, device)
    valid_test = test_masks.flatten() > 0
    test_q_lower = test_preds[valid_test, 0]
    test_q_upper = test_preds[valid_test, 2]
    test_y = test_targets.flatten()[valid_test]

    test_nse = compute_nse(test_preds, test_targets, test_masks)

    # Uncalibrated
    test_raw = compute_uncertainty_metrics(
        test_q_lower, test_q_upper, test_y, alpha=cfg["alpha"],
    )

    # Calibrated
    test_cal_lower, test_cal_upper = calibrator.calibrate(test_q_lower, test_q_upper)
    test_cal = compute_uncertainty_metrics(
        test_cal_lower, test_cal_upper, test_y, alpha=cfg["alpha"],
    )

    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"NSE (median): {test_nse:.4f}")
    print(f"\nUncalibrated 90% interval:")
    print(f"  PICP: {test_raw['picp']:.4f} | MPIW: {test_raw['mpiw']:.4f} | Winkler: {test_raw['winkler_score']:.4f}")
    print(f"  Coverage low/normal/high: {test_raw['coverage_low_flow']:.4f}/{test_raw['coverage_normal_flow']:.4f}/{test_raw['coverage_high_flow']:.4f}")
    print(f"\nCQR Calibrated 90% interval:")
    print(f"  PICP: {test_cal['picp']:.4f} | MPIW: {test_cal['mpiw']:.4f} | Winkler: {test_cal['winkler_score']:.4f}")
    print(f"  Coverage low/normal/high: {test_cal['coverage_low_flow']:.4f}/{test_cal['coverage_normal_flow']:.4f}/{test_cal['coverage_high_flow']:.4f}")

    # Save results
    results = {
        "model": "lpu_stream_quantile",
        "config": cfg,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "q_cal": float(q_cal),
        "test_nse": float(test_nse),
        "test_uncalibrated": test_raw,
        "test_calibrated": test_cal,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / "lpu_stream_quantile_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
