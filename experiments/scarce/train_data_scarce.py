"""
train_data_scarce.py — Data-scarce basin experiments.

Train LPU-Stream with limited data (1/3/5 years) and evaluate:
  - NSE (prediction accuracy)
  - PICP (uncertainty coverage)
  - Physics consistency metrics

Usage:
    python experiments/scarce/train_data_scarce.py --years 1
    python experiments/scarce/train_data_scarce.py --years 3 --model ea_lstm
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
from src.data.dataset import CamelsDataset, get_basin_list, create_dataloaders
from src.losses.pinball_loss import PinballLoss
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.data.data_preprocessing import VAL_START, VAL_END, TEST_START, TEST_END
from src.utils import get_device, set_seed

QUANTILES = [0.05, 0.5, 0.95]

# Physics constraint parameters
PRCP_STD = 7.58
DELTA_RAW = 2.0
DELTA_NORM = DELTA_RAW / PRCP_STD
LAMBDA_MONO = 0.01  # Reduced from 0.1 — less aggressive for scarce data
EXTREME_ALPHA = 2.0


def train_model(model, train_loader, val_loader, device, q95=None, with_physics=False, epochs=30, lr=1e-3):
    """Train quantile model with optional physics constraints.

    Early stopping monitors the *training* pinball loss (patience 5, max 30
    epochs); the baseline/ensemble scripts instead monitor validation loss.
    """
    pinball = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_loss, best_state, best_epoch = float("inf"), None, 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        mono_total, mono_n = 0.0, 0

        for dynamic, static, target, mask, _, _ in train_loader:
            dynamic, static = dynamic.to(device), static.to(device)
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(dynamic, static)

            # Extreme weighting
            if with_physics and q95 is not None:
                is_extreme = (target > q95).float()
                weights = 1.0 + EXTREME_ALPHA * is_extreme
            else:
                weights = torch.ones_like(target)

            # Weighted pinball loss
            per_sample = torch.zeros_like(target)
            for i, tau in enumerate(QUANTILES):
                q_pred = pred[:, i:i+1]
                error = target - q_pred
                per_sample = per_sample + torch.max(tau * error, (tau - 1) * error)
            per_sample = per_sample / len(QUANTILES)
            loss = (per_sample * weights * mask).sum() / mask.sum().clamp(min=1.0)

            # Monotonicity loss
            if with_physics:
                dynamic_aug = dynamic.clone()
                dynamic_aug[:, :, 0] = dynamic_aug[:, :, 0] + DELTA_NORM
                with torch.no_grad():
                    pred_aug = model(dynamic_aug, static)
                mono_violation = torch.nn.functional.relu(pred[:, 2:3] - pred_aug[:, 2:3])  # upper quantile
                mono_violation = mono_violation + torch.nn.functional.relu(pred[:, 0:1] - pred_aug[:, 0:1])  # lower
                mono_loss = mono_violation.mean()
                loss = loss + LAMBDA_MONO * mono_loss
                mono_total += mono_loss.item()
                mono_n += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()

        train_loss = total_loss / max(n, 1)
        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss, best_epoch = train_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_loss, best_epoch


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    for dynamic, static, target, mask, _, _ in loader:
        pred = model(dynamic.to(device), static.to(device))
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def compute_nse(pred, target, mask):
    median = pred[:, 1]
    valid = mask.flatten() > 0
    p, t = median[valid], target.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, required=True, choices=[1, 3, 5, 15])
    parser.add_argument("--n_basins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-physics", action="store_true")
    parser.add_argument("--no-static", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Select basins via uniform random sampling.
    all_basins = get_basin_list()
    rng = np.random.RandomState(args.seed)
    selected = rng.choice(all_basins, min(args.n_basins, len(all_basins)), replace=False).tolist()

    # Training period and seq_len scale with the data-scarcity level
    # (seq_len = 30/90/180/365 days for 1/3/5/15-yr).
    year_map = {1: "1981-09-30", 3: "1983-09-30", 5: "1985-09-30", 15: "1995-09-30"}
    seqlen_map = {1: 30, 3: 90, 5: 180, 15: 365}
    train_end = year_map[args.years]
    train_start = "1980-10-01"
    seq_len = seqlen_map[args.years]

    suffix = ""
    if args.with_physics: suffix += "_physics"
    if args.no_static: suffix += "_nostatic"
    print("=" * 60)
    print(f"Data-Scarce Experiment: {args.years} years, {len(selected)} basins")
    print(f"Physics: {args.with_physics}")
    print(f"Train: {train_start} to {train_end}")
    print(f"Seq_len: {seq_len}")
    print(f"Val:   {VAL_START} to {VAL_END}")
    print(f"Test:  {TEST_START} to {TEST_END}")
    print("=" * 60)

    # Create datasets with limited training data
    batch_size = 512 if args.years <= 3 else 1024

    print("\nCreating datasets...")
    train_ds = CamelsDataset(selected, train_start, train_end, seq_len, preview=True)
    val_ds = CamelsDataset(selected, VAL_START, VAL_END, seq_len, preview=True)
    test_ds = CamelsDataset(selected, TEST_START, TEST_END, seq_len, preview=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
                                                num_workers=4, persistent_workers=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Compute Q95 for extreme weighting
    q95 = train_ds.compute_q95() if args.with_physics else None
    if q95:
        print(f"Q95 (train): {q95:.4f}")

    # Train LPU-Stream Quantile
    physics_tag = " + Physics" if args.with_physics else ""
    print(f"\nTraining LPU-Stream Quantile{physics_tag}...")
    model = LPUStreamModel(quantiles=QUANTILES, no_static=args.no_static).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    best_loss, best_epoch = train_model(model, train_loader, val_loader, device,
                                         q95=q95, with_physics=args.with_physics)
    train_time = time.time() - t0
    print(f"Best epoch: {best_epoch}, loss: {best_loss:.4f}, time: {train_time:.0f}s")

    # Predict
    val_preds, val_targets, val_masks = predict(model, val_loader, device)
    test_preds, test_targets, test_masks = predict(model, test_loader, device)

    # NSE
    test_nse = compute_nse(test_preds, test_targets, test_masks)

    # CQR Calibration
    val_valid = val_masks.flatten() > 0
    calibrator = CQRCalibrator(alpha=0.1)
    q_cal = calibrator.fit(val_preds[val_valid, 0], val_preds[val_valid, 2], val_targets.flatten()[val_valid])

    # Uncalibrated metrics
    test_valid = test_masks.flatten() > 0
    test_y = test_targets.flatten()[test_valid]
    raw = compute_uncertainty_metrics(test_preds[test_valid, 0], test_preds[test_valid, 2], test_y, alpha=0.1)

    # Calibrated metrics
    cal_l, cal_u = calibrator.calibrate(test_preds[test_valid, 0], test_preds[test_valid, 2])
    cal = compute_uncertainty_metrics(cal_l, cal_u, test_y, alpha=0.1)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({args.years} years training)")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_ds)}, NSE: {test_nse:.4f}")
    print(f"CQR q_cal: {q_cal:.4f}")
    print(f"Uncalibrated PICP: {raw['picp']:.4f}, MPIW: {raw['mpiw']:.4f}")
    print(f"Calibrated   PICP: {cal['picp']:.4f}, MPIW: {cal['mpiw']:.4f}")
    print(f"Coverage low/normal/high: {cal['coverage_low_flow']:.4f}/{cal['coverage_normal_flow']:.4f}/{cal['coverage_high_flow']:.4f}")

    # Save
    results = {
        "experiment": "data_scarce",
        "years": args.years,
        "n_basins": len(selected),
        "train_samples": len(train_ds),
        "train_period": f"{train_start} to {train_end}",
        "test_nse": test_nse,
        "q_cal": float(q_cal),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "test_uncalibrated": raw,
        "test_calibrated": cal,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = PROJECT_ROOT / "results" / "tables" / f"scarce_{args.years}yr{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
