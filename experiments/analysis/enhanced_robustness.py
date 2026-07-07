"""Enhanced robustness analyses for the manuscript.

Computes three checks from model predictions rather than hard-coded numbers:

1. Calibration-window sensitivity for full-dataset CQR.
2. Basin-cluster bootstrap confidence intervals for uncertainty metrics.
3. Deployable predicted-regime CQR on the full 671-basin model.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.cqr import CQRCalibrator
from src.models.lpu_stream import LPUStreamModel
from src.utils import get_device

QUANTILES = [0.05, 0.5, 0.95]
CKPT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "lpu_stream_quantile_best.pt"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "enhanced_robustness.json"


def predict_with_groups(model, loader, device):
    model.eval()
    preds, targets, masks, basin_idx, year_idx = [], [], [], [], []
    with torch.no_grad():
        for dynamic, static, target, mask, bidx, yidx in loader:
            pred = model(dynamic.to(device), static.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
            basin_idx.append(bidx.numpy())
            year_idx.append(yidx.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets).reshape(-1)
    masks = np.concatenate(masks).reshape(-1) > 0
    basin_idx = np.concatenate(basin_idx).reshape(-1)
    year_idx = np.concatenate(year_idx).reshape(-1)
    return preds[masks], targets[masks], basin_idx[masks], year_idx[masks]


def winkler_score(lo, hi, y, alpha=0.1):
    width = hi - lo
    penalty_lower = (2.0 / alpha) * np.maximum(lo - y, 0)
    penalty_upper = (2.0 / alpha) * np.maximum(y - hi, 0)
    return width + penalty_lower + penalty_upper


def metrics(lo, hi, y, alpha=0.1):
    in_interval = (y >= lo) & (y <= hi)
    q33, q67 = np.percentile(y, [33, 67])
    regimes = {
        "low": y <= q33,
        "normal": (y > q33) & (y <= q67),
        "high": y > q67,
    }
    return {
        "picp": float(in_interval.mean()),
        "mpiw": float((hi - lo).mean()),
        "winkler": float(winkler_score(lo, hi, y, alpha).mean()),
        "coverage_low": float(in_interval[regimes["low"]].mean()),
        "coverage_normal": float(in_interval[regimes["normal"]].mean()),
        "coverage_high": float(in_interval[regimes["high"]].mean()),
    }


def fit_q_cal(q_lower, q_upper, y, alpha=0.1):
    cal = CQRCalibrator(alpha=alpha)
    return float(cal.fit(q_lower, q_upper, y))


def cluster_bootstrap(lo, hi, y, basin_idx, alpha=0.1, n_boot=500, seed=20260707):
    """Cluster bootstrap by basin for pooled uncertainty metrics."""
    rng = np.random.default_rng(seed)
    basins = np.unique(basin_idx)
    basin_to_pos = {b: i for i, b in enumerate(basins)}
    n_b = len(basins)

    in_interval = ((y >= lo) & (y <= hi)).astype(np.float64)
    width = (hi - lo).astype(np.float64)
    wink = winkler_score(lo, hi, y, alpha).astype(np.float64)

    count = np.zeros(n_b, dtype=np.float64)
    in_sum = np.zeros(n_b, dtype=np.float64)
    width_sum = np.zeros(n_b, dtype=np.float64)
    wink_sum = np.zeros(n_b, dtype=np.float64)
    for b in basins:
        i = basin_to_pos[b]
        m = basin_idx == b
        count[i] = m.sum()
        in_sum[i] = in_interval[m].sum()
        width_sum[i] = width[m].sum()
        wink_sum[i] = wink[m].sum()

    draws = {"picp": [], "mpiw": [], "winkler": []}
    for _ in range(n_boot):
        sample = rng.integers(0, n_b, size=n_b)
        n = count[sample].sum()
        draws["picp"].append(in_sum[sample].sum() / n)
        draws["mpiw"].append(width_sum[sample].sum() / n)
        draws["winkler"].append(wink_sum[sample].sum() / n)

    out = {}
    for key, vals in draws.items():
        arr = np.asarray(vals)
        out[key] = {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }
    return out


def calibration_window_sensitivity(vp, vy, vyear, tq05, tq95, ty):
    max_year = int(np.max(vyear))
    rows = {}
    for n_years in [1, 2, 3, 4, 5]:
        m = vyear >= max_year - n_years + 1
        q_cal = fit_q_cal(vp[m, 0], vp[m, 2], vy[m], alpha=0.1)
        lo, hi = tq05 - q_cal, tq95 + q_cal
        row = metrics(lo, hi, ty, alpha=0.1)
        row["q_cal"] = q_cal
        row["n_cal"] = int(m.sum())
        rows[str(n_years)] = row
    return rows


def predicted_regime_cqr(vp, vy, tq05, tq50, tq95, ty):
    vq50 = vp[:, 1]
    p33, p67 = np.percentile(vq50, [33, 67])
    val_masks = {
        "low": vq50 <= p33,
        "normal": (vq50 > p33) & (vq50 <= p67),
        "high": vq50 > p67,
    }
    test_masks = {
        "low": tq50 <= p33,
        "normal": (tq50 > p33) & (tq50 <= p67),
        "high": tq50 > p67,
    }
    q_by_regime = {}
    qc = np.zeros_like(ty, dtype=np.float32)
    for key in ["low", "normal", "high"]:
        q_by_regime[key] = fit_q_cal(vp[val_masks[key], 0], vp[val_masks[key], 2], vy[val_masks[key]], alpha=0.1)
        qc[test_masks[key]] = q_by_regime[key]

    lo, hi = tq05 - qc, tq95 + qc
    observed_q33, observed_q67 = np.percentile(ty, [33, 67])
    observed_masks = {
        "low": ty <= observed_q33,
        "normal": (ty > observed_q33) & (ty <= observed_q67),
        "high": ty > observed_q67,
    }
    in_interval = (ty >= lo) & (ty <= hi)
    return {
        "thresholds": {"q50_33": float(p33), "q50_67": float(p67)},
        "q_cal": q_by_regime,
        "counts": {k: {"cal": int(val_masks[k].sum()), "test": int(test_masks[k].sum())}
                   for k in val_masks},
        "overall": metrics(lo, hi, ty, alpha=0.1),
        "coverage_by_observed_regime": {
            k: float(in_interval[m].mean()) for k, m in observed_masks.items()
        },
        "coverage_by_predicted_regime": {
            k: float(in_interval[m].mean()) for k, m in test_masks.items()
        },
    }


def main():
    start = time.time()
    device = get_device()
    model = LPUStreamModel(quantiles=QUANTILES).to(device)
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Creating dataloaders...", flush=True)
    _, val_loader, test_loader = create_dataloaders(
        seq_len=365, batch_size=1024, basin_list=get_basin_list())

    print("Predicting validation and test sets...", flush=True)
    vp, vy, _, vyear = predict_with_groups(model, val_loader, device)
    tp, ty, tbasin, _ = predict_with_groups(model, test_loader, device)
    tq05, tq50, tq95 = tp[:, 0], tp[:, 1], tp[:, 2]

    q_full = fit_q_cal(vp[:, 0], vp[:, 2], vy, alpha=0.1)
    full_lo, full_hi = tq05 - q_full, tq95 + q_full
    full_metrics = metrics(full_lo, full_hi, ty, alpha=0.1)
    full_metrics["q_cal"] = q_full
    full_metrics["n_cal"] = int(len(vy))

    print("Computing sensitivity and bootstrap checks...", flush=True)
    sensitivity = calibration_window_sensitivity(vp, vy, vyear, tq05, tq95, ty)
    bootstrap = cluster_bootstrap(full_lo, full_hi, ty, tbasin, alpha=0.1, n_boot=500)
    pred_regime = predicted_regime_cqr(vp, vy, tq05, tq50, tq95, ty)

    out = {
        "experiment": "enhanced_robustness_full_671",
        "n_basins": int(len(np.unique(tbasin))),
        "full_cqr": full_metrics,
        "calibration_window_sensitivity": sensitivity,
        "cluster_bootstrap_95ci": bootstrap,
        "predicted_regime_cqr": pred_regime,
        "runtime_s": float(time.time() - start),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"Saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
