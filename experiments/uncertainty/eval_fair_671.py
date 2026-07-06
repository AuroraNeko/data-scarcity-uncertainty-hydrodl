"""
eval_fair_671.py — Fair, same-basin-set uncertainty method comparison on the
FULL 671-basin CAMELS-US test set.

Why this script exists
----------------------
The Deep Ensembles row published in Table 3 was produced on a 300-basin subset
(deep_ensembles_fair_results.json records ``n_basins: 300``), whereas CQR and
MC Dropout use the full 671-basin set. That makes the NSE column of Table 3
non-comparable across rows. This script re-evaluates ALL methods on the
identical full 671-basin test set so coverage / width / Winkler / NSE are
genuinely like-for-like.

All methods: 671 basins, 15-year training period, seq_len = 365, alpha = 0.10.
CQR calibration uses the shared 5-year validation period (1995-2000).

Methods
-------
  MC Dropout         point model (lpu_stream_best.pt), 50 stochastic passes,
                     5th/95th percentiles form the interval (raw, no CQR).
  Deep Ensembles (5) quantile models, prediction = mean across members (raw).
  Deep Ensembles+CQR ensemble mean + CQR calibration on the val set.
  CQR (single)       lpu_stream_quantile_best.pt + CQR calibration.

Usage
-----
    python experiments/uncertainty/eval_fair_671.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.utils import get_device

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
OUT_PATH = PROJECT_ROOT / "results" / "tables" / "fair_comparison_671.json"
QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.1
N_MC = 50
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]

device = get_device()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _quantile_ckpt_path(seed: int) -> Path:
    return CKPT_DIR / ("lpu_stream_quantile_best.pt" if seed == 42
                       else f"ensemble_seed{seed}.pt")


def load_quantile_model(seed: int) -> LPUStreamModel:
    ckpt = torch.load(_quantile_ckpt_path(seed), weights_only=False)
    cfg = ckpt.get("config", {})
    model = LPUStreamModel(
        n_dynamic=cfg.get("n_dynamic", 15),
        n_static=cfg.get("n_static", 13),
        hidden_size=cfg.get("hidden_size", 128),
        embed_dim=cfg.get("embed_dim", 32),
        dropout=cfg.get("dropout", 0.3),
        quantiles=QUANTILES,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_point_model() -> LPUStreamModel:
    ckpt = torch.load(CKPT_DIR / "lpu_stream_best.pt", weights_only=False)
    cfg = ckpt.get("config", {})
    model = LPUStreamModel(
        n_dynamic=cfg.get("n_dynamic", 15),
        n_static=cfg.get("n_static", 13),
        hidden_size=cfg.get("hidden_size", 128),
        embed_dim=cfg.get("embed_dim", 32),
        dropout=cfg.get("dropout", 0.3),
        quantiles=None,  # point prediction
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_quantile(model, loader):
    preds, targets, masks = [], [], []
    for dynamic, static, target, mask, _, _ in loader:
        pred = model(dynamic.to(device), static.to(device))
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


@torch.no_grad()
def mc_dropout_predict(model, loader, n_mc: int):
    """n_mc stochastic forward passes with dropout enabled."""
    model.train()  # keep dropout active at inference
    mc_preds, target, mask = [], None, None
    for i in range(n_mc):
        pass_preds = []
        for dynamic, static, t, m, _, _ in loader:
            pred = model(dynamic.to(device), static.to(device))
            pass_preds.append(pred.cpu().numpy())
            if i == 0:
                if target is None:
                    target, mask = [], []
                target.append(t.numpy())
                mask.append(m.numpy())
        mc_preds.append(np.concatenate(pass_preds).squeeze(-1))
        if (i + 1) % 10 == 0:
            print(f"   MC pass {i + 1}/{n_mc}", flush=True)
    model.eval()
    stacked = np.stack(mc_preds, axis=0)  # (n_mc, n_samples)
    target = np.concatenate(target)
    mask = np.concatenate(mask)
    q05 = np.percentile(stacked, 5, axis=0)
    q50 = np.percentile(stacked, 50, axis=0)
    q95 = np.percentile(stacked, 95, axis=0)
    return q05, q50, q95, target, mask


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def quantile_metrics(test_pred, val_pred, test_target, test_mask,
                     val_target, val_mask, calibrate: bool) -> dict:
    """test_pred/val_pred: (n_samples, 3) quantile arrays."""
    tv = test_mask.flatten() > 0
    vv = val_mask.flatten() > 0
    test_y = test_target.flatten()[tv]
    val_y = val_target.flatten()[vv]

    med = test_pred[tv, 1]
    ss_res = np.sum((test_y - med) ** 2)
    ss_tot = np.sum((test_y - test_y.mean()) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    q05, q95 = test_pred[tv, 0], test_pred[tv, 2]
    raw = compute_uncertainty_metrics(q05, q95, test_y, alpha=ALPHA)
    out = {"nse": nse, "raw": raw}
    if calibrate:
        cal = CQRCalibrator(ALPHA)
        q_cal = cal.fit(val_pred[vv, 0], val_pred[vv, 2], val_y)
        lo, hi = cal.calibrate(q05, q95)
        out["cal"] = compute_uncertainty_metrics(lo, hi, test_y, alpha=ALPHA)
        out["q_cal"] = float(q_cal)
    return out


def mc_metrics(q05, q50, q95, target, mask) -> dict:
    valid = mask.flatten() > 0
    y = target.flatten()[valid]
    ss_res = np.sum((y - q50[valid]) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    raw = compute_uncertainty_metrics(q05[valid], q95[valid], y, alpha=ALPHA)
    return {"nse": nse, "raw": raw}


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def _row(name, nse, m):
    return (f"{name:<24}{nse:>8.3f}{m['picp']:>9.3f}{m['mpiw']:>9.3f}"
            f"{m['winkler_score']:>10.3f}")


def print_table(results):
    print("\n" + "=" * 72)
    print(f"FAIR comparison — ALL methods on FULL {results['n_basins']}-basin test set")
    print("=" * 72)
    hdr = f"{'Method':<24}{'NSE':>8}{'PICP':>9}{'MPIW':>9}{'Winkler':>10}"
    print(hdr)
    print("-" * 72)
    mc = results["mc_dropout"]
    print(_row("MC Dropout", mc["nse"], mc["raw"]))
    ens = results["deep_ensembles"]
    print(_row("Deep Ensembles (5)", ens["nse"], ens["raw"]))
    print(_row("Deep Ensembles+CQR", ens["nse"], ens["cal"]))
    cqr = results["cqr_single"]
    print(_row("CQR (single)", cqr["nse"], cqr["cal"]))
    print("=" * 72)
    print(f"Ensemble q_cal = {ens.get('q_cal', 0):.4f}   "
          f"CQR q_cal = {cqr.get('q_cal', 0):.4f}")
    if ens["nse"] > 0.80:
        print(">> Ensemble NSE recovered to ~full-data level on 671 basins: "
              "the old '300-basin' 0.736 was a subset artifact.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    basins = get_basin_list()
    print(f"Fair comparison on FULL basin set: {len(basins)} basins")
    _, val_loader, test_loader = create_dataloaders(
        seq_len=365, batch_size=1024, basin_list=basins)

    # ===== Deep Ensembles (5 quantile members) =====
    print("\n[1/4] Loading + predicting 5 ensemble members...")
    val_members, test_members = [], []
    test_target = test_mask = val_target = val_mask = None
    for seed in ENSEMBLE_SEEDS:
        model = load_quantile_model(seed)
        vp, vt, vm = predict_quantile(model, val_loader)
        tp, tt, tm = predict_quantile(model, test_loader)
        val_members.append(vp)
        test_members.append(tp)
        if test_target is None:
            test_target, test_mask = tt, tm
            val_target, val_mask = vt, vm
        print(f"   seed {seed}: done")

    val_ens = np.mean(np.stack(val_members, 0), 0)
    test_ens = np.mean(np.stack(test_members, 0), 0)
    ens = quantile_metrics(test_ens, val_ens, test_target, test_mask,
                           val_target, val_mask, calibrate=True)

    # ===== CQR single (member 0) =====
    print("\n[2/4] CQR single model (seed 42)...")
    single = load_quantile_model(42)
    vp_s, _, _ = predict_quantile(single, val_loader)
    tp_s, _, _ = predict_quantile(single, test_loader)
    cqr = quantile_metrics(tp_s, vp_s, test_target, test_mask,
                           val_target, val_mask, calibrate=True)

    # ===== MC Dropout =====
    print(f"\n[3/4] MC Dropout ({N_MC} passes)...")
    mc_model = load_point_model()
    q05, q50, q95, mc_target, mc_mask = mc_dropout_predict(
        mc_model, test_loader, N_MC)
    mc = mc_metrics(q05, q50, q95, mc_target, mc_mask)

    # ===== Assemble + save =====
    print("\n[4/4] Saving results...")
    results = {
        "experiment": "fair_comparison_671",
        "n_basins": len(basins),
        "methods": {
            "mc_dropout": {"n_mc": N_MC, **mc},
            "deep_ensembles": {"n_members": len(ENSEMBLE_SEEDS),
                               "seeds": ENSEMBLE_SEEDS, **ens},
            "cqr_single": {"seed": 42, **cqr},
        },
        # convenience flat copy
        "mc_dropout": mc,
        "deep_ensembles": ens,
        "cqr_single": cqr,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print_table(results)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
