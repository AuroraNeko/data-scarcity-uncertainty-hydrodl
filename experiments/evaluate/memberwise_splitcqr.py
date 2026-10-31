"""Per-member split-CQR then interval averaging (legacy ``encqr`` artifact IDs).

The archived 671-basin comparison stores ensemble predictions only aggregated
over the five quantile checkpoints. This script reloads those checkpoints,
regenerates per-member predictions on the archived calibration and test
windows, validates them sample-by-sample against the archived single_qr and
ensemble arrays, and fits one split-conformal radius per member fitted
on that member's calibration residuals, with the final interval the average of
the five calibrated member intervals. This is a static ensemble calibration
variant; it differs from the leave-out, sequential EnCQR algorithm of
Jensen et al.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CamelsDataset, PROCESSED_DIR  # noqa: E402
from src.models.lpu_stream import LPUStreamModel  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.corrections.common import ALPHA, conformal_radius, metrics  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "corrected_full_671"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "corrected_full_671"
MEMBERS = [
    ("seed42", "lpu_stream_quantile_best.pt"),
    ("seed123", "ensemble_seed123.pt"),
    ("seed456", "ensemble_seed456.pt"),
    ("seed789", "ensemble_seed789.pt"),
    ("seed999", "ensemble_seed999.pt"),
]
SEQ_LEN = 365
BATCH = 1024
QUANTILES = [0.05, 0.50, 0.95]
PERIODS = {
    "calibration": ("1995-10-01", "2000-09-30"),
    "test": ("2000-10-01", "2008-12-31"),
}


def context_start(target_start: str, seq_len: int) -> str:
    date = datetime.strptime(target_start, "%Y-%m-%d") - timedelta(days=seq_len)
    return date.strftime("%Y-%m-%d")


def recover_target_affine() -> tuple[float, float]:
    """Target standardization from the canonical stats file used by the 671 run.

    The stored *_norm CSV columns were verified to follow the same affine
    (mu=0.597676, sigma=0.643817), so the legacy dataset path and this file
    describe one consistent standardization.
    """
    stats = json.loads(
        (PROJECT_ROOT / "data" / "metadata" / "normalization_stats.json").read_text())
    return float(stats["target_mean"]), float(stats["target_std"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    for dynamic, static, target, mask, _, _ in loader:
        out = model(dynamic.to(device), static.to(device))
        preds.append(out.cpu().numpy())
        targets.append(target.numpy().reshape(-1))
        masks.append(mask.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def basin_rows(lower, upper, y, basin, scheme):
    penalty = 2.0 / ALPHA
    width = upper - lower
    winkler = width + np.maximum(lower - y, 0.0) * penalty + np.maximum(y - upper, 0.0) * penalty
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)
    rows = []
    for b in np.unique(basin):
        sel = basin == b
        rows.append((scheme, str(b), int(sel.sum()), float(covered[sel].sum()),
                     float(width[sel].sum()), float(winkler[sel].sum())))
    return rows


def paired_cluster_diff(rows_a, rows_b, n_boot=1000, seed=20260711):
    """Paired whole-basin bootstrap of pooled metric differences (a minus b)."""
    a = {r[1]: r[2:] for r in rows_a}
    b = {r[1]: r[2:] for r in rows_b}
    basins = sorted(a)
    arr_a = np.asarray([a[x] for x in basins], dtype=np.float64)
    arr_b = np.asarray([b[x] for x in basins], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(basins)
    samples = np.empty((n_boot, 3))
    for i in range(n_boot):
        pick = rng.integers(0, n, size=n)
        sa, sb = arr_a[pick].sum(axis=0), arr_b[pick].sum(axis=0)
        samples[i] = [
            sa[1] / sa[0] - sb[1] / sb[0],
            sa[2] / sa[0] - sb[2] / sb[0],
            sa[3] / sa[0] - sb[3] / sb[0],
        ]
    point = arr_a.sum(axis=0) / [arr_a.sum(0)[0]] - arr_b.sum(axis=0) / arr_b.sum(0)[0]
    return {
        "picp_diff": float(point[1]),
        "picp_diff_ci95": [float(np.quantile(samples[:, 0], 0.025)),
                           float(np.quantile(samples[:, 0], 0.975))],
        "mpiw_diff": float(point[2]),
        "mpiw_diff_ci95": [float(np.quantile(samples[:, 1], 0.025)),
                           float(np.quantile(samples[:, 1], 0.975))],
        "winkler_diff": float(point[3]),
        "winkler_diff_ci95": [float(np.quantile(samples[:, 2], 0.025)),
                              float(np.quantile(samples[:, 2], 0.975))],
    }


def main():
    device = get_device()
    mu, sigma = recover_target_affine()
    print(f"target affine recovered: mu={mu:.6f} sigma={sigma:.6f}", flush=True)

    index = np.load(PRED_DIR / "evaluation_index.npz", allow_pickle=True)
    basin_ids = index["basin_ids"].tolist()
    archive = {p: np.load(PRED_DIR / f"{p}_predictions.npz", allow_pickle=True)
               for p in PERIODS}

    datasets, loaders = {}, {}
    for period, (t0, t1) in PERIODS.items():
        dataset = CamelsDataset(
            basin_ids, context_start(t0, SEQ_LEN), t1, seq_len=SEQ_LEN,
            normalization_stats=None, preview=True,
            target_start_date=t0, target_end_date=t1,
        )
        b, d = dataset.sample_metadata()
        assert np.array_equal(d, index[f"{period}_dates"]), f"{period} date misalignment"
        blocks = b.reshape(len(basin_ids), -1)
        assert (blocks[:, 0] == np.asarray(basin_ids)).all() and (
            blocks == blocks[:, :1]).all(), f"{period} basin misalignment"
        loaders[period] = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH, shuffle=False, num_workers=0)
        datasets[period] = dataset
        del b, d

    member_preds = {}
    for tag, ckpt_name in MEMBERS:
        started = time.time()
        ckpt = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
        config = dict(ckpt["config"])
        model = LPUStreamModel(
            n_dynamic=config.get("n_dynamic", 15),
            n_static=config.get("n_static", 13),
            hidden_size=config.get("hidden_size", 128),
            embed_dim=config.get("embed_dim", 32),
            dropout=config.get("dropout", 0.3),
            quantiles=QUANTILES,
        ).to(device)
        # train_loss checkpoints carry their selected best_state separately
        # from the final model_state_dict, and these models were trained
        # without the monotonic parameterization.
        state = ckpt.get("best_state") if ckpt.get("procedure") == "train_loss" else None
        model.load_state_dict(state or ckpt["model_state_dict"])
        member_preds[tag] = {}
        for period in PERIODS:
            pred, target, mask = predict(model, loaders[period], device)
            # The archived npz stores raw normalized-unit model outputs;
            # de-standardization happens at metric time, so keep both forms.
            member_preds[tag][period] = pred.astype(np.float64)
            if tag == "seed42":
                ref = archive[period]["single_qr"].astype(np.float64)
                diff = np.abs(pred.astype(np.float64) - ref).max()
                assert diff < 5e-3, f"seed42 {period} mismatch: max {diff}"
                print(f"  crosscheck seed42 vs archived single_qr [{period}]: "
                      f"max|diff|={diff:.2e}", flush=True)
        print(f"{tag}: {time.time()-started:.0f}s", flush=True)

    mean_norm = {p: np.mean([member_preds[t][p] for t, _ in MEMBERS], axis=0)
                 for p in PERIODS}
    diff = np.abs(mean_norm["test"] - archive["test"]["ensemble"].astype(np.float64)).max()
    assert diff < 5e-3, f"ensemble mean mismatch: max {diff}"
    print(f"crosscheck member-mean vs archived ensemble [test]: max|diff|={diff:.2e}", flush=True)

    cal_mask = archive["calibration"]["mask"] > 0
    test_mask = archive["test"]["mask"] > 0
    y_cal = archive["calibration"]["target"].astype(np.float64)[cal_mask] * sigma + mu
    y_test = archive["test"]["target"].astype(np.float64)[test_mask] * sigma + mu

    def to_log(period_norm_array, selection):
        return period_norm_array[selection] * sigma + mu

    radii = {}
    for tag, _ in MEMBERS:
        q = to_log(member_preds[tag]["calibration"], cal_mask)
        radii[tag] = conformal_radius(np.maximum(q[:, 0] - y_cal, y_cal - q[:, 2]))
    q_mean_cal = to_log(mean_norm["calibration"], cal_mask)
    r_ensemble = conformal_radius(np.maximum(
        q_mean_cal[:, 0] - y_cal, y_cal - q_mean_cal[:, 2]))
    method = json.loads((TABLE_DIR / "method_comparison.json").read_text())
    assert abs(radii["seed42"] - method["q_cal"]["single_qr"]) < 5e-4
    assert abs(r_ensemble - method["q_cal"]["ensemble"]) < 5e-4
    print(f"radii: per-member {['%.4f' % radii[t] for t, _ in MEMBERS]}, "
          f"ensemble {r_ensemble:.4f} (archived matches)", flush=True)

    q_mean = to_log(mean_norm["test"], test_mask)
    mean_radius = float(np.mean([radii[t] for t, _ in MEMBERS]))
    basin_array = datasets["test"].sample_metadata()[0][test_mask]

    single = to_log(member_preds["seed42"]["test"], test_mask)
    schemes = {
        "single_qr_cqr": {
            "lower": single[:, 0] - radii["seed42"],
            "upper": single[:, 2] + radii["seed42"],
        },
        "ensemble_raw": {"lower": q_mean[:, 0], "upper": q_mean[:, 2]},
        "ensemble_cqr": {
            "lower": q_mean[:, 0] - r_ensemble, "upper": q_mean[:, 2] + r_ensemble,
        },
        "encqr": {
            "lower": q_mean[:, 0] - mean_radius, "upper": q_mean[:, 2] + mean_radius,
        },
    }
    results, all_rows = {}, {}
    for scheme, bounds in schemes.items():
        m = metrics(bounds["lower"], bounds["upper"], y_test)
        results[scheme] = m
        all_rows[scheme] = basin_rows(bounds["lower"], bounds["upper"], y_test,
                                      basin_array, scheme)
        print(f"{scheme}: picp={m['picp']:.4f} mpiw={m['mpiw']:.4f} "
              f"winkler={m['winkler']:.4f}", flush=True)

    ref_check = [
        ("single_qr_cqr", "single_qr_cqr"),
        ("ensemble_cqr", "ensemble_cqr"),
    ]
    for ours, theirs in ref_check:
        ref = method["metrics"][theirs]
        assert abs(results[ours]["picp"] - ref["picp"]) < 5e-4, (ours, "picp")
        assert abs(results[ours]["mpiw"] - ref["mpiw"]) < 5e-4, (ours, "mpiw")
        assert abs(results[ours]["winkler"] - ref["winkler_score"]) < 5e-4, (ours, "winkler")
    print("crosscheck against archived method_comparison metrics passed", flush=True)

    diffs = {
        "encqr_minus_single": paired_cluster_diff(all_rows["encqr"], all_rows["single_qr_cqr"]),
        "encqr_minus_ensemble": paired_cluster_diff(all_rows["encqr"], all_rows["ensemble_cqr"]),
    }

    out = {
        "protocol_version": "memberwise_split_cqr_interval_average_v2",
        "legacy_artifact_id": "encqr",
        "method_identity": "memberwise_split_cqr_interval_average",
        "description": "Per-member split-conformal radii averaged; distinct from EnCQR, "
                       "regenerated from the five archived quantile checkpoints",
        "members": [t for t, _ in MEMBERS],
        "member_radii": radii,
        "ensemble_radius_on_aggregated_scores": float(r_ensemble),
        "mean_member_radius": mean_radius,
        "target_affine": {"mu": mu, "sigma": sigma},
        "periods": {p: list(PERIODS[p]) for p in PERIODS},
        "metrics": results,
        "paired_cluster_bootstrap": diffs,
        "n_test_observed": int(test_mask.sum()),
        "n_calibration_observed": int(cal_mask.sum()),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = TABLE_DIR / "memberwise_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", out_path, flush=True)

    per_member_path = PRED_DIR / "per_member_predictions.npz"
    temp = per_member_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        calibration_dates=index["calibration_dates"],
        test_dates=index["test_dates"],
        basin_ids=index["basin_ids"],
        calibration_mask=archive["calibration"]["mask"],
        test_mask=archive["test"]["mask"],
        target_calibration=archive["calibration"]["target"],
        target_test=archive["test"]["target"],
        **{f"{tag}_{period}": (member_preds[tag][period] * sigma + mu).astype(np.float32)
           for tag, _ in MEMBERS for period in PERIODS},
    )
    temp.replace(per_member_path)
    print("wrote", per_member_path, flush=True)


if __name__ == "__main__":
    main()
