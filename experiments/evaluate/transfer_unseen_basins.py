"""Ungauged-basin transfer evaluation of the scarcity models.

Restored from revision_work/tier2_ungauged.py (removed as author-side
scratch): loads the 12 archived scarcity checkpoints and infers on the 621
CAMELS basins that the 50-basin pipeline never touched. Training basins,
normalization statistics, and static-attribute standardization all come from
the seen basins, so this is a deployable transfer setting. Three calibration
rules are compared on the unseen test period: raw quantiles, the transferred
radius (fitted on the seen basins' 1995-2000 window), and a local radius
refitted on the unseen basins' own 1995-2000 window.

Produces the ungauged_scarcity tables and prediction archives consumed by
experiments/evaluate/unseen_basins.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.corrections.common import ALPHA, LENGTHS, SEEDS, conformal_radius, metrics  # noqa: E402
from src.data.data_preprocessing import TEST_END, TEST_START, VAL_END, VAL_START  # noqa: E402
from src.data.dataset import CamelsDataset, get_basin_list  # noqa: E402
from src.models.lpu_stream import LPUStreamModel  # noqa: E402
from src.utils import get_device, sha256_file  # noqa: E402

CKPT_DIR = ROOT / "results" / "checkpoints" / "corrected_scarcity"
META_DIR = ROOT / "results" / "tables" / "corrected_scarcity"
OUT_TABLE = ROOT / "results" / "tables" / "ungauged_scarcity"
OUT_PRED = ROOT / "results" / "predictions" / "ungauged_scarcity"
SEQ_LEN = 180
BATCH = 512
BOOT_SEED = 20260711


def context_start(target_start: str, seq_len: int) -> str:
    date = datetime.strptime(target_start, "%Y-%m-%d") - timedelta(days=seq_len)
    return date.strftime("%Y-%m-%d")


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


def per_basin_nse(y_log, q50_log, basin):
    observed = np.maximum(np.expm1(y_log), 0.0)
    predicted = np.maximum(np.expm1(q50_log), 0.0)
    values = {}
    for b in np.unique(basin):
        sel = basin == b
        denom = np.square(observed[sel] - observed[sel].mean()).sum()
        if denom > 0:
            values[str(b)] = float(1.0 - np.square(observed[sel] - predicted[sel]).sum() / denom)
    return values


def basin_rows(lower, upper, y, basin):
    """Per-basin (n, covered, width sum, Winkler sum) for cluster bootstrap."""
    penalty = 2.0 / ALPHA
    width = upper - lower
    winkler = width + np.maximum(lower - y, 0.0) * penalty + np.maximum(y - upper, 0.0) * penalty
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)
    rows = []
    for b in np.unique(basin):
        sel = basin == b
        rows.append((
            str(b), int(sel.sum()), float(covered[sel].sum()),
            float(width[sel].sum()), float(winkler[sel].sum()),
        ))
    return rows


def median_ci(values, n_boot=2000):
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    boots = np.median(rng.choice(array, size=(n_boot, array.size), replace=True), axis=1)
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def evaluate_run(name, ckpt, loaders, device):
    meta = json.loads((META_DIR / f"{name}.json").read_text())
    stats = ckpt["normalization_stats"]
    model = LPUStreamModel(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    out = {}
    for period in ("val", "test"):
        pred, target, mask = predict(model, loaders[period], device)
        pred_log = pred.astype(np.float64) * stats["target_std"] + stats["target_mean"]
        target_log = target.astype(np.float64) * stats["target_std"] + stats["target_mean"]
        out[period] = (pred_log, target_log, mask)

    val_pred, val_y, val_mask = out["val"]
    obs_v = val_mask > 0
    r_local = conformal_radius(np.maximum(
        val_pred[obs_v, 0] - val_y[obs_v], val_y[obs_v] - val_pred[obs_v, 2]))
    r_transfer = float(meta["q_cal_log"])

    pred_log, target_log, mask = out["test"]
    obs = mask > 0
    q05, q50, q95 = (pred_log[obs, i] for i in range(3))
    y = target_log[obs]
    basin = loaders["test"].dataset.sample_metadata()[0][obs]

    schemes = {
        "raw": metrics(q05, q95, y),
        "transfer": metrics(q05 - r_transfer, q95 + r_transfer, y),
        "local": metrics(q05 - r_local, q95 + r_local, y),
    }
    nse_map = per_basin_nse(y, q50, basin)
    nse_values = list(nse_map.values())
    return {
        "r_transfer": r_transfer,
        "r_local": float(r_local),
        "schemes": schemes,
        "nse": {
            "n_basins": len(nse_values),
            "median": float(np.median(nse_values)),
            "mean": float(np.mean(nse_values)),
            "q25": float(np.quantile(nse_values, 0.25)),
            "q75": float(np.quantile(nse_values, 0.75)),
            "fraction_above_zero": float(np.mean(np.asarray(nse_values) > 0)),
            "median_ci95": median_ci(nse_values),
            "by_basin": nse_map,
        },
        "pooled_log_nse": float(1.0 - np.square(y - q50).sum() / np.square(y - y.mean()).sum()),
        "basin_rows": {
            "raw": basin_rows(q05, q95, y, basin),
            "transfer": basin_rows(q05 - r_transfer, q95 + r_transfer, y, basin),
            "local": basin_rows(q05 - r_local, q95 + r_local, y, basin),
        },
        "_arrays": out,
        "_basin_all": loaders["test"].dataset.sample_metadata()[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = get_device()
    all_basins = get_basin_list()
    ref_meta = json.loads(
        (META_DIR / "scarcity_15yr_basinseed42_modelseed42.json").read_text())
    selected = ref_meta["selected_basins"]
    unseen = sorted(set(all_basins) - set(selected))
    print(f"basins: {len(all_basins)} total, {len(selected)} seen, {len(unseen)} unseen", flush=True)

    lengths = ["1yr", "15yr"] if args.smoke else list(LENGTHS)
    seeds = [42] if args.smoke else list(SEEDS)
    if args.smoke:
        unseen = unseen[:8]
        out_table = Path(__file__).resolve().parent / "_smoke_ungauged"
        out_table.mkdir(exist_ok=True)
        out_pred = out_table
    else:
        out_table, out_pred = OUT_TABLE, OUT_PRED
        out_table.mkdir(parents=True, exist_ok=True)
        out_pred.mkdir(parents=True, exist_ok=True)

    periods = {
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END),
    }

    for length in lengths:
        ref_name = f"scarcity_{length}_basinseed42_modelseed42"
        ckpt_ref = torch.load(CKPT_DIR / f"{ref_name}.pt", map_location="cpu", weights_only=False)
        stats = ckpt_ref["normalization_stats"]
        loaders = {}
        for period, (t0, t1) in periods.items():
            dataset = CamelsDataset(
                unseen, context_start(t0, SEQ_LEN), t1, seq_len=SEQ_LEN,
                normalization_stats=stats, preview=True,
                target_start_date=t0, target_end_date=t1,
            )
            miss = dataset.dynamic_missing_fraction().max()
            assert miss <= 0.01, (length, period, float(miss))
            loaders[period] = torch.utils.data.DataLoader(
                dataset, batch_size=BATCH, shuffle=False, num_workers=0)
        del ckpt_ref

        for seed in seeds:
            name = f"scarcity_{length}_basinseed42_modelseed{seed}"
            result_path = out_table / f"scarcity_{length}_unseen_modelseed{seed}.json"
            if result_path.exists() and not args.force and not args.smoke:
                try:
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    artifact = ROOT / existing["predictions"]
                    if sha256_file(artifact) == existing["predictions_sha256"]:
                        print(f"skip existing {result_path.name}", flush=True)
                        continue
                except (KeyError, OSError, ValueError):
                    pass
            started = time.time()
            ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
            result = evaluate_run(name, ckpt, loaders, device)
            arrays, basin_all = result.pop("_arrays"), result.pop("_basin_all")
            elapsed = time.time() - started

            if not args.smoke:
                pred_npz = out_pred / f"scarcity_{length}_unseen_modelseed{seed}.npz"
                temp = pred_npz.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temp,
                    val_prediction_log=arrays["val"][0].astype(np.float32),
                    val_target_log=arrays["val"][1].astype(np.float32),
                    val_mask=arrays["val"][2],
                    val_basin_ids=loaders["val"].dataset.sample_metadata()[0],
                    val_target_dates=loaders["val"].dataset.sample_metadata()[1],
                    test_prediction_log=arrays["test"][0].astype(np.float32),
                    test_target_log=arrays["test"][1].astype(np.float32),
                    test_mask=arrays["test"][2],
                    test_basin_ids=basin_all,
                    test_target_dates=loaders["test"].dataset.sample_metadata()[1],
                )
                temp.replace(pred_npz)
                result["predictions"] = pred_npz.relative_to(ROOT).as_posix()
                result["predictions_sha256"] = sha256_file(pred_npz)
                result["checkpoint_sha256"] = sha256_file(CKPT_DIR / f"{name}.pt")

            result.update({
                "protocol_version": "ungauged_transfer_v1",
                "years": length,
                "model_seed": seed,
                "n_unseen_basins": len(unseen),
                "calibration_target_period": [VAL_START, VAL_END],
                "test_target_period": [TEST_START, TEST_END],
                "seq_len": SEQ_LEN,
                "inference_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            })
            temp_json = result_path.with_suffix(".tmp.json")
            temp_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            temp_json.replace(result_path)
            nse = result["nse"]["median"]
            print(
                f"{name}: {elapsed:.0f}s NSEmed={nse:.3f} "
                f"raw={result['schemes']['raw']['picp']:.4f} "
                f"transfer={result['schemes']['transfer']['picp']:.4f} "
                f"local={result['schemes']['local']['picp']:.4f} "
                f"r_local={result['r_local']:.4f}",
                flush=True,
            )
    print("done", flush=True)


if __name__ == "__main__":
    main()
