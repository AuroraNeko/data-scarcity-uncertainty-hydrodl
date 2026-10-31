"""Shared loaders, interval metrics, and reproduction cross-checks for the corrected scarcity analyses.

All analyses are offline: they consume the archived per-sample predictions of
the 12 matched scarcity runs (4 record lengths x 3 model seeds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.protocol import conformal_radius as _shared_radius, interval_metrics, project_nonempty

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "results" / "predictions" / "corrected_scarcity"
TABLE_DIR = ROOT / "results" / "tables" / "corrected_scarcity"
PROCESSED_DIR = ROOT / "data" / "processed" / "camels_us"
ALPHA = 0.10
LENGTHS = ("1yr", "3yr", "5yr", "15yr")
SEEDS = (42, 123, 456)


def load_run(length: str, seed: int):
    name = f"scarcity_{length}_basinseed42_modelseed{seed}"
    data = np.load(PRED_DIR / f"{name}.npz", allow_pickle=True)
    meta = json.loads((TABLE_DIR / f"{name}.json").read_text())
    return name, data, meta


def observed_split(data):
    mask = data["test_mask"].astype(bool)
    cal_mask = data["val_mask"].astype(bool)
    cal = {
        "q": data["val_prediction_log"][cal_mask].astype(np.float64),
        "y": data["val_target_log"][cal_mask].astype(np.float64),
        "dates": data["val_target_dates"][cal_mask],
        "basin": data["val_basin_ids"][cal_mask],
    }
    test = {
        "q": data["test_prediction_log"][mask].astype(np.float64),
        "y": data["test_target_log"][mask].astype(np.float64),
        "dates": data["test_target_dates"][mask],
        "basin": data["test_basin_ids"][mask],
    }
    return cal, test


def conformal_radius(scores: np.ndarray, alpha: float = ALPHA) -> float:
    return _shared_radius(scores, alpha)


def metrics(lower, upper, y, alpha: float = ALPHA) -> dict:
    lower,upper,_=project_nonempty(lower,upper)
    result=interval_metrics(lower,upper,y,alpha)
    return {key:result[key] for key in ['picp','mpiw','winkler','p_below','p_above']}


def crosscheck(cal, test, meta, tol: float = 5e-4) -> float:
    """Reproduce the archived raw/global-CQR metrics; raise on any mismatch."""
    raw = metrics(test["q"][:, 0], test["q"][:, 2], test["y"])
    ref_raw = meta["raw_interval_metrics"]
    for key, ref in (("picp", "picp"), ("mpiw", "mpiw"), ("winkler", "winkler_score")):
        assert abs(raw[key] - ref_raw[ref]) < tol, (key, raw[key], ref_raw[ref])
    scores = np.maximum(cal["q"][:, 0] - cal["y"], cal["y"] - cal["q"][:, 2])
    radius = conformal_radius(scores)
    assert abs(radius - meta["q_cal_log"]) < tol, (radius, meta["q_cal_log"])
    glob = metrics(test["q"][:, 0] - radius, test["q"][:, 2] + radius, test["y"])
    ref_cal = meta["calibrated_interval_metrics"]
    for key, ref in (("picp", "picp"), ("mpiw", "mpiw"), ("winkler", "winkler_score")):
        assert abs(glob[key] - ref_cal[ref]) < tol, (key, glob[key], ref_cal[ref])
    return radius


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def load(year, seed):
    """Calibration/test arrays for one archived corrected scarcity run."""
    p = ROOT / f"results/predictions/corrected_scarcity/scarcity_{year}yr_basinseed42_modelseed{seed}.npz"
    d = np.load(p)
    cm = d["val_mask"] > 0
    tm = d["test_mask"] > 0
    cal = {k: d[v][cm] for k, v in {"q": "val_prediction_log", "y": "val_target_log",
                                    "b": "val_basin_ids", "date": "val_target_dates"}.items()}
    test = {k: d[v][tm] for k, v in {"q": "test_prediction_log", "y": "test_target_log",
                                     "b": "test_basin_ids", "date": "test_target_dates"}.items()}
    cal["q"] = cal["q"].astype(float)
    cal["y"] = cal["y"].astype(float)
    test["q"] = test["q"].astype(float)
    test["y"] = test["y"].astype(float)
    return cal, test
