"""Shared loaders, validation, and regime rules for the corrected scarcity runs.

Every analysis of the 12 primary fits (4 training lengths x 3 model seeds)
goes through this module: artifact loading, protocol and hash validation,
interval metrics, and the within-basin regime definitions, so none of them
can drift between consumers. All analyses are offline: they consume the
archived per-sample predictions of the 12 matched runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.protocol import conformal_radius as _shared_radius, interval_metrics, project_nonempty
from src.utils import repository_path, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "corrected_scarcity"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "corrected_scarcity"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "camels_us"
ALPHA = 0.10
LENGTHS = ("1yr", "3yr", "5yr", "15yr")
YEARS = [1, 3, 5, 15]
SEEDS = (42, 123, 456)
PROTOCOL_VERSION = "scarcity_v2_fixed_end_fixed_memory"


def artifact_path(years: int, seed: int) -> Path:
    return TABLE_DIR / f"scarcity_{years}yr_basinseed42_modelseed{seed}.json"


def load_result(years: int, seed: int) -> dict:
    path = artifact_path(years, seed)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"Unexpected protocol in {path}")
    for key, expected in {
        "years": years,
        "model_seed": seed,
        "basin_seed": 42,
        "seq_len": 180,
        "batch_size": 512,
        "epochs": 30,
        "calibration_target_period": ["1995-10-01", "2000-09-30"],
        "test_target_period": ["2000-10-01", "2008-12-31"],
    }.items():
        if result[key] != expected:
            raise ValueError(f"{path.name}: {key}={result[key]!r}, expected {expected!r}")

    for path_key, hash_key in [
        ("checkpoint", "checkpoint_sha256"),
        ("predictions", "predictions_sha256"),
    ]:
        data_path = repository_path(PROJECT_ROOT, result[path_key])
        if sha256_file(data_path) != result[hash_key]:
            raise ValueError(f"Hash mismatch for {data_path}")
    return result


def load_predictions(result: dict) -> dict[str, np.ndarray]:
    path = repository_path(PROJECT_ROOT, result["predictions"])
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def metric_value(result: dict, metric: str) -> float:
    mapping = {
        "median_nse": result["point_metrics"]["median"],
        "pooled_log_nse": result["point_metrics"]["pooled_log_nse"],
        "raw_picp": result["raw_interval_metrics"]["picp"],
        "cqr_picp": result["calibrated_interval_metrics"]["picp"],
        "raw_mpiw_log": result["raw_interval_metrics"]["mpiw"],
        "cqr_mpiw_log": result["calibrated_interval_metrics"]["mpiw"],
        "raw_winkler_log": result["raw_interval_metrics"]["winkler_score"],
        "cqr_winkler_log": result["calibrated_interval_metrics"]["winkler_score"],
        "raw_high_flow_picp": result["raw_interval_metrics"]["basin_relative_regimes"]["high"]["picp"],
        "cqr_high_flow_picp": result["calibrated_interval_metrics"]["basin_relative_regimes"]["high"]["picp"],
        "q_cal_log": result["q_cal_log"],
    }
    return float(mapping[metric])


def per_basin_regimes(values: np.ndarray, basin_ids: np.ndarray) -> np.ndarray:
    regimes = np.empty(len(values), dtype="U6")
    for basin_id in np.unique(basin_ids):
        selected = basin_ids == basin_id
        q33, q67 = np.quantile(values[selected], [0.33, 0.67])
        regimes[selected & (values <= q33)] = "low"
        regimes[selected & (values > q33) & (values <= q67)] = "normal"
        regimes[selected & (values > q67)] = "high"
    return regimes


def reference_regimes(
    reference_values: np.ndarray,
    reference_basins: np.ndarray,
    values: np.ndarray,
    basin_ids: np.ndarray,
) -> np.ndarray:
    regimes = np.empty(len(values), dtype="U6")
    for basin_id in np.unique(basin_ids):
        reference = reference_values[reference_basins == basin_id]
        if len(reference) == 0:
            raise ValueError(f"No reference values for basin {basin_id}")
        q33, q67 = np.quantile(reference, [0.33, 0.67])
        selected = basin_ids == basin_id
        regimes[selected & (values <= q33)] = "low"
        regimes[selected & (values > q33) & (values <= q67)] = "normal"
        regimes[selected & (values > q67)] = "high"
    return regimes


def basin_interval_metrics(result: dict, alpha: float = ALPHA) -> dict[str, dict[str, float]]:
    data = load_predictions(result)
    valid = data["test_mask"] > 0
    prediction = data["test_prediction_log"][valid]
    target = data["test_target_log"][valid]
    basins = data["test_basin_ids"][valid]
    regimes = per_basin_regimes(target, basins)
    q_cal = float(result["q_cal_log"])
    intervals = {
        "raw": (prediction[:, 0], prediction[:, 2]),
        "cqr": (prediction[:, 0] - q_cal, prediction[:, 2] + q_cal),
    }
    output = {}
    for basin in np.unique(basins):
        selected = basins == basin
        high = selected & (regimes == "high")
        basin_metrics = {}
        for method, (lower, upper) in intervals.items():
            covered = (target >= lower) & (target <= upper)
            width = upper - lower
            winkler = (
                width
                + (2 / alpha) * np.maximum(lower - target, 0)
                + (2 / alpha) * np.maximum(target - upper, 0)
            )
            basin_metrics[f"{method}_picp"] = float(covered[selected].mean())
            basin_metrics[f"{method}_mpiw_log"] = float(width[selected].mean())
            basin_metrics[f"{method}_winkler_log"] = float(winkler[selected].mean())
            basin_metrics[f"{method}_high_flow_picp"] = float(covered[high].mean())
        output[str(basin)] = basin_metrics
    return output


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


def load(year, seed):
    """Calibration/test arrays for one archived corrected scarcity run."""
    p = PROJECT_ROOT / f"results/predictions/corrected_scarcity/scarcity_{year}yr_basinseed42_modelseed{seed}.npz"
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
