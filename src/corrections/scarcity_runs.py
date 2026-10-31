"""Shared loaders, validation, and regime rules for the corrected scarcity runs.

Every analysis of the 12 primary fits (4 training lengths x 3 model seeds)
goes through this module so that artifact validation, hash checking, and the
within-basin regime definitions cannot drift between consumers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import repository_path, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YEARS = [1, 3, 5, 15]
SEEDS = [42, 123, 456]
ALPHA = 0.10
PROTOCOL_VERSION = "scarcity_v2_fixed_end_fixed_memory"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "corrected_scarcity"


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
