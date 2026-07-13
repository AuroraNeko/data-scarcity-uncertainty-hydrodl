"""Analyze high-flow and basin-wise calibration robustness without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.analyze_corrected_scarcity import (  # noqa: E402
    ALPHA,
    SEEDS,
    YEARS,
    load_predictions,
    load_result,
    per_basin_regimes,
    reference_regimes,
)
from src.losses.cqr import finite_sample_quantile  # noqa: E402


TOP_FLOW_FRACTIONS = [1.0, 0.33, 0.20, 0.10, 0.05, 0.02]
METHODS = ["raw", "global_cqr", "predicted_regime_cqr"]
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "corrected_scarcity"
RUN_CSV = TABLE_DIR / "conditional_robustness_by_run.csv"
BASIN_CSV = TABLE_DIR / "basin_calibration_distribution.csv"
SUMMARY_JSON = TABLE_DIR / "conditional_robustness.json"


def interval_arrays(result: dict) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    data = load_predictions(result)
    val_valid = data["val_mask"] > 0
    test_valid = data["test_mask"] > 0
    val_prediction = data["val_prediction_log"][val_valid]
    val_target = data["val_target_log"][val_valid]
    val_basins = data["val_basin_ids"][val_valid]
    test_prediction = data["test_prediction_log"][test_valid]
    test_target = data["test_target_log"][test_valid]
    test_basins = data["test_basin_ids"][test_valid]

    scores = np.maximum(
        val_prediction[:, 0] - val_target,
        val_target - val_prediction[:, 2],
    )
    global_q = finite_sample_quantile(scores, ALPHA)
    if not np.isclose(global_q, result["q_cal_log"], atol=1e-10):
        raise ValueError("Stored and recomputed global CQR adjustments differ")

    predicted_val_regimes = per_basin_regimes(val_prediction[:, 1], val_basins)
    predicted_test_regimes = reference_regimes(
        val_prediction[:, 1],
        val_basins,
        test_prediction[:, 1],
        test_basins,
    )
    sample_adjustment = np.zeros(len(test_target), dtype=np.float64)
    for regime in ["low", "normal", "high"]:
        adjustment = finite_sample_quantile(
            scores[predicted_val_regimes == regime], ALPHA
        )
        sample_adjustment[predicted_test_regimes == regime] = adjustment

    raw_lower = test_prediction[:, 0]
    raw_upper = test_prediction[:, 2]
    intervals = {
        "raw": (raw_lower, raw_upper),
        "global_cqr": (raw_lower - global_q, raw_upper + global_q),
        "predicted_regime_cqr": (
            raw_lower - sample_adjustment,
            raw_upper + sample_adjustment,
        ),
    }
    return test_target, test_basins.astype(str), intervals


def top_flow_mask(
    target: np.ndarray,
    basins: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Select an exact upper share within each basin using stable ranks."""
    selected = np.zeros(len(target), dtype=bool)
    for basin in np.unique(basins):
        indices = np.flatnonzero(basins == basin)
        count = len(indices) if fraction >= 1 else max(1, int(np.ceil(len(indices) * fraction)))
        order = np.argsort(target[indices], kind="mergesort")
        selected[indices[order[-count:]]] = True
    return selected


def run_rows(years: int, seed: int, result: dict) -> list[dict]:
    target, basins, intervals = interval_arrays(result)
    rows = []
    for fraction in TOP_FLOW_FRACTIONS:
        threshold_mask = top_flow_mask(target, basins, fraction)
        for method, (lower, upper) in intervals.items():
            covered = (target >= lower) & (target <= upper)
            width = upper - lower
            winkler = (
                width
                + (2 / ALPHA) * np.maximum(lower - target, 0)
                + (2 / ALPHA) * np.maximum(target - upper, 0)
            )
            for basin in np.unique(basins):
                local = threshold_mask & (basins == basin)
                rows.append({
                    "years": years,
                    "model_seed": seed,
                    "method": method,
                    "top_flow_fraction": fraction,
                    "basin_id": basin,
                    "n": int(local.sum()),
                    "picp": float(covered[local].mean()),
                    "mpiw_log": float(width[local].mean()),
                    "winkler_log": float(winkler[local].mean()),
                })
    return rows


def weighted_value(frame: pd.DataFrame, metric: str) -> float:
    return float(np.average(frame[metric], weights=frame["n"]))


def bootstrap_metrics(frame: pd.DataFrame, seed: int) -> dict:
    basins = frame["basin_id"].to_numpy()
    if len(np.unique(basins)) != len(frame):
        raise ValueError("Bootstrap input must contain one row per basin")
    weights = frame["n"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(frame), size=(5000, len(frame)))
    denominator = weights[draws].sum(axis=1)
    output = {}
    for metric in ["picp", "mpiw_log", "winkler_log"]:
        values = frame[metric].to_numpy(dtype=np.float64)
        samples = (values[draws] * weights[draws]).sum(axis=1) / denominator
        output[metric] = {
            "value": float(np.average(values, weights=weights)),
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    return output


def aggregate_curve(frame: pd.DataFrame) -> dict:
    output = {}
    for years in YEARS:
        output[str(years)] = {}
        for method in METHODS:
            output[str(years)][method] = {}
            for fraction in TOP_FLOW_FRACTIONS:
                selected = frame[
                    (frame["years"] == years)
                    & (frame["method"] == method)
                    & np.isclose(frame["top_flow_fraction"], fraction)
                ]
                basin_mean = selected.groupby("basin_id", as_index=False).agg({
                    "n": "first",
                    "picp": "mean",
                    "mpiw_log": "mean",
                    "winkler_log": "mean",
                })
                metrics = bootstrap_metrics(
                    basin_mean,
                    seed=20260730 + years * 100 + METHODS.index(method) * 10 + int(fraction * 100),
                )
                seed_values = {}
                for metric in ["picp", "mpiw_log", "winkler_log"]:
                    values = []
                    for model_seed in SEEDS:
                        seed_frame = selected[selected["model_seed"] == model_seed]
                        values.append(weighted_value(seed_frame, metric))
                    metrics[metric]["seed_values"] = values
                    metrics[metric]["seed_sd"] = float(np.std(values, ddof=1))
                metrics["n_per_seed"] = int(basin_mean["n"].sum())
                output[str(years)][method][f"{fraction:.2f}"] = metrics
    return output


def basin_calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    selected = frame[
        frame["years"].isin([1, 15])
        & np.isclose(frame["top_flow_fraction"], 1.0)
    ]
    basin = selected.groupby(
        ["years", "method", "basin_id"], as_index=False
    ).agg({
        "n": "first",
        "picp": "mean",
        "mpiw_log": "mean",
        "winkler_log": "mean",
    })
    basin["absolute_coverage_error"] = np.abs(basin["picp"] - 0.90)
    basin["within_0p85_0p95"] = basin["picp"].between(0.85, 0.95, inclusive="both")

    summary = {}
    for years in [1, 15]:
        summary[str(years)] = {}
        for method in METHODS:
            group = basin[(basin["years"] == years) & (basin["method"] == method)]
            summary[str(years)][method] = {
                "n_basins": int(len(group)),
                "median_picp": float(group["picp"].median()),
                "q25_picp": float(group["picp"].quantile(0.25)),
                "q75_picp": float(group["picp"].quantile(0.75)),
                "mean_absolute_coverage_error": float(group["absolute_coverage_error"].mean()),
                "fraction_within_0p85_0p95": float(group["within_0p85_0p95"].mean()),
                "pooled_picp": weighted_value(group, "picp"),
            }
    return basin, summary


def paired_predicted_improvement(frame: pd.DataFrame) -> dict:
    output = {}
    for years in YEARS:
        output[str(years)] = {}
        for fraction in TOP_FLOW_FRACTIONS:
            selected = frame[
                (frame["years"] == years)
                & np.isclose(frame["top_flow_fraction"], fraction)
                & frame["method"].isin(["global_cqr", "predicted_regime_cqr"])
            ]
            basin = selected.groupby(
                ["method", "basin_id"], as_index=False
            ).agg({"n": "first", "picp": "mean"})
            pivot = basin.pivot(index="basin_id", columns="method", values="picp")
            weights = basin[basin["method"] == "global_cqr"].set_index("basin_id").loc[pivot.index, "n"].to_numpy(dtype=float)
            delta = (
                pivot["predicted_regime_cqr"] - pivot["global_cqr"]
            ).to_numpy(dtype=float)
            rng = np.random.default_rng(20260810 + years * 100 + int(fraction * 100))
            draws = rng.integers(0, len(delta), size=(5000, len(delta)))
            samples = (delta[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1)
            output[str(years)][f"{fraction:.2f}"] = {
                "mean_delta_picp": float(np.average(delta, weights=weights)),
                "mean_delta_picp_ci95": [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ],
                "median_basin_delta_picp": float(np.median(delta)),
                "fraction_basins_positive": float(np.mean(delta > 0)),
            }
    return output


def main() -> None:
    results = {
        (years, seed): load_result(years, seed)
        for years in YEARS
        for seed in SEEDS
    }
    rows = []
    for years in YEARS:
        for seed in SEEDS:
            rows.extend(run_rows(years, seed, results[(years, seed)]))
    frame = pd.DataFrame(rows)
    frame.to_csv(RUN_CSV, index=False)

    curve = aggregate_curve(frame)
    basin, basin_summary = basin_calibration(frame)
    basin.to_csv(BASIN_CSV, index=False)
    predicted_improvement = paired_predicted_improvement(frame)
    output = {
        "protocol_version": "conditional_robustness_v1_saved_predictions",
        "top_flow_definition": "exact highest-flow share selected independently within each basin from observed test flow",
        "diagnostic_only": True,
        "top_flow_fractions": TOP_FLOW_FRACTIONS,
        "methods": METHODS,
        "model_seeds": SEEDS,
        "bootstrap": "5000 paired whole-basin resamples after averaging each basin over model seeds",
        "threshold_curve": curve,
        "basin_calibration_summary": basin_summary,
        "predicted_minus_global": predicted_improvement,
        "artifacts": {
            "by_run_csv": RUN_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "basin_calibration_csv": BASIN_CSV.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    one_year_high = predicted_improvement["1"]["0.33"]
    print(f"Wrote {SUMMARY_JSON}")
    print(
        "1-year top-33% predicted-minus-global PICP: "
        f"{100 * one_year_high['mean_delta_picp']:+.2f} pp "
        f"(95% CI {100 * one_year_high['mean_delta_picp_ci95'][0]:+.2f}, "
        f"{100 * one_year_high['mean_delta_picp_ci95'][1]:+.2f})"
    )


if __name__ == "__main__":
    main()
