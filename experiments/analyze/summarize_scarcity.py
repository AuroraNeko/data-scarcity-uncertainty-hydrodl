"""Aggregate and diagnose the controlled scarcity experiment artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.corrections.scarcity_runs import (  # noqa: E402,F401
    ALPHA,
    PROTOCOL_VERSION,
    SEEDS,
    TABLE_DIR,
    YEARS,
    artifact_path,
    basin_interval_metrics,
    load_predictions,
    load_result,
    metric_value,
    per_basin_regimes,
    reference_regimes,
)
from src.evaluation.protocol import project_nonempty  # noqa: E402
from src.losses.cqr import compute_uncertainty_metrics, finite_sample_quantile  # noqa: E402
from src.utils import repository_path as _repository_path, sha256_file  # noqa: E402


def repository_path(value: str) -> Path:
    """Resolve an archived repository-relative path against the project root."""
    return _repository_path(PROJECT_ROOT, value)


def sha256(path: Path) -> str:
    return sha256_file(path)


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def group_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
) -> dict:
    covered = (observed >= lower) & (observed <= upper)
    output = {}
    for name in ["low", "normal", "high"]:
        selected = groups == name
        output[name] = {
            "picp": float(covered[selected].mean()),
            "mpiw_log": float((upper[selected] - lower[selected]).mean()),
            "n": int(selected.sum()),
        }
    return output


def conditional_cqr(result: dict) -> dict:
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
        raise ValueError("Stored q_cal does not match the saved predictions")

    observed_test_regimes = per_basin_regimes(test_target, test_basins)
    raw_lower = test_prediction[:, 0]
    raw_upper = test_prediction[:, 2]
    global_lower = raw_lower - global_q
    global_upper = raw_upper + global_q

    predicted_val_regimes = per_basin_regimes(val_prediction[:, 1], val_basins)
    predicted_test_regimes = reference_regimes(
        val_prediction[:, 1],
        val_basins,
        test_prediction[:, 1],
        test_basins,
    )
    predicted_adjustments = {}
    sample_adjustment = np.zeros(len(test_target), dtype=np.float64)
    for name in ["low", "normal", "high"]:
        adjustment = finite_sample_quantile(scores[predicted_val_regimes == name], ALPHA)
        predicted_adjustments[name] = adjustment
        sample_adjustment[predicted_test_regimes == name] = adjustment
    predicted_lower = raw_lower - sample_adjustment
    predicted_upper = raw_upper + sample_adjustment
    predicted_lower, predicted_upper, _ = project_nonempty(predicted_lower, predicted_upper)

    observed_val_regimes = per_basin_regimes(val_target, val_basins)
    observed_test_assignment = reference_regimes(
        val_target, val_basins, test_target, test_basins
    )
    oracle_adjustments = {}
    oracle_sample_adjustment = np.zeros(len(test_target), dtype=np.float64)
    for name in ["low", "normal", "high"]:
        adjustment = finite_sample_quantile(scores[observed_val_regimes == name], ALPHA)
        oracle_adjustments[name] = adjustment
        oracle_sample_adjustment[observed_test_assignment == name] = adjustment
    oracle_lower = raw_lower - oracle_sample_adjustment
    oracle_upper = raw_upper + oracle_sample_adjustment
    oracle_lower, oracle_upper, _ = project_nonempty(oracle_lower, oracle_upper)

    return {
        "global_q_cal_log": global_q,
        "predicted_regime_q_cal_log": predicted_adjustments,
        "oracle_observed_regime_q_cal_log": oracle_adjustments,
        "overall": {
            "raw": compute_uncertainty_metrics(raw_lower, raw_upper, test_target, ALPHA),
            "global_cqr": compute_uncertainty_metrics(global_lower, global_upper, test_target, ALPHA),
            "predicted_regime_cqr": compute_uncertainty_metrics(
                predicted_lower, predicted_upper, test_target, ALPHA
            ),
            "oracle_observed_regime_cqr": compute_uncertainty_metrics(
                oracle_lower, oracle_upper, test_target, ALPHA
            ),
        },
        "by_observed_test_regime": {
            "raw": group_coverage(raw_lower, raw_upper, test_target, observed_test_regimes),
            "global_cqr": group_coverage(
                global_lower, global_upper, test_target, observed_test_regimes
            ),
            "predicted_regime_cqr": group_coverage(
                predicted_lower, predicted_upper, test_target, observed_test_regimes
            ),
            "oracle_observed_regime_cqr": group_coverage(
                oracle_lower, oracle_upper, test_target, observed_test_regimes
            ),
        },
        "by_predicted_test_regime": group_coverage(
            predicted_lower,
            predicted_upper,
            test_target,
            predicted_test_regimes,
        ),
    }


def calibration_window_sensitivity(result: dict) -> dict:
    data = load_predictions(result)
    val_valid = data["val_mask"] > 0
    test_valid = data["test_mask"] > 0
    val_prediction = data["val_prediction_log"][val_valid]
    val_target = data["val_target_log"][val_valid]
    val_dates = data["val_target_dates"][val_valid]
    test_prediction = data["test_prediction_log"][test_valid]
    test_target = data["test_target_log"][test_valid]
    starts = {
        1: np.datetime64("1999-10-01"),
        2: np.datetime64("1998-10-01"),
        3: np.datetime64("1997-10-01"),
        4: np.datetime64("1996-10-01"),
        5: np.datetime64("1995-10-01"),
    }
    output = {}
    for years, start in starts.items():
        selected = val_dates >= start
        scores = np.maximum(
            val_prediction[selected, 0] - val_target[selected],
            val_target[selected] - val_prediction[selected, 2],
        )
        adjustment = finite_sample_quantile(scores, ALPHA)
        lower = test_prediction[:, 0] - adjustment
        upper = test_prediction[:, 2] + adjustment
        output[str(years)] = {
            "n_calibration": int(selected.sum()),
            "q_cal_log": adjustment,
            **compute_uncertainty_metrics(lower, upper, test_target, ALPHA),
        }
    return output


def per_basin_endpoint_effect(results: dict[tuple[int, int], dict]) -> dict:
    by_seed = {}
    basin_deltas = {}
    for seed in SEEDS:
        short = results[(1, seed)]["point_metrics"]["by_basin"]
        long = results[(15, seed)]["point_metrics"]["by_basin"]
        common = sorted(set(short) & set(long))
        delta = {basin: float(long[basin] - short[basin]) for basin in common}
        by_seed[str(seed)] = {
            "median_delta_nse_15_minus_1": float(np.median(list(delta.values()))),
            "fraction_improved_with_15_years": float(np.mean(np.asarray(list(delta.values())) > 0)),
        }
        for basin, value in delta.items():
            basin_deltas.setdefault(basin, []).append(value)

    averaged_by_basin = {
        basin: float(np.mean(basin_deltas[basin]))
        for basin in sorted(basin_deltas)
    }
    averaged = np.asarray(list(averaged_by_basin.values()), dtype=np.float64)
    rng = np.random.default_rng(20260711)
    bootstrap = np.empty(5000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = np.median(
            rng.choice(averaged, size=len(averaged), replace=True)
        )
    return {
        "by_seed": by_seed,
        "by_basin": {
            basin: {"delta_nse_15_minus_1": value}
            for basin, value in averaged_by_basin.items()
        },
        "across_seed_average": {
            "n_basins": int(len(averaged)),
            "median_delta_nse_15_minus_1": float(np.median(averaged)),
            "q25": float(np.quantile(averaged, 0.25)),
            "q75": float(np.quantile(averaged, 0.75)),
            "fraction_improved_with_15_years": float(np.mean(averaged > 0)),
            "median_delta_ci95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        },
    }


def basin_interval_metrics(result: dict) -> dict[str, dict[str, float]]:
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
                + (2 / ALPHA) * np.maximum(lower - target, 0)
                + (2 / ALPHA) * np.maximum(target - upper, 0)
            )
            basin_metrics[f"{method}_picp"] = float(covered[selected].mean())
            basin_metrics[f"{method}_mpiw_log"] = float(width[selected].mean())
            basin_metrics[f"{method}_winkler_log"] = float(winkler[selected].mean())
            basin_metrics[f"{method}_high_flow_picp"] = float(covered[high].mean())
        output[str(basin)] = basin_metrics
    return output


def basin_cluster_endpoint_intervals(results: dict[tuple[int, int], dict]) -> dict:
    """Paired endpoint effects after averaging each basin over model seeds."""
    metrics = [
        "raw_picp",
        "cqr_picp",
        "raw_mpiw_log",
        "cqr_mpiw_log",
        "raw_winkler_log",
        "cqr_winkler_log",
        "raw_high_flow_picp",
        "cqr_high_flow_picp",
    ]
    basin_values = {
        years: {
            seed: basin_interval_metrics(results[(years, seed)])
            for seed in SEEDS
        }
        for years in [1, 15]
    }
    basin_ids = sorted(basin_values[1][SEEDS[0]])
    rng = np.random.default_rng(20260715)
    draws = rng.integers(0, len(basin_ids), size=(5000, len(basin_ids)))
    output = {}
    by_basin = {basin: {} for basin in basin_ids}
    for metric in metrics:
        endpoint = {}
        arrays = {}
        for years in [1, 15]:
            arrays[years] = np.asarray([
                np.mean([
                    basin_values[years][seed][basin][metric]
                    for seed in SEEDS
                ])
                for basin in basin_ids
            ], dtype=np.float64)
            samples = arrays[years][draws].mean(axis=1)
            endpoint[f"{years}yr"] = {
                "mean": float(arrays[years].mean()),
                "ci95": [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ],
            }
        delta = arrays[15] - arrays[1]
        for basin, value in zip(basin_ids, delta):
            by_basin[basin][f"delta_{metric}_15_minus_1"] = float(value)
        delta_samples = delta[draws].mean(axis=1)
        endpoint["difference_15_minus_1"] = {
            "mean": float(delta.mean()),
            "ci95": [
                float(np.percentile(delta_samples, 2.5)),
                float(np.percentile(delta_samples, 97.5)),
            ],
            "fraction_basins_positive": float((delta > 0).mean()),
        }
        output[metric] = endpoint
    return {
        "unit": "basin",
        "n_basins": len(basin_ids),
        "model_seed_handling": "metric averaged across seeds 42, 123, and 456 before paired resampling",
        "bootstrap_repetitions": int(len(draws)),
        "metrics": output,
        "by_basin": by_basin,
    }


def write_basin_endpoint_effects(point_effect: dict, interval_effect: dict) -> Path:
    """Write the basin-level values underlying the paired endpoint figure."""
    point_rows = point_effect["by_basin"]
    interval_rows = interval_effect["by_basin"]
    basin_ids = sorted(point_rows)
    if basin_ids != sorted(interval_rows):
        raise ValueError("Point and interval endpoint analyses use different basins")

    fields = [
        "basin_id",
        "delta_nse_15_minus_1",
        "delta_raw_picp_15_minus_1",
        "delta_cqr_picp_15_minus_1",
        "delta_raw_mpiw_log_15_minus_1",
        "delta_cqr_mpiw_log_15_minus_1",
        "delta_raw_winkler_log_15_minus_1",
        "delta_cqr_winkler_log_15_minus_1",
        "delta_raw_high_flow_picp_15_minus_1",
        "delta_cqr_high_flow_picp_15_minus_1",
    ]
    path = TABLE_DIR / "basin_endpoint_effects.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for basin in basin_ids:
            writer.writerow({
                "basin_id": basin,
                **point_rows[basin],
                **interval_rows[basin],
            })
    return path


def main() -> None:
    results = {(years, seed): load_result(years, seed) for years in YEARS for seed in SEEDS}

    reference_basins = results[(1, 42)]["selected_basins"]
    for result in results.values():
        if result["selected_basins"] != reference_basins:
            raise ValueError("Basin sets differ across controlled runs")

    reference_metadata = None
    for result in results.values():
        data = load_predictions(result)
        metadata = (
            data["val_basin_ids"],
            data["val_target_dates"],
            data["val_mask"],
            data["test_basin_ids"],
            data["test_target_dates"],
            data["test_mask"],
        )
        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for actual, expected in zip(metadata, reference_metadata):
                if not np.array_equal(actual, expected):
                    raise ValueError("Calibration/test samples differ across runs")

    metric_names = [
        "median_nse",
        "pooled_log_nse",
        "raw_picp",
        "cqr_picp",
        "raw_mpiw_log",
        "cqr_mpiw_log",
        "raw_winkler_log",
        "cqr_winkler_log",
        "raw_high_flow_picp",
        "cqr_high_flow_picp",
        "q_cal_log",
    ]
    by_year = {
        str(years): {
            metric: summary([metric_value(results[(years, seed)], metric) for seed in SEEDS])
            for metric in metric_names
        }
        for years in YEARS
    }
    endpoint = {
        metric: summary([
            metric_value(results[(15, seed)], metric)
            - metric_value(results[(1, seed)], metric)
            for seed in SEEDS
        ])
        for metric in metric_names
    }

    conditional = {
        f"{years}yr_seed{seed}": conditional_cqr(results[(years, seed)])
        for years in YEARS
        for seed in SEEDS
    }
    conditional_by_year = {}
    for years in YEARS:
        conditional_by_year[str(years)] = {}
        for method in ["raw", "global_cqr", "predicted_regime_cqr", "oracle_observed_regime_cqr"]:
            conditional_by_year[str(years)][method] = {}
            for regime in ["low", "normal", "high"]:
                values = [
                    conditional[f"{years}yr_seed{seed}"]["by_observed_test_regime"][method][regime]["picp"]
                    for seed in SEEDS
                ]
                conditional_by_year[str(years)][method][regime] = summary(values)

    sensitivity = {
        str(seed): calibration_window_sensitivity(results[(15, seed)])
        for seed in SEEDS
    }
    point_endpoint = per_basin_endpoint_effect(results)
    interval_endpoint = basin_cluster_endpoint_intervals(results)
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "training_target_periods": {
                str(years): results[(years, SEEDS[0])]["train_target_period"]
                for years in YEARS
            },
            "calibration_target_period": results[(YEARS[0], SEEDS[0])]["calibration_target_period"],
            "test_target_period": results[(YEARS[0], SEEDS[0])]["test_target_period"],
            "sequence_length_days": results[(YEARS[0], SEEDS[0])]["seq_len"],
            "batch_size": results[(YEARS[0], SEEDS[0])]["batch_size"],
            "epochs": results[(YEARS[0], SEEDS[0])]["epochs"],
            "checkpoint_selection": results[(YEARS[0], SEEDS[0])]["checkpoint_selection"],
            "n_model_fits": len(YEARS) * len(SEEDS),
        },
        "years": YEARS,
        "model_seeds": SEEDS,
        "n_basins": len(reference_basins),
        "by_year": by_year,
        "paired_endpoint_effect_15_minus_1": endpoint,
        "per_basin_endpoint_effect": point_endpoint,
        "basin_cluster_endpoint_intervals": interval_endpoint,
        "conditional_cqr_by_run": conditional,
        "conditional_cqr_by_year": conditional_by_year,
        "calibration_window_sensitivity_15yr": sensitivity,
        "artifacts": {
            f"{years}yr_seed{seed}": {
                "result": artifact_path(years, seed).relative_to(PROJECT_ROOT).as_posix(),
                "result_sha256": sha256(artifact_path(years, seed)),
                "checkpoint_sha256": results[(years, seed)]["checkpoint_sha256"],
                "predictions_sha256": results[(years, seed)]["predictions_sha256"],
            }
            for years in YEARS
            for seed in SEEDS
        },
    }
    summary_path = TABLE_DIR / "scarcity_summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    basin_effect_path = write_basin_endpoint_effects(point_endpoint, interval_endpoint)

    csv_path = TABLE_DIR / "scarcity_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["years", "model_seed", *metric_names])
        writer.writeheader()
        for years in YEARS:
            for seed in SEEDS:
                writer.writerow({
                    "years": years,
                    "model_seed": seed,
                    **{metric: metric_value(results[(years, seed)], metric) for metric in metric_names},
                })

    print(f"Verified and summarized {len(results)} controlled runs")
    print(f"Summary: {summary_path}")
    print(f"Run table: {csv_path}")
    print(f"Basin endpoint table: {basin_effect_path}")
    for years in YEARS:
        values = by_year[str(years)]
        print(
            f"{years:>2} yr: NSE={values['median_nse']['mean']:.3f}+/-{values['median_nse']['sd']:.3f}, "
            f"raw PICP={values['raw_picp']['mean']:.3f}+/-{values['raw_picp']['sd']:.3f}, "
            f"CQR={values['cqr_picp']['mean']:.3f}+/-{values['cqr_picp']['sd']:.3f}"
        )


if __name__ == "__main__":
    main()
