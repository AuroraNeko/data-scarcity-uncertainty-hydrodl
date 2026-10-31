"""Summarize replicated-subset and equal-update scarcity experiments.

The endpoint analyses average each basin over the three model seeds before
forming paired 15-year minus 1-year effects. Replicated-subset uncertainty is
estimated with a two-stage bootstrap that resamples training subsets and then
basins within each selected subset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.corrections.common import (  # noqa: E402
    basin_interval_metrics,
    metric_value,
)


MODEL_SEEDS = [42, 123, 456]
ENDPOINT_YEARS = [1, 15]
SUBSET_IDS = [
    "primary_seed42",
    "stratified_1",
    "stratified_2",
    "stratified_3",
    "stratified_4",
]
REPLICATION_PROTOCOL = "scarcity_v3_replicated_subsets_fixed_epochs"
EQUAL_UPDATE_PROTOCOL = "scarcity_v3_equal_optimizer_updates"
STANDARD_LSTM_PROTOCOL = "scarcity_v3_standard_lstm_endpoint_sensitivity"
ORIGINAL_PROTOCOL = "scarcity_v2_fixed_end_fixed_memory"

REPLICATION_DIR = ROOT / "results" / "tables" / "scarcity_replications"
EQUAL_UPDATE_DIR = ROOT / "results" / "tables" / "scarcity_equal_updates"
ARCHITECTURE_DIR = ROOT / "results" / "tables" / "scarcity_standard_lstm"
ORIGINAL_DIR = ROOT / "results" / "tables" / "corrected_scarcity"
SUBSET_TABLE = REPLICATION_DIR / "replication_subsets.csv"

INTERVAL_METRICS = [
    "raw_picp",
    "cqr_picp",
    "raw_mpiw_log",
    "cqr_mpiw_log",
    "raw_winkler_log",
    "cqr_winkler_log",
    "raw_high_flow_picp",
    "cqr_high_flow_picp",
]
ENDPOINT_METRICS = ["nse", *INTERVAL_METRICS]
STATISTIC = {metric: ("median" if metric == "nse" else "mean") for metric in ENDPOINT_METRICS}
RUN_METRICS = [
    "median_nse",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    path = ROOT.joinpath(*parts)
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {value!r}") from exc
    return path


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required experiment result not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_subset_table() -> dict[str, list[str]]:
    with SUBSET_TABLE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    for subset_id in SUBSET_IDS:
        selected = [row for row in rows if row["subset_id"] == subset_id]
        selected.sort(key=lambda row: int(row["selection_order"]))
        basin_ids = [row["basin_id"] for row in selected]
        if len(basin_ids) != 50 or len(set(basin_ids)) != 50:
            raise ValueError(f"Subset {subset_id} must contain 50 unique basins")
        output[subset_id] = basin_ids
    all_basins = [basin for subset in output.values() for basin in subset]
    if len(set(all_basins)) != 250:
        raise ValueError("The five replication subsets must be mutually disjoint")
    return output


def result_path(group: str, subset_id: str, years: int, seed: int) -> Path:
    if subset_id == "primary_seed42" and group == "fixed_epochs":
        return ORIGINAL_DIR / f"scarcity_{years}yr_basinseed42_modelseed{seed}.json"
    directory = {
        "replications": REPLICATION_DIR,
        "fixed_updates": EQUAL_UPDATE_DIR,
        "standard_lstm": ARCHITECTURE_DIR,
    }[group]
    return directory / f"scarcity_{years}yr_subset{subset_id}_modelseed{seed}.json"


def validate_result(
    path: Path,
    result: dict,
    *,
    subset_id: str,
    expected_basins: list[str],
    years: int,
    seed: int,
    protocol: str,
) -> None:
    expected = {
        "protocol_version": protocol,
        "years": years,
        "model_seed": seed,
        "seq_len": 180,
        "batch_size": 512,
        "epochs": 30,
        "calibration_target_period": ["1995-10-01", "2000-09-30"],
        "test_target_period": ["2000-10-01", "2008-12-31"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{path.name}: {key}={result.get(key)!r}, expected {value!r}")
    if result["selected_basins"] != expected_basins:
        raise ValueError(f"{path.name}: selected basins do not match {subset_id}")
    if protocol != ORIGINAL_PROTOCOL and result.get("subset_id") != subset_id:
        raise ValueError(f"{path.name}: unexpected subset_id")
    if protocol != ORIGINAL_PROTOCOL:
        if result.get("checkpoint_metadata_sanitized") is not True:
            raise ValueError(f"{path.name}: checkpoint metadata has not passed the path audit")
        tensor_hash = result.get("checkpoint_tensor_sha256", "")
        if len(tensor_hash) != 64:
            raise ValueError(f"{path.name}: missing stable model-state tensor digest")
    if protocol == REPLICATION_PROTOCOL and result.get("training_budget") != "fixed_epochs":
        raise ValueError(f"{path.name}: replication must use fixed epochs")
    if protocol == EQUAL_UPDATE_PROTOCOL:
        if result.get("training_budget") != "fixed_updates":
            raise ValueError(f"{path.name}: equal-update run must use fixed updates")
        if result.get("optimizer_steps") != 1080:
            raise ValueError(f"{path.name}: expected 1080 optimizer steps")
    if protocol == STANDARD_LSTM_PROTOCOL:
        if result.get("architecture") != "standard_lstm":
            raise ValueError(f"{path.name}: expected standard_lstm architecture")
        if result.get("training_budget") != "fixed_epochs":
            raise ValueError(f"{path.name}: architecture sensitivity must use fixed epochs")

    for path_key, hash_key in [
        ("checkpoint", "checkpoint_sha256"),
        ("predictions", "predictions_sha256"),
    ]:
        artifact = repository_path(result[path_key])
        if sha256(artifact) != result[hash_key]:
            raise ValueError(f"Hash mismatch for {artifact}")


def load_results(
    group: str,
    subset_ids: list[str],
    subset_basins: dict[str, list[str]],
) -> dict[tuple[str, int, int], dict]:
    results = {}
    for subset_id in subset_ids:
        for years in ENDPOINT_YEARS:
            for seed in MODEL_SEEDS:
                path = result_path(group, subset_id, years, seed)
                result = read_json(path)
                if group == "fixed_epochs":
                    protocol = ORIGINAL_PROTOCOL
                elif group == "replications":
                    protocol = REPLICATION_PROTOCOL
                elif group == "fixed_updates":
                    protocol = EQUAL_UPDATE_PROTOCOL
                else:
                    protocol = STANDARD_LSTM_PROTOCOL
                validate_result(
                    path,
                    result,
                    subset_id=subset_id,
                    expected_basins=subset_basins[subset_id],
                    years=years,
                    seed=seed,
                    protocol=protocol,
                )
                results[(subset_id, years, seed)] = result
    return results


def run_table(results: dict[tuple[str, int, int], dict], path: Path) -> None:
    fields = ["subset_id", "years", "model_seed", *RUN_METRICS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subset_id, years, seed in sorted(results):
            result = results[(subset_id, years, seed)]
            writer.writerow({
                "subset_id": subset_id,
                "years": years,
                "model_seed": seed,
                **{metric: metric_value(result, metric) for metric in RUN_METRICS},
            })


def statistic(values: np.ndarray, name: str) -> float:
    return float(np.median(values) if name == "median" else np.mean(values))


def bootstrap_effect(values: np.ndarray, statistic_name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    sampled = values[draws]
    if statistic_name == "median":
        estimates = np.median(sampled, axis=1)
    else:
        estimates = np.mean(sampled, axis=1)
    return {
        "statistic": statistic_name,
        "estimate": statistic(values, statistic_name),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "fraction_basins_positive": float(np.mean(values > 0)),
        "n_basins": int(len(values)),
    }


def two_stage_bootstrap(
    values_by_subset: dict[str, np.ndarray],
    statistic_name: str,
    seed: int,
) -> dict:
    subset_ids = list(values_by_subset)
    values = np.concatenate([values_by_subset[subset_id] for subset_id in subset_ids])
    rng = np.random.default_rng(seed)
    estimates = np.empty(5000, dtype=np.float64)
    for index in range(len(estimates)):
        chosen_subsets = rng.integers(0, len(subset_ids), size=len(subset_ids))
        sampled = []
        for subset_index in chosen_subsets:
            local = values_by_subset[subset_ids[subset_index]]
            sampled.append(rng.choice(local, size=len(local), replace=True))
        estimates[index] = statistic(np.concatenate(sampled), statistic_name)
    return {
        "statistic": statistic_name,
        "estimate": statistic(values, statistic_name),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "fraction_basins_positive": float(np.mean(values > 0)),
        "n_basins": int(len(values)),
        "n_training_subsets": len(subset_ids),
        "bootstrap": "resample training subsets, then basins within each selected subset",
        "bootstrap_repetitions": int(len(estimates)),
    }


def basin_seed_averages(
    results: dict[tuple[str, int, int], dict], subset_id: str
) -> dict[int, dict[str, dict[str, float]]]:
    per_run = {}
    for years in ENDPOINT_YEARS:
        for seed in MODEL_SEEDS:
            result = results[(subset_id, years, seed)]
            interval = basin_interval_metrics(result)
            point = result["point_metrics"]["by_basin"]
            if set(interval) != set(point):
                raise ValueError(f"Point and interval basin IDs differ for {subset_id}, {years}, {seed}")
            per_run[(years, seed)] = {
                basin: {"nse": float(point[basin]), **interval[basin]}
                for basin in sorted(point)
            }

    output = {}
    for years in ENDPOINT_YEARS:
        basin_ids = sorted(per_run[(years, MODEL_SEEDS[0])])
        output[years] = {}
        for basin in basin_ids:
            output[years][basin] = {
                metric: float(np.mean([
                    per_run[(years, seed)][basin][metric]
                    for seed in MODEL_SEEDS
                ]))
                for metric in ENDPOINT_METRICS
            }
    if set(output[1]) != set(output[15]) or len(output[1]) != 50:
        raise ValueError(f"Endpoint basin mismatch for {subset_id}")
    return output


def endpoint_deltas(
    averaged: dict[int, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    return {
        basin: {
            metric: averaged[15][basin][metric] - averaged[1][basin][metric]
            for metric in ENDPOINT_METRICS
        }
        for basin in sorted(averaged[1])
    }


def summarize_replications(
    primary: dict[tuple[str, int, int], dict],
    replicated: dict[tuple[str, int, int], dict],
) -> dict:
    results = {**primary, **replicated}
    delta_rows = []
    deltas_by_subset = {}
    subset_summaries = {}
    for subset_index, subset_id in enumerate(SUBSET_IDS):
        averaged = basin_seed_averages(results, subset_id)
        deltas = endpoint_deltas(averaged)
        deltas_by_subset[subset_id] = {
            metric: np.asarray([deltas[basin][metric] for basin in sorted(deltas)])
            for metric in ENDPOINT_METRICS
        }
        subset_summaries[subset_id] = {
            "n_basins": len(deltas),
            "effects_15_minus_1": {
                metric: bootstrap_effect(
                    deltas_by_subset[subset_id][metric],
                    STATISTIC[metric],
                    20260720 + subset_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(ENDPOINT_METRICS)
            },
        }
        for basin in sorted(deltas):
            delta_rows.append({
                "subset_id": subset_id,
                "basin_id": basin,
                **{f"delta_{metric}_15_minus_1": deltas[basin][metric] for metric in ENDPOINT_METRICS},
            })

    combined = {
        metric: two_stage_bootstrap(
            {subset_id: deltas_by_subset[subset_id][metric] for subset_id in SUBSET_IDS},
            STATISTIC[metric],
            20260730 + metric_index,
        )
        for metric_index, metric in enumerate(ENDPOINT_METRICS)
    }

    effects_path = REPLICATION_DIR / "scarcity_replication_basin_effects.csv"
    fields = [
        "subset_id",
        "basin_id",
        *[f"delta_{metric}_15_minus_1" for metric in ENDPOINT_METRICS],
    ]
    with effects_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(delta_rows)

    return {
        "protocol_version": REPLICATION_PROTOCOL,
        "design": {
            "endpoint_years": ENDPOINT_YEARS,
            "model_seeds": MODEL_SEEDS,
            "training_subsets": SUBSET_IDS,
            "n_basins_per_subset": 50,
            "n_unique_basins": 250,
            "new_model_fits": 24,
            "primary_model_fits_reused": 6,
            "seed_handling": "per-basin metrics averaged across three model seeds before pairing",
        },
        "by_subset": subset_summaries,
        "combined_effects_15_minus_1": combined,
        "basin_effect_table": effects_path.relative_to(ROOT).as_posix(),
    }


def run_level_summary(results: dict[tuple[str, int, int], dict], subset_id: str) -> dict:
    output = {}
    for years in ENDPOINT_YEARS:
        output[str(years)] = {}
        for metric in RUN_METRICS:
            values = np.asarray([
                metric_value(results[(subset_id, years, seed)], metric)
                for seed in MODEL_SEEDS
            ], dtype=np.float64)
            output[str(years)][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
    return output


def optimizer_budget(result: dict, group: str) -> dict:
    if group == "fixed_updates":
        return {
            "optimizer_steps": int(result["optimizer_steps"]),
            "observed_targets_seen": int(result["observed_targets_seen"]),
        }
    steps = math.ceil(result["train_windows"] / result["batch_size"]) * result["epochs"]
    return {
        "optimizer_steps": int(steps),
        "observed_targets_seen": int(result["train_observed_targets"] * result["epochs"]),
    }


def summarize_equal_updates(
    fixed_epochs: dict[tuple[str, int, int], dict],
    fixed_updates: dict[tuple[str, int, int], dict],
) -> dict:
    group_results = {"fixed_epochs": fixed_epochs, "fixed_updates": fixed_updates}
    output = {}
    for group_index, (group, results) in enumerate(group_results.items()):
        averaged = basin_seed_averages(results, "primary_seed42")
        deltas = endpoint_deltas(averaged)
        output[group] = {
            "by_year": run_level_summary(results, "primary_seed42"),
            "optimizer_budget": {
                str(years): optimizer_budget(results[("primary_seed42", years, 42)], group)
                for years in ENDPOINT_YEARS
            },
            "effects_15_minus_1": {
                metric: bootstrap_effect(
                    np.asarray([deltas[basin][metric] for basin in sorted(deltas)]),
                    STATISTIC[metric],
                    20260810 + group_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(ENDPOINT_METRICS)
            },
        }
    return {
        "protocol_version": EQUAL_UPDATE_PROTOCOL,
        "design": {
            "subset_id": "primary_seed42",
            "endpoint_years": ENDPOINT_YEARS,
            "model_seeds": MODEL_SEEDS,
            "fixed_update_budget": 1080,
            "purpose": "separate record-length evidence from the optimizer-step exposure induced by fixed epochs",
        },
        "training_budgets": output,
    }


def summarize_architecture_sensitivity(
    lpu_stream: dict[tuple[str, int, int], dict],
    standard_lstm: dict[tuple[str, int, int], dict],
) -> dict:
    architecture_results = {
        "lpu_stream": lpu_stream,
        "standard_lstm": standard_lstm,
    }
    output = {}
    for architecture_index, (architecture, results) in enumerate(architecture_results.items()):
        averaged = basin_seed_averages(results, "primary_seed42")
        deltas = endpoint_deltas(averaged)
        reference = results[("primary_seed42", 1, MODEL_SEEDS[0])]
        output[architecture] = {
            "n_parameters": int(reference["n_parameters"]),
            "by_year": run_level_summary(results, "primary_seed42"),
            "effects_15_minus_1": {
                metric: bootstrap_effect(
                    np.asarray([deltas[basin][metric] for basin in sorted(deltas)]),
                    STATISTIC[metric],
                    20260830 + architecture_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(ENDPOINT_METRICS)
            },
        }
    return {
        "protocol_version": STANDARD_LSTM_PROTOCOL,
        "design": {
            "subset_id": "primary_seed42",
            "endpoint_years": ENDPOINT_YEARS,
            "model_seeds": MODEL_SEEDS,
            "training_budget": "30 fixed epochs",
            "shared_settings": "dates, normalization rule, inputs, hidden size, final-hidden/head dropout, optimizer, quantiles, CQR, and evaluation",
            "architectural_contrast": (
                "LPU-Stream injects a learned static embedding at every time step; "
                "the standard LSTM fuses standardized static attributes only in its output head"
            ),
        },
        "architectures": output,
    }


def main() -> None:
    subset_basins = load_subset_table()
    primary = load_results("fixed_epochs", ["primary_seed42"], subset_basins)
    replicated = load_results("replications", SUBSET_IDS[1:], subset_basins)
    equal_updates = load_results("fixed_updates", ["primary_seed42"], subset_basins)
    standard_lstm = load_results("standard_lstm", ["primary_seed42"], subset_basins)

    replication_summary = summarize_replications(primary, replicated)
    replication_path = REPLICATION_DIR / "scarcity_replication_summary.json"
    replication_path.write_text(json.dumps(replication_summary, indent=2), encoding="utf-8")
    run_table({**primary, **replicated}, REPLICATION_DIR / "scarcity_replication_runs.csv")

    equal_summary = summarize_equal_updates(primary, equal_updates)
    equal_path = EQUAL_UPDATE_DIR / "equal_update_summary.json"
    equal_path.write_text(json.dumps(equal_summary, indent=2), encoding="utf-8")
    run_table(equal_updates, EQUAL_UPDATE_DIR / "equal_update_runs.csv")

    architecture_summary = summarize_architecture_sensitivity(primary, standard_lstm)
    architecture_path = ARCHITECTURE_DIR / "architecture_sensitivity_summary.json"
    architecture_path.write_text(json.dumps(architecture_summary, indent=2), encoding="utf-8")
    run_table(standard_lstm, ARCHITECTURE_DIR / "standard_lstm_runs.csv")

    nse = replication_summary["combined_effects_15_minus_1"]["nse"]
    picp = replication_summary["combined_effects_15_minus_1"]["raw_picp"]
    print("Verified 30 endpoint runs across five disjoint subsets")
    print(
        f"Combined NSE effect: {nse['estimate']:.3f} "
        f"[{nse['ci95'][0]:.3f}, {nse['ci95'][1]:.3f}]"
    )
    print(
        f"Combined raw PICP effect: {100 * picp['estimate']:.2f} pp "
        f"[{100 * picp['ci95'][0]:.2f}, {100 * picp['ci95'][1]:.2f}]"
    )
    print(f"Replication summary: {replication_path}")
    print(f"Equal-update summary: {equal_path}")
    print(f"Architecture summary: {architecture_path}")


if __name__ == "__main__":
    main()
