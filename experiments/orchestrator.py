"""Run the corrected, resumable experiment and manuscript-analysis pipeline.

The orchestrator never terminates unrelated processes. Each stage checks its
own authoritative artifact and skips work that is already complete.

Data download and preprocessing must be completed first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TABLES = ROOT / "results" / "tables"
CHECKPOINTS = ROOT / "results" / "checkpoints"
LOG_DIR = ROOT / "results" / "logs"
FULL_TRAINING_DIR = TABLES / "full_checkpoint_training"
SCARCITY_PROTOCOL = "scarcity_v2_fixed_end_fixed_memory"
SCARCITY_TARGET_COUNTS = {1: 18250, 3: 54750, 5: 91300, 15: 266322}
FULL_MODEL_CONFIG = {
    "n_dynamic": 15,
    "n_static": 13,
    "hidden_size": 128,
    "embed_dim": 32,
    "dropout": 0.3,
    "seq_len": 365,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "epochs": 30,
}


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    """Resolve a stored repository-relative path on Windows or POSIX."""
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


def artifact_matches(result: dict, path_key: str, hash_key: str) -> bool:
    try:
        path = repository_path(result[path_key])
        expected = result[hash_key]
        return (
            path.is_file()
            and isinstance(expected, str)
            and len(expected) == 64
            and sha256(path) == expected
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run(command: list[str], timeout_seconds: int, log_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    print(f"RUN {' '.join(command)}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )
    print(f"DONE {log_name} ({elapsed:.1f} s)", flush=True)


def scarcity_result_path(years: int, seed: int) -> Path:
    return (
        TABLES
        / "corrected_scarcity"
        / f"scarcity_{years}yr_basinseed42_modelseed{seed}.json"
    )


def scarcity_complete(years: int, seed: int) -> bool:
    result = load_json(scarcity_result_path(years, seed))
    return bool(
        result
        and result.get("protocol_version") == SCARCITY_PROTOCOL
        and result.get("years") == years
        and result.get("n_basins") == 50
        and result.get("basin_seed") == 42
        and result.get("model_seed") == seed
        and result.get("seq_len") == 180
        and result.get("batch_size") == 512
        and result.get("epochs") == 30
        and result.get("checkpoint_selection")
        == "fixed_epoch_budget_final_checkpoint"
        and result.get("train_observed_targets") == SCARCITY_TARGET_COUNTS[years]
        and result.get("calibration_windows") == 91350
        and result.get("calibration_observed_targets") == 91350
        and result.get("test_windows") == 150700
        and result.get("test_observed_targets") == 149239
        and result.get("max_dynamic_missing_fraction") == 0.0
        and result.get("calibration_target_period")
        == ["1995-10-01", "2000-09-30"]
        and result.get("test_target_period") == ["2000-10-01", "2008-12-31"]
        and artifact_matches(result, "checkpoint", "checkpoint_sha256")
        and artifact_matches(result, "predictions", "predictions_sha256")
    )


def checkpoint_config_matches(
    path: Path,
    seed: int,
    *,
    quantile: bool,
    ensemble_member: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        if any(config.get(key) != value for key, value in FULL_MODEL_CONFIG.items()):
            return False
        if config.get("seed") != seed:
            return False
        if quantile:
            if ensemble_member:
                return bool(
                    checkpoint.get("procedure") == "train_loss"
                    and checkpoint.get("completed") is True
                    and checkpoint.get("epoch") == FULL_MODEL_CONFIG["epochs"]
                    and isinstance(checkpoint.get("best_state"), dict)
                )
            return bool(
                config.get("quantiles") == [0.05, 0.5, 0.95]
                and isinstance(checkpoint.get("model_state_dict"), dict)
            )
        if config.get("quantiles") is not None:
            return False
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            return False
        return True
    except Exception:
        return False


def full_checkpoint_complete(
    path: Path,
    metadata_name: str,
    seed: int,
    *,
    quantile: bool,
    checkpoint_selection: str,
) -> bool:
    if not checkpoint_config_matches(path, seed, quantile=quantile):
        return False
    metadata = load_json(FULL_TRAINING_DIR / metadata_name)
    if not metadata:
        return False
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return bool(
            metadata.get("training_run_complete") is True
            and metadata.get("model_seed") == seed
            and metadata.get("checkpoint_selection") == checkpoint_selection
            and metadata.get("best_epoch") == checkpoint.get("epoch")
            and artifact_matches(metadata, "checkpoint", "checkpoint_sha256")
        )
    except Exception:
        return False


def processed_data_audit_complete(report: dict | None) -> bool:
    if not report:
        return False
    static_missing = report.get("static_missing_basin_counts", {})
    channel_mismatches = report.get("channel_pair_mismatch_counts", {})
    return bool(
        report.get("n_files") == 671
        and report.get("total_rows") == 7107903
        and report.get("observed_targets") == 6943212
        and report.get("missing_targets") == 164691
        and report.get("calibration_rows") == 1225917
        and report.get("calibration_observed_targets") == 1225916
        and report.get("calibration_missing_targets") == 1
        and report.get("test_rows") == 2022394
        and report.get("test_observed_targets") == 2020930
        and report.get("test_missing_targets") == 1464
        and report.get("first_dates") == {"1980-01-01": 671}
        and report.get("last_dates") == {"2008-12-31": 671}
        and not report.get("discontinuous_basins")
        and not report.get("duplicate_date_basins")
        and report.get("max_full_missing_fraction") == 0.0
        and report.get("max_test_missing_fraction") == 0.0
        and report.get("invalid_flow_masks") == 0
        and report.get("observed_nonfinite_targets") == 0
        and report.get("masked_finite_targets") == 0
        and static_missing.get("geol_porostiy") == 3
        and all(
            value == 0
            for key, value in static_missing.items()
            if key != "geol_porostiy"
        )
        and channel_mismatches.get("tmax_maurer__tmin_maurer") == 0
        and channel_mismatches.get("tmax_nldas__tmin_nldas") == 0
        and channel_mismatches.get("tmax_daymet__tmin_daymet", 0) > 0
    )


def stage_data_audit() -> None:
    output = TABLES / "processed_data_audit.json"
    report = load_json(output)
    if processed_data_audit_complete(report):
        print("SKIP processed-data audit (matching report exists)")
        return
    run(
        [
            PYTHON,
            "-u",
            "experiments/analysis/audit_processed_data.py",
            "--processed-dir",
            "data/processed/camels_us",
            "--output",
            str(output),
        ],
        2 * 3600,
        "audit_processed_data.log",
    )


def stage_scarcity() -> None:
    for years in [1, 3, 5, 15]:
        for seed in [42, 123, 456]:
            if scarcity_complete(years, seed):
                print(f"SKIP scarcity {years} yr seed {seed}")
                continue
            run(
                [
                    PYTHON,
                    "-u",
                    "experiments/scarce/train_scarcity_corrected.py",
                    "--years",
                    str(years),
                    "--basin-seed",
                    "42",
                    "--model-seed",
                    str(seed),
                    "--seq-len",
                    "180",
                    "--batch-size",
                    "512",
                    "--epochs",
                    "30",
                ],
                4 * 3600,
                f"scarcity_{years}yr_seed{seed}.log",
            )
    run(
        [PYTHON, "experiments/analysis/analyze_corrected_scarcity.py"],
        2 * 3600,
        "analyze_corrected_scarcity.log",
    )
    run(
        [PYTHON, "experiments/analysis/basin_representativeness.py"],
        3600,
        "basin_representativeness.log",
    )


def stage_scarcity_robustness() -> None:
    run(
        [PYTHON, "experiments/analysis/analyze_conditional_robustness.py"],
        4 * 3600,
        "analyze_conditional_robustness.log",
    )
    run(
        [PYTHON, "experiments/analysis/make_scarcity_replication_subsets.py"],
        3600,
        "make_scarcity_replication_subsets.log",
    )
    run(
        [PYTHON, "-u", "experiments/run_scarcity_robustness.py", "--stage", "all"],
        36 * 3600,
        "run_scarcity_robustness.log",
    )
    run(
        [PYTHON, "experiments/analysis/sanitize_scarcity_checkpoint_metadata.py"],
        2 * 3600,
        "sanitize_scarcity_checkpoint_metadata.log",
    )
    run(
        [PYTHON, "experiments/analysis/analyze_scarcity_robustness_experiments.py"],
        6 * 3600,
        "analyze_scarcity_robustness_experiments.log",
    )


def stage_full_models() -> None:
    point = CHECKPOINTS / "lpu_stream_best.pt"
    if full_checkpoint_complete(
        point,
        "point_seed42.json",
        42,
        quantile=False,
        checkpoint_selection="minimum_validation_mse",
    ):
        print("SKIP full-basin point checkpoint")
    else:
        run(
            [PYTHON, "-u", "experiments/baseline/train_model.py", "--model", "lpu_stream"],
            6 * 3600,
            "train_full_point.log",
        )
        if not full_checkpoint_complete(
            point,
            "point_seed42.json",
            42,
            quantile=False,
            checkpoint_selection="minimum_validation_mse",
        ):
            raise RuntimeError("Full-basin point training did not produce matching completion metadata")

    quantile = CHECKPOINTS / "lpu_stream_quantile_best.pt"
    if full_checkpoint_complete(
        quantile,
        "quantile_seed42.json",
        42,
        quantile=True,
        checkpoint_selection="minimum_training_pinball_loss",
    ):
        print("SKIP full-basin quantile checkpoint")
    else:
        run(
            [PYTHON, "-u", "experiments/uncertainty/train_quantile.py"],
            6 * 3600,
            "train_full_quantile.log",
        )
        if not full_checkpoint_complete(
            quantile,
            "quantile_seed42.json",
            42,
            quantile=True,
            checkpoint_selection="minimum_training_pinball_loss",
        ):
            raise RuntimeError("Full-basin quantile training did not produce matching completion metadata")

    ensemble_paths = {
        seed: CHECKPOINTS / f"ensemble_seed{seed}.pt"
        for seed in [123, 456, 789, 999]
    }
    if all(
        checkpoint_config_matches(
            path,
            seed,
            quantile=True,
            ensemble_member=True,
        )
        for seed, path in ensemble_paths.items()
    ):
        print("SKIP full-basin ensemble checkpoints")
    else:
        run(
            [PYTHON, "-u", "experiments/uncertainty/retrain_ensembles_correct.py"],
            20 * 3600,
            "train_full_ensembles.log",
        )


def stage_full_evaluation() -> None:
    # The evaluator validates checkpoint and normalization hashes before loading
    # either split cache, then recomputes all compact metrics and intervals.
    run(
        [
            PYTHON,
            "-u",
            "experiments/uncertainty/evaluate_corrected_methods.py",
            "--batch-size",
            "1024",
            "--n-mc",
            "50",
            "--bootstrap",
            "1000",
        ],
        8 * 3600,
        "evaluate_corrected_methods.log",
    )


def stage_outputs() -> None:
    run(
        [PYTHON, "experiments/analysis/make_corrected_figures.py"],
        2 * 3600,
        "make_corrected_figures.log",
    )
    if (ROOT / "paper" / "manuscript.tex").exists():
        run(
            [PYTHON, "experiments/analysis/export_manuscript_results.py"],
            3600,
            "export_manuscript_results.log",
        )
    run(
        [PYTHON, "experiments/analysis/verify_manuscript.py"],
        2 * 3600,
        "verify_manuscript.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-full-training",
        action="store_true",
        help="Use existing full-basin checkpoints; still run corrected evaluation.",
    )
    parser.add_argument(
        "--skip-robustness-training",
        action="store_true",
        help="Use existing compact robustness summaries and do not run endpoint sensitivities.",
    )
    args = parser.parse_args()
    stages = [stage_data_audit, stage_scarcity]
    if not args.skip_robustness_training:
        stages.append(stage_scarcity_robustness)
    if not args.skip_full_training:
        stages.append(stage_full_models)
    stages.extend([stage_full_evaluation, stage_outputs])
    for stage in stages:
        print(f"\n=== {stage.__name__} ===", flush=True)
        stage()
    print("\nCorrected pipeline complete.")


if __name__ == "__main__":
    main()
