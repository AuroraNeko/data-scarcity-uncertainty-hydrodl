"""Run endpoint subset replications and equal-update sensitivity experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "experiments" / "train_primary" / "train_scarcity_corrected.py"
SUBSET_FILE = ROOT / "results" / "tables" / "scarcity_replications" / "replication_subsets.csv"
MODEL_SEEDS = [42, 123, 456]
ENDPOINT_YEARS = [1, 15]
REPLICATION_SUBSETS = ["stratified_1", "stratified_2", "stratified_3", "stratified_4"]
PROTOCOLS = {
    "scarcity_replications": "scarcity_v3_replicated_subsets_fixed_epochs",
    "scarcity_equal_updates": "scarcity_v3_equal_optimizer_updates",
    "scarcity_standard_lstm": "scarcity_v3_standard_lstm_endpoint_sensitivity",
}


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


def result_complete(
    path: Path,
    group: str,
    years: int,
    subset_id: str,
    model_seed: int,
) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if any([
            result.get("protocol_version") != PROTOCOLS[group],
            result.get("years") != years,
            result.get("subset_id") != subset_id,
            result.get("model_seed") != model_seed,
            result.get("seq_len") != 180,
            result.get("batch_size") != 512,
            result.get("epochs") != 30,
        ]):
            return False
        if group == "scarcity_equal_updates" and (
            result.get("training_budget") != "fixed_updates"
            or result.get("optimizer_steps") != 1080
        ):
            return False
        if group != "scarcity_equal_updates" and result.get("training_budget") != "fixed_epochs":
            return False
        if group == "scarcity_standard_lstm" and result.get("architecture") != "standard_lstm":
            return False
        for path_key, hash_key in [
            ("checkpoint", "checkpoint_sha256"),
            ("predictions", "predictions_sha256"),
        ]:
            artifact = repository_path(result[path_key])
            if not artifact.is_file() or sha256(artifact) != result[hash_key]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def result_path(group: str, years: int, subset_id: str, model_seed: int) -> Path:
    return (
        ROOT
        / "results"
        / "tables"
        / group
        / f"scarcity_{years}yr_subset{subset_id}_modelseed{model_seed}.json"
    )


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def train_command(
    years: int,
    subset_id: str,
    model_seed: int,
    output_group: str,
    training_budget: str,
    force: bool,
    architecture: str = "lpu_stream",
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--years",
        str(years),
        "--n-basins",
        "50",
        "--basin-list-file",
        SUBSET_FILE.relative_to(ROOT).as_posix(),
        "--subset-id",
        subset_id,
        "--model-seed",
        str(model_seed),
        "--seq-len",
        "180",
        "--batch-size",
        "512",
        "--epochs",
        "30",
        "--output-group",
        output_group,
        "--training-budget",
        training_budget,
        "--architecture",
        architecture,
    ]
    if training_budget == "fixed_updates":
        command.extend(["--steps-per-round", "36"])
    if force:
        command.append("--force")
    return command


def run_replications(force: bool) -> None:
    group = "scarcity_replications"
    for subset_id in REPLICATION_SUBSETS:
        for years in ENDPOINT_YEARS:
            for model_seed in MODEL_SEEDS:
                expected = result_path(group, years, subset_id, model_seed)
                if result_complete(expected, group, years, subset_id, model_seed) and not force:
                    print(f"SKIP completed {expected.name}", flush=True)
                    continue
                if expected.exists() and not force:
                    raise RuntimeError(f"Incomplete or mismatched result: {expected}; rerun with --force")
                log = (
                    ROOT
                    / "results"
                    / "logs"
                    / group
                    / f"{years}yr_{subset_id}_seed{model_seed}.log"
                )
                run_logged(
                    train_command(
                        years,
                        subset_id,
                        model_seed,
                        group,
                        "fixed_epochs",
                        force,
                    ),
                    log,
                )


def run_equal_updates(force: bool) -> None:
    group = "scarcity_equal_updates"
    subset_id = "primary_seed42"
    for years in ENDPOINT_YEARS:
        for model_seed in MODEL_SEEDS:
            expected = result_path(group, years, subset_id, model_seed)
            if result_complete(expected, group, years, subset_id, model_seed) and not force:
                print(f"SKIP completed {expected.name}", flush=True)
                continue
            if expected.exists() and not force:
                raise RuntimeError(f"Incomplete or mismatched result: {expected}; rerun with --force")
            log = (
                ROOT
                / "results"
                / "logs"
                / group
                / f"{years}yr_{subset_id}_seed{model_seed}.log"
            )
            run_logged(
                train_command(
                    years,
                    subset_id,
                    model_seed,
                    group,
                    "fixed_updates",
                    force,
                ),
                log,
            )


def run_architecture_sensitivity(force: bool) -> None:
    group = "scarcity_standard_lstm"
    subset_id = "primary_seed42"
    for years in ENDPOINT_YEARS:
        for model_seed in MODEL_SEEDS:
            expected = result_path(group, years, subset_id, model_seed)
            if result_complete(expected, group, years, subset_id, model_seed) and not force:
                print(f"SKIP completed {expected.name}", flush=True)
                continue
            if expected.exists() and not force:
                raise RuntimeError(f"Incomplete or mismatched result: {expected}; rerun with --force")
            log = (
                ROOT
                / "results"
                / "logs"
                / group
                / f"{years}yr_{subset_id}_seed{model_seed}.log"
            )
            run_logged(
                train_command(
                    years,
                    subset_id,
                    model_seed,
                    group,
                    "fixed_epochs",
                    force,
                    architecture="standard_lstm",
                ),
                log,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "replications", "equal-updates", "architecture"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not SUBSET_FILE.exists():
        raise FileNotFoundError(
            f"Run experiments/train_primary/make_replication_subsets.py first: {SUBSET_FILE}"
        )
    if args.stage in {"all", "replications"}:
        run_replications(args.force)
    if args.stage in {"all", "equal-updates"}:
        run_equal_updates(args.force)
    if args.stage in {"all", "architecture"}:
        run_architecture_sensitivity(args.force)


if __name__ == "__main__":
    main()
