"""Run the 12 primary scarcity fits (4 training lengths x 3 model seeds).

Each fit is launched as a subprocess of train_scarcity_corrected.py with
the declared primary protocol arguments. A fit whose recorded artifacts
already validate (protocol identity plus checkpoint/prediction SHA-256)
is skipped, so interrupted runs can resume where they stopped.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import repository_path, sha256_file  # noqa: E402

TRAIN_SCRIPT = ROOT / "experiments" / "train_primary" / "train_scarcity_corrected.py"
MODEL_SEEDS = [42, 123, 456]
YEARS = [1, 3, 5, 15]
PROTOCOL = "scarcity_v2_fixed_end_fixed_memory"


def result_path(years: int, model_seed: int) -> Path:
    return (
        ROOT
        / "results"
        / "tables"
        / "corrected_scarcity"
        / f"scarcity_{years}yr_basinseed42_modelseed{model_seed}.json"
    )


def result_complete(years: int, model_seed: int) -> bool:
    path = result_path(years, model_seed)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if any([
            result.get("protocol_version") != PROTOCOL,
            result.get("years") != years,
            result.get("model_seed") != model_seed,
            result.get("basin_seed") != 42,
            result.get("seq_len") != 180,
            result.get("batch_size") != 512,
            result.get("epochs") != 30,
        ]):
            return False
        for path_key, hash_key in [
            ("checkpoint", "checkpoint_sha256"),
            ("predictions", "predictions_sha256"),
        ]:
            artifact = repository_path(ROOT, result[path_key])
            if not artifact.is_file() or sha256_file(artifact) != result[hash_key]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    force = "--force" in sys.argv
    for years in YEARS:
        for model_seed in MODEL_SEEDS:
            expected = result_path(years, model_seed)
            if result_complete(years, model_seed) and not force:
                print(f"SKIP completed {expected.name}", flush=True)
                continue
            if expected.exists() and not force:
                raise RuntimeError(f"Incomplete or mismatched result: {expected}; rerun with --force")
            log = ROOT / "results" / "logs" / "corrected_scarcity" / f"{years}yr_seed{model_seed}.log"
            command = [
                sys.executable, str(TRAIN_SCRIPT),
                "--years", str(years),
                "--model-seed", str(model_seed),
            ]
            if force:
                command.append("--force")
            run_logged(command, log)


if __name__ == "__main__":
    main()
