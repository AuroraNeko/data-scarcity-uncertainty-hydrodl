"""Single entry point that reproduces every number, table, and figure of the
paper in dependency order.

The stage table below is the only place where execution order lives. Each
entry is a plain script with one job; this file adds no science logic, it only
runs the scripts in order, checks that each exists, and stops at the first
failure.

Usage:
    python run_pipeline.py --list              # show stages and their scripts
    python run_pipeline.py --dry-run           # resolve every script path, run nothing
    python run_pipeline.py                     # run all stages in order
    python run_pipeline.py --from 6            # run from stage 6 onward
    python run_pipeline.py --stage 6           # run only stage 6
    python run_pipeline.py --only coverage_groups.py   # run one script

Stages 2, 3 and 5 retrain models and need a CUDA GPU (RTX-class, >=8 GB) plus
the preprocessed CAMELS-US archive under data/. Stages 1, 4, 7, 8 and 9 are
table-driven and run on CPU. Stage 6 reloads trained checkpoints for inference
(method evaluation, dense tau export, ungauged transfer); every script in it
except export_dense_taus.py, whose timing suite needs CUDA, also completes on
CPU, though much faster on GPU. Runtime for the full rerun is dominated by
training (several GPU-days for all endpoints and seeds).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES: dict[int, tuple[str, list[str]]] = {
    1: ("prepare: build and verify the training cache", [
        "experiments/prepare/build_training_cache.py",
        "experiments/prepare/verify_training_cache.py",
    ]),
    2: ("train-primary: 12 main fits, disjoint endpoint replication, equal-update and standard-LSTM sensitivities", [
        "experiments/train_primary/make_replication_subsets.py",
        "experiments/train_primary/run_primary_fits.py",
        "experiments/train_primary/run_endpoint_replications.py",
    ]),
    3: ("train-benchmark: point and uncertainty baselines on all 671 basins", [
        "experiments/train_benchmark/prepare_gpu.py",
        "experiments/train_benchmark/train_point_model.py",
        "experiments/train_benchmark/train_single_qr.py",
        "experiments/train_benchmark/retrain_ensemble_corrected.py",
        "experiments/train_benchmark/train_mdn.py",
        "experiments/train_benchmark/train_gaussian_quantile.py",
        "experiments/train_benchmark/train_mc_dropout_noise.py",
        "experiments/train_benchmark/train_point_reselect.py",
    ]),
    4: ("correct: median-bias-fix diagnostics on the archived scarcity runs", [
        "experiments/bias_correction/apply_bias_fix.py",
    ]),
    5: ("train-sensitivity: 48 endpoint fits (alternative years, validation, multi-level, TCN, memory)", [
        "experiments/train_sensitivity/run_queue.py",
        "experiments/train_sensitivity/verify_chunked_recurrence.py",
    ]),
    6: ("evaluate: corrected-method comparison, memberwise split-CQR, dense tau grids, coverage groups, ungauged transfer", [
        "experiments/evaluate/evaluate_corrected_methods.py",
        "experiments/train_benchmark/export_dense_taus.py",
        "experiments/evaluate/memberwise_splitcqr.py",
        "experiments/evaluate/coverage_groups.py",
        "experiments/evaluate/distribution_scores.py",
        "experiments/evaluate/method_comparison.py",
        "experiments/evaluate/calibration_variants.py",
        "experiments/evaluate/transfer_unseen_basins.py",
        "experiments/evaluate/unseen_basins.py",
        "experiments/evaluate/calibration_blocks.py",
        "experiments/evaluate/hydrology_context.py",
        "experiments/evaluate/selection_simulation.py",
    ]),
    7: ("audit: data-chain and checkpoint hygiene checks", [
        "experiments/prepare/audit_processed_data.py",
        "experiments/prepare/audit_checkpoint_metadata.py",
    ]),
    8: ("analyze: training summaries, paired effects, representativeness, learning curves", [
        "experiments/analyze/training_summary.py",
        "experiments/analyze/summarize_scarcity.py",
        "experiments/analyze/summarize_replications.py",
        "experiments/analyze/conditional_robustness.py",
        "experiments/analyze/basin_representativeness.py",
        "experiments/analyze/learning_curves.py",
    ]),
    9: ("figures: all paper figures from the frozen tables", [
        "experiments/figures/make_main_figures.py",
        "experiments/figures/make_manuscript_figures.py",
    ]),
}


def resolve(script: str) -> Path:
    return ROOT / script


def match_by_basename(target: str) -> str | None:
    """Resolve a bare filename against the stage table, or None if not unique."""
    matches = sorted({s for _title, scripts in STAGES.values() for s in scripts
                      if Path(s).name == Path(target).name})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"AMBIGUOUS {target}: matches {', '.join(matches)}")
    return None


def run_script(script: str, dry: bool) -> bool:
    path = resolve(script)
    if not path.is_file():
        print(f"MISSING {script}", file=sys.stderr)
        return False
    print(f"RUN {script}", flush=True)
    if dry:
        return True
    start = time.time()
    code = subprocess.call([sys.executable, "-u", str(path)], cwd=ROOT)
    print(f"{'OK ' if code == 0 else 'FAIL'} {script} ({time.time() - start:.1f}s)", flush=True)
    return code == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", type=int, help="run only this stage number")
    group.add_argument("--from", dest="start", type=int, help="run from this stage onward")
    group.add_argument("--only", help="run a single script by filename")
    parser.add_argument("--list", action="store_true", help="list stages and scripts")
    parser.add_argument("--dry-run", action="store_true", help="resolve paths without running")
    args = parser.parse_args()

    if args.list:
        for number, (title, scripts) in STAGES.items():
            print(f"stage {number}: {title}")
            for s in scripts:
                print(f"    {s}")
        return 0

    if args.only:
        target = args.only
        if not resolve(target).is_file():
            target = match_by_basename(target) or target
        return 0 if run_script(target, args.dry_run) else 1

    numbers = sorted(STAGES)
    if args.stage is not None and args.stage not in STAGES:
        print(f"--stage must be one of {numbers}", file=sys.stderr)
        return 2
    if args.start is not None and not numbers[0] <= args.start <= numbers[-1]:
        print(f"--from must be between {numbers[0]} and {numbers[-1]}", file=sys.stderr)
        return 2
    todo = STAGES if args.stage is None and args.start is None else {
        n: v for n, v in STAGES.items()
        if (args.stage is None or n == args.stage) and (args.start is None or n >= args.start)
    }
    for number, (title, scripts) in todo.items():
        print(f"\n=== stage {number}: {title} ===", flush=True)
        for s in scripts:
            if not run_script(s, args.dry_run):
                return 1
    print("\nAll requested stages completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
