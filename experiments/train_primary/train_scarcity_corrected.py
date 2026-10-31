"""Run the controlled training-record scarcity experiment and sensitivities.

Protocol v2 changes only the number of supervised training years. Every
condition shares the same basin set, training end date, 180-day input memory,
batch size, calibration targets, test targets, optimizer, and epoch budget.
Normalization is fitted separately from the allowed training basins and dates.
Optional v3 modes repeat endpoint conditions across basin subsets, optimizer
budgets, or a standard-LSTM static-fusion architecture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_preprocessing import (  # noqa: E402
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
)
from src.data.dataset import (  # noqa: E402
    CamelsDataset,
    compute_normalization_stats,
    get_basin_list,
)
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics  # noqa: E402
from src.losses.pinball_loss import PinballLoss  # noqa: E402
from src.models.lpu_stream import LPUStreamModel  # noqa: E402
from src.models.lstm import LSTMModel  # noqa: E402
from src.utils import get_device, seeded_generator, set_seed  # noqa: E402


PROTOCOL_VERSION = "scarcity_v2_fixed_end_fixed_memory"
REPLICATION_PROTOCOL_VERSION = "scarcity_v3_replicated_subsets_fixed_epochs"
EQUAL_UPDATE_PROTOCOL_VERSION = "scarcity_v3_equal_optimizer_updates"
STANDARD_LSTM_PROTOCOL_VERSION = "scarcity_v3_standard_lstm_endpoint_sensitivity"
QUANTILES = [0.05, 0.50, 0.95]
ALPHA = 0.10
TRAIN_STARTS = {
    1: "1994-10-01",
    3: "1992-10-01",
    5: "1990-10-01",
    15: "1980-10-01",
}


def context_start(target_start: str, seq_len: int) -> str:
    date = datetime.strptime(target_start, "%Y-%m-%d") - timedelta(days=seq_len)
    return date.strftime("%Y-%m-%d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_identifier(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{field} must contain only letters, digits, '_' or '-': {value!r}")
    return value


def select_basins(args: argparse.Namespace, all_basins: list[str]) -> tuple[list[str], dict]:
    if args.basin_list_file is None:
        basin_rng = np.random.RandomState(args.basin_seed)
        selected = basin_rng.choice(
            all_basins, min(args.n_basins, len(all_basins)), replace=False
        ).tolist()
        return selected, {
            "method": "random_without_replacement",
            "basin_seed": args.basin_seed,
        }

    if not args.subset_id:
        raise ValueError("--subset-id is required with --basin-list-file")
    subset_id = safe_identifier(args.subset_id, "subset_id")
    path = Path(args.basin_list_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get("subset_id") == subset_id
        ]
    if not rows:
        raise ValueError(f"No rows for subset_id={subset_id!r} in {path}")
    if all(row.get("selection_order") for row in rows):
        rows.sort(key=lambda row: int(row["selection_order"]))
    selected = [row["basin_id"] for row in rows]
    if len(selected) != args.n_basins:
        raise ValueError(
            f"Subset {subset_id!r} contains {len(selected)} basins, expected {args.n_basins}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"Subset {subset_id!r} contains duplicate basin IDs")
    unknown = sorted(set(selected) - set(all_basins))
    if unknown:
        raise ValueError(f"Subset {subset_id!r} contains unknown basins: {unknown[:5]}")
    return selected, {
        "method": "explicit_subset_file",
        "subset_id": subset_id,
        "source": path.relative_to(PROJECT_ROOT).as_posix(),
    }


def make_loader(
    dataset: CamelsDataset,
    batch_size: int,
    shuffle: bool,
    model_seed: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": True,
        "num_workers": num_workers if shuffle else 0,
        "generator": seeded_generator(model_seed),
    }
    if shuffle and num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, **kwargs)


def train_fixed_epochs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        valid_count = 0.0
        started = time.time()
        for dynamic, static, target, mask, _, _ in loader:
            dynamic = dynamic.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(dynamic, static)
            loss = criterion(prediction, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = float(mask.sum().item())
            loss_sum += float(loss.item()) * count
            valid_count += count

        train_loss = loss_sum / max(valid_count, 1.0)
        history.append(train_loss)
        scheduler.step(train_loss)
        print(
            f"Epoch {epoch:02d}/{epochs}: loss={train_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}, time={time.time()-started:.1f}s",
            flush=True,
        )

    return history


def train_fixed_updates(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    rounds: int,
    steps_per_round: int,
    learning_rate: float,
) -> tuple[list[float], int, int]:
    """Train with an identical optimizer-step budget across record lengths."""
    if steps_per_round <= 0:
        raise ValueError("steps_per_round must be positive")

    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    history = []
    optimizer_steps = 0
    observed_targets_seen = 0
    iterator = iter(loader)

    for round_index in range(1, rounds + 1):
        model.train()
        loss_sum = 0.0
        valid_count = 0.0
        started = time.time()
        for _ in range(steps_per_round):
            try:
                dynamic, static, target, mask, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                dynamic, static, target, mask, _, _ = next(iterator)

            dynamic = dynamic.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(dynamic, static)
            loss = criterion(prediction, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count = int(mask.sum().item())
            loss_sum += float(loss.item()) * count
            valid_count += count
            optimizer_steps += 1
            observed_targets_seen += count

        train_loss = loss_sum / max(valid_count, 1.0)
        history.append(train_loss)
        scheduler.step(train_loss)
        print(
            f"Budget round {round_index:02d}/{rounds}: loss={train_loss:.6f}, "
            f"steps={steps_per_round}, lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"time={time.time()-started:.1f}s",
            flush=True,
        )

    return history, optimizer_steps, observed_targets_seen


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    masks = []
    for dynamic, static, target, mask, _, _ in loader:
        output = model(
            dynamic.to(device, non_blocking=True),
            static.to(device, non_blocking=True),
        )
        predictions.append(output.cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets).reshape(-1),
        np.concatenate(masks).reshape(-1),
    )


def inverse_log_scale(values: np.ndarray, stats: dict) -> np.ndarray:
    return values * stats["target_std"] + stats["target_mean"]


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.square(observed - observed.mean()).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(observed - predicted).sum() / denominator)


def per_basin_nse(
    observed_log: np.ndarray,
    predicted_log: np.ndarray,
    basin_ids: np.ndarray,
) -> dict:
    observed = np.maximum(np.expm1(observed_log), 0.0)
    predicted = np.maximum(np.expm1(predicted_log), 0.0)
    values = {}
    for basin_id in np.unique(basin_ids):
        selected = basin_ids == basin_id
        value = nse(observed[selected], predicted[selected])
        if np.isfinite(value):
            values[str(basin_id)] = value

    array = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(20260711)
    bootstrap = np.empty(2000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = np.median(rng.choice(array, size=len(array), replace=True))

    return {
        "n_basins": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "fraction_above_zero": float(np.mean(array > 0)),
        "median_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "by_basin": values,
    }


def basin_relative_coverage(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    basin_ids: np.ndarray,
) -> dict:
    covered = (observed >= lower) & (observed <= upper)
    regime_masks = {"low": [], "normal": [], "high": []}
    for basin_id in np.unique(basin_ids):
        selected = basin_ids == basin_id
        indices = np.flatnonzero(selected)
        q33, q67 = np.quantile(observed[selected], [0.33, 0.67])
        local = observed[selected]
        regime_masks["low"].extend(indices[local <= q33])
        regime_masks["normal"].extend(indices[(local > q33) & (local <= q67)])
        regime_masks["high"].extend(indices[local > q67])

    output = {}
    for name, indices in regime_masks.items():
        index_array = np.asarray(indices, dtype=np.int64)
        output[name] = {
            "picp": float(covered[index_array].mean()),
            "n": int(len(index_array)),
            "mpiw_log": float((upper[index_array] - lower[index_array]).mean()),
        }
    return output


def cluster_bootstrap_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    basin_ids: np.ndarray,
    n_resamples: int = 1000,
) -> dict:
    unique = np.unique(basin_ids)
    alpha = ALPHA
    aggregates = []
    for basin_id in unique:
        selected = basin_ids == basin_id
        lo = lower[selected]
        hi = upper[selected]
        y = observed[selected]
        width = hi - lo
        winkler = (
            width
            + (2 / alpha) * np.maximum(lo - y, 0)
            + (2 / alpha) * np.maximum(y - hi, 0)
        )
        aggregates.append((
            selected.sum(),
            ((y >= lo) & (y <= hi)).sum(),
            width.sum(),
            winkler.sum(),
        ))
    aggregates = np.asarray(aggregates, dtype=np.float64)

    rng = np.random.default_rng(20260711)
    samples = np.empty((n_resamples, 3), dtype=np.float64)
    for index in range(n_resamples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        total = aggregates[chosen].sum(axis=0)
        samples[index] = [total[1] / total[0], total[2] / total[0], total[3] / total[0]]

    names = ["picp", "mpiw_log", "winkler_log"]
    return {
        name: [
            float(np.quantile(samples[:, column], 0.025)),
            float(np.quantile(samples[:, column], 0.975)),
        ]
        for column, name in enumerate(names)
    }


def interval_summary(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    basin_ids: np.ndarray,
    point_prediction: np.ndarray,
) -> dict:
    metrics = compute_uncertainty_metrics(lower, upper, observed, alpha=ALPHA)
    residual_std = np.std(observed - point_prediction)
    metrics["residual_width_ratio"] = float(
        (upper - lower).mean() / (2 * 1.6448536269514722 * residual_std)
    )
    metrics["basin_relative_regimes"] = basin_relative_coverage(
        lower, upper, observed, basin_ids
    )
    metrics["cluster_bootstrap_ci95"] = cluster_bootstrap_metrics(
        lower, upper, observed, basin_ids
    )
    return metrics


def save_predictions(
    path: Path,
    val_prediction_log: np.ndarray,
    val_target_log: np.ndarray,
    val_mask: np.ndarray,
    val_basins: np.ndarray,
    val_dates: np.ndarray,
    test_prediction_log: np.ndarray,
    test_target_log: np.ndarray,
    test_mask: np.ndarray,
    test_basins: np.ndarray,
    test_dates: np.ndarray,
) -> None:
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        val_prediction_log=val_prediction_log,
        val_target_log=val_target_log,
        val_mask=val_mask,
        val_basin_ids=val_basins,
        val_target_dates=val_dates,
        test_prediction_log=test_prediction_log,
        test_target_log=test_target_log,
        test_mask=test_mask,
        test_basin_ids=test_basins,
        test_target_dates=test_dates,
    )
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, required=True, choices=sorted(TRAIN_STARTS))
    parser.add_argument("--n-basins", type=int, default=50)
    parser.add_argument("--basin-seed", type=int, default=42)
    parser.add_argument("--basin-list-file", type=str)
    parser.add_argument("--subset-id", type=str)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--training-budget",
        choices=["fixed_epochs", "fixed_updates"],
        default="fixed_epochs",
    )
    parser.add_argument("--steps-per-round", type=int)
    parser.add_argument("--output-group", default="corrected_scarcity")
    parser.add_argument(
        "--architecture",
        choices=["lpu_stream", "standard_lstm"],
        default="lpu_stream",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_group = safe_identifier(args.output_group, "output_group")
    if args.training_budget == "fixed_updates" and not args.steps_per_round:
        raise ValueError("--steps-per-round is required for --training-budget fixed_updates")
    if args.architecture == "standard_lstm":
        if output_group != "scarcity_standard_lstm":
            raise ValueError("standard_lstm sensitivity must use --output-group scarcity_standard_lstm")
        if args.training_budget != "fixed_epochs" or args.years not in {1, 15}:
            raise ValueError("standard_lstm sensitivity is restricted to fixed-epoch 1/15-year endpoints")

    set_seed(args.model_seed, deterministic=True)
    all_basins = get_basin_list()
    selected_basins, basin_selection = select_basins(args, all_basins)

    if args.basin_list_file:
        tag = f"{args.years}yr_subset{args.subset_id}_modelseed{args.model_seed}"
    else:
        tag = f"{args.years}yr_basinseed{args.basin_seed}_modelseed{args.model_seed}"
    table_dir = PROJECT_ROOT / "results" / "tables" / output_group
    checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints" / output_group
    prediction_dir = PROJECT_ROOT / "results" / "predictions" / output_group
    for directory in (table_dir, checkpoint_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    result_path = table_dir / f"scarcity_{tag}.json"
    checkpoint_path = checkpoint_dir / f"scarcity_{tag}.pt"
    prediction_path = prediction_dir / f"scarcity_{tag}.npz"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"Result already exists: {result_path}; pass --force to replace it")

    train_target_start = TRAIN_STARTS[args.years]
    normalization = compute_normalization_stats(
        selected_basins, train_target_start, TRAIN_END
    )

    dataset_kwargs = {
        "seq_len": args.seq_len,
        "normalization_stats": normalization,
        "preview": True,
    }
    train_dataset = CamelsDataset(
        selected_basins,
        context_start(train_target_start, args.seq_len),
        TRAIN_END,
        target_start_date=train_target_start,
        target_end_date=TRAIN_END,
        **dataset_kwargs,
    )
    val_dataset = CamelsDataset(
        selected_basins,
        context_start(VAL_START, args.seq_len),
        VAL_END,
        target_start_date=VAL_START,
        target_end_date=VAL_END,
        **dataset_kwargs,
    )
    test_dataset = CamelsDataset(
        selected_basins,
        context_start(TEST_START, args.seq_len),
        TEST_END,
        target_start_date=TEST_START,
        target_end_date=TEST_END,
        **dataset_kwargs,
    )

    max_missing = max(
        train_dataset.dynamic_missing_fraction().max(),
        val_dataset.dynamic_missing_fraction().max(),
        test_dataset.dynamic_missing_fraction().max(),
    )
    if max_missing > 0.01:
        raise ValueError(
            f"Dynamic missing fraction {max_missing:.3%} exceeds the 1% protocol limit"
        )

    if args.validate_only:
        print("Protocol validation passed", flush=True)
        print(
            f"  train={len(train_dataset)} windows / "
            f"{train_dataset.valid_target_count()} observed targets",
            flush=True,
        )
        print(f"  calibration={len(val_dataset)} windows", flush=True)
        print(f"  test={len(test_dataset)} windows", flush=True)
        print(f"  max dynamic missing={max_missing:.6f}", flush=True)
        return

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.model_seed, args.num_workers
    )
    val_loader = make_loader(val_dataset, args.batch_size, False, args.model_seed, 0)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.model_seed, 0)

    device = get_device()
    if args.architecture == "lpu_stream":
        model = LPUStreamModel(
            quantiles=QUANTILES,
            enforce_monotonic=True,
        ).to(device)
        model_config = {
            "architecture": "lpu_stream",
            "n_dynamic": 15,
            "n_static": 13,
            "hidden_size": 128,
            "embed_dim": 32,
            "dropout": 0.3,
            "quantiles": QUANTILES,
            "enforce_monotonic": True,
        }
    else:
        model = LSTMModel(
            n_dynamic=15,
            n_static=13,
            hidden_size=128,
            num_layers=1,
            dropout=0.3,
            quantiles=QUANTILES,
            enforce_monotonic=True,
        ).to(device)
        model_config = {
            "architecture": "standard_lstm",
            "n_dynamic": 15,
            "n_static": 13,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.3,
            "static_fusion": "concatenate_standardized_attributes_with_final_hidden_state",
            "quantiles": QUANTILES,
            "enforce_monotonic": True,
        }
    started = time.time()
    if args.training_budget == "fixed_epochs":
        training_loss = train_fixed_epochs(
            model,
            train_loader,
            device,
            args.epochs,
            args.learning_rate,
        )
        optimizer_steps = len(train_loader) * args.epochs
        observed_targets_seen = train_dataset.valid_target_count() * args.epochs
        checkpoint_selection = "fixed_epoch_budget_final_checkpoint"
        if args.architecture == "standard_lstm":
            protocol_version = STANDARD_LSTM_PROTOCOL_VERSION
        else:
            protocol_version = (
                PROTOCOL_VERSION
                if output_group == "corrected_scarcity" and not args.basin_list_file
                else REPLICATION_PROTOCOL_VERSION
            )
    else:
        training_loss, optimizer_steps, observed_targets_seen = train_fixed_updates(
            model,
            train_loader,
            device,
            args.epochs,
            args.steps_per_round,
            args.learning_rate,
        )
        checkpoint_selection = "fixed_optimizer_update_budget_final_checkpoint"
        protocol_version = EQUAL_UPDATE_PROTOCOL_VERSION
    training_seconds = time.time() - started

    val_prediction, val_target, val_mask = predict(model, val_loader, device)
    test_prediction, test_target, test_mask = predict(model, test_loader, device)
    val_prediction_log = inverse_log_scale(val_prediction, normalization)
    val_target_log = inverse_log_scale(val_target, normalization)
    test_prediction_log = inverse_log_scale(test_prediction, normalization)
    test_target_log = inverse_log_scale(test_target, normalization)
    val_basins, val_dates = val_dataset.sample_metadata()
    test_basins, test_dates = test_dataset.sample_metadata()

    val_valid = val_mask > 0
    test_valid = test_mask > 0
    calibrator = CQRCalibrator(alpha=ALPHA)
    q_cal = calibrator.fit(
        val_prediction_log[val_valid, 0],
        val_prediction_log[val_valid, 2],
        val_target_log[val_valid],
    )
    calibrated_lower, calibrated_upper = calibrator.calibrate(
        test_prediction_log[test_valid, 0],
        test_prediction_log[test_valid, 2],
    )

    valid_basins = test_basins[test_valid]
    valid_target = test_target_log[test_valid]
    raw = interval_summary(
        test_prediction_log[test_valid, 0],
        test_prediction_log[test_valid, 2],
        valid_target,
        valid_basins,
        test_prediction_log[test_valid, 1],
    )
    calibrated = interval_summary(
        calibrated_lower,
        calibrated_upper,
        valid_target,
        valid_basins,
        test_prediction_log[test_valid, 1],
    )
    point = per_basin_nse(
        valid_target,
        test_prediction_log[test_valid, 1],
        valid_basins,
    )
    point["pooled_log_nse"] = nse(
        valid_target, test_prediction_log[test_valid, 1]
    )

    checkpoint_temp = checkpoint_path.with_suffix(".tmp.pt")
    checkpoint_args = vars(args).copy()
    if args.basin_list_file:
        checkpoint_args["basin_list_file"] = basin_selection["source"]
    torch.save(
        {
            "protocol_version": protocol_version,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "normalization_stats": normalization,
            "selected_basins": selected_basins,
            "training_loss": training_loss,
            "args": checkpoint_args,
        },
        checkpoint_temp,
    )
    checkpoint_temp.replace(checkpoint_path)
    checkpoint_tensor_hash = tensor_state_sha256(model.state_dict())
    save_predictions(
        prediction_path,
        val_prediction_log,
        val_target_log,
        val_mask,
        val_basins,
        val_dates,
        test_prediction_log,
        test_target_log,
        test_mask,
        test_basins,
        test_dates,
    )

    result = {
        "protocol_version": protocol_version,
        "architecture": args.architecture,
        "years": args.years,
        "n_basins": len(selected_basins),
        "basin_seed": args.basin_seed,
        "subset_id": args.subset_id,
        "basin_selection": basin_selection,
        "model_seed": args.model_seed,
        "selected_basins": selected_basins,
        "train_target_period": [train_target_start, TRAIN_END],
        "calibration_target_period": [VAL_START, VAL_END],
        "test_target_period": [TEST_START, TEST_END],
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "training_budget": args.training_budget,
        "steps_per_round": args.steps_per_round,
        "optimizer_steps": optimizer_steps,
        "observed_targets_seen": observed_targets_seen,
        "checkpoint_selection": checkpoint_selection,
        "normalization_fit_period": [train_target_start, TRAIN_END],
        "normalization_target_count": normalization["target_count"],
        "train_windows": len(train_dataset),
        "train_observed_targets": train_dataset.valid_target_count(),
        "calibration_windows": len(val_dataset),
        "calibration_observed_targets": val_dataset.valid_target_count(),
        "test_windows": len(test_dataset),
        "test_observed_targets": test_dataset.valid_target_count(),
        "max_dynamic_missing_fraction": float(max_missing),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "training_loss": training_loss,
        "q_cal_log": q_cal,
        "point_metrics": point,
        "raw_interval_metrics": raw,
        "calibrated_interval_metrics": calibrated,
        "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_tensor_sha256": checkpoint_tensor_hash,
        "checkpoint_metadata_sanitized": True,
        "predictions": prediction_path.relative_to(PROJECT_ROOT).as_posix(),
        "predictions_sha256": sha256(prediction_path),
        "timestamp": datetime.now().isoformat(),
    }
    atomic_json(result_path, result)

    print("\nControlled scarcity result", flush=True)
    print(f"  Result: {result_path}", flush=True)
    print(f"  Per-basin median NSE: {point['median']:.4f}", flush=True)
    print(f"  Raw PICP: {raw['picp']:.4f}", flush=True)
    print(f"  CQR PICP: {calibrated['picp']:.4f}", flush=True)
    print(f"  q_cal (log scale): {q_cal:.4f}", flush=True)


if __name__ == "__main__":
    main()
