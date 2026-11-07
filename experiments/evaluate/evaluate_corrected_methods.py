"""Evaluate uncertainty methods under one corrected full-basin protocol.

Calibration and test datasets supply one sequence length of antecedent
context while retaining every declared target date. The script evaluates
raw and globally conformalized versions of each interval method and
quantifies uncertainty by resampling whole basins.

The script reuses the archived full-data checkpoints. It does not retrain or
modify them. Cached predictions are keyed by checkpoint hashes and protocol.

Usage:
    python -u experiments/evaluate/evaluate_corrected_methods.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_preprocessing import TEST_END, TEST_START, VAL_END, VAL_START
from src.data.dataset import CamelsDataset, get_basin_list
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.models.lpu_stream import LPUStreamModel
from src.utils import get_device, set_seed


QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
SEQ_LEN = 365
ENSEMBLE_SEEDS = [42, 123, 456, 789, 999]
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions" / "corrected_full_671"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "corrected_full_671"


def checkpoint_path(seed: int) -> Path:
    if seed == 42:
        return CHECKPOINT_DIR / "lpu_stream_quantile_best.pt"
    return CHECKPOINT_DIR / f"ensemble_seed{seed}.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_start(target_start: str) -> str:
    start = pd.Timestamp(target_start) - pd.Timedelta(days=SEQ_LEN)
    return start.strftime("%Y-%m-%d")


def load_quantile_model(seed: int, device: torch.device) -> LPUStreamModel:
    checkpoint = torch.load(checkpoint_path(seed), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = LPUStreamModel(
        n_dynamic=config.get("n_dynamic", 15),
        n_static=config.get("n_static", 13),
        hidden_size=config.get("hidden_size", 128),
        embed_dim=config.get("embed_dim", 32),
        dropout=config.get("dropout", 0.3),
        quantiles=QUANTILES,
    ).to(device)
    state = checkpoint.get("best_state") if checkpoint.get("procedure") == "train_loss" else None
    model.load_state_dict(state or checkpoint["model_state_dict"])
    model.eval()
    return model


def load_point_model(device: torch.device) -> LPUStreamModel:
    path = CHECKPOINT_DIR / "lpu_stream_best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = LPUStreamModel(
        n_dynamic=config.get("n_dynamic", 15),
        n_static=config.get("n_static", 13),
        hidden_size=config.get("hidden_size", 128),
        embed_dim=config.get("embed_dim", 32),
        dropout=config.get("dropout", 0.3),
        quantiles=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_dataset_and_loader(
    basins: list[str],
    target_start: str,
    target_end: str,
    batch_size: int,
) -> tuple[CamelsDataset, DataLoader]:
    dataset = CamelsDataset(
        basins,
        context_start(target_start),
        target_end,
        seq_len=SEQ_LEN,
        target_start_date=target_start,
        target_end_date=target_end,
        preview=True,
    )
    expected = len(dataset.basin_ids) * len(pd.date_range(target_start, target_end, freq="D"))
    if len(dataset) != expected:
        raise RuntimeError(f"Expected {expected:,} samples, found {len(dataset):,}")
    if dataset.dynamic_missing_fraction().max() > 0:
        raise RuntimeError("Corrected evaluation data contain missing dynamic inputs")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict_split(
    loader: DataLoader,
    quantile_models: list[LPUStreamModel],
    point_model: LPUStreamModel,
    device: torch.device,
    n_mc: int,
    split_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    outputs = {
        "target": [],
        "mask": [],
        "basin_index": [],
        "single_qr": [],
        "ensemble": [],
        "mc_dropout": [],
    }
    timings = {"single_qr": 0.0, "ensemble_additional": 0.0, "mc_dropout": 0.0}
    point_model.train()
    warmed_up = False

    for batch_number, (dynamic, static, target, mask, basin_index, _) in enumerate(loader, 1):
        dynamic = dynamic.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)

        if not warmed_up:
            _ = quantile_models[0](dynamic, static)
            for model in quantile_models[1:]:
                _ = model(dynamic, static)
            _ = point_model(dynamic, static)
            synchronize(device)
            set_seed(split_seed)
            point_model.train()
            warmed_up = True

        synchronize(device)
        started = time.perf_counter()
        single = quantile_models[0](dynamic, static)
        synchronize(device)
        timings["single_qr"] += time.perf_counter() - started

        members = [single]
        synchronize(device)
        started = time.perf_counter()
        for model in quantile_models[1:]:
            members.append(model(dynamic, static))
        ensemble = torch.stack(members, dim=0).mean(dim=0)
        synchronize(device)
        timings["ensemble_additional"] += time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        mc_members = [point_model(dynamic, static).squeeze(-1) for _ in range(n_mc)]
        mc_stack = torch.stack(mc_members, dim=0)
        mc_interval = torch.quantile(
            mc_stack,
            torch.tensor(QUANTILES, device=device, dtype=mc_stack.dtype),
            dim=0,
        ).transpose(0, 1)
        synchronize(device)
        timings["mc_dropout"] += time.perf_counter() - started

        outputs["target"].append(target.numpy())
        outputs["mask"].append(mask.numpy())
        outputs["basin_index"].append(basin_index.numpy())
        outputs["single_qr"].append(single.cpu().numpy())
        outputs["ensemble"].append(ensemble.cpu().numpy())
        outputs["mc_dropout"].append(mc_interval.cpu().numpy())

        if batch_number % 100 == 0 or batch_number == len(loader):
            print(f"  batch {batch_number:,}/{len(loader):,}", flush=True)

    point_model.eval()
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}
    arrays["target"] = arrays["target"].reshape(-1).astype(np.float32)
    arrays["mask"] = arrays["mask"].reshape(-1).astype(np.uint8)
    arrays["basin_index"] = arrays["basin_index"].reshape(-1).astype(np.int16)
    for name in ("single_qr", "ensemble", "mc_dropout"):
        arrays[name] = arrays[name].astype(np.float32)
    return arrays, timings


def protocol_identifier(
    n_mc: int,
    batch_size: int,
    checkpoint_hashes: dict[str, str],
    normalization_hash: str,
) -> str:
    payload = {
        "seq_len": SEQ_LEN,
        "val": [VAL_START, VAL_END],
        "test": [TEST_START, TEST_END],
        "n_mc": n_mc,
        "batch_size": batch_size,
        "checkpoints": checkpoint_hashes,
        "normalization_stats": normalization_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def prediction_cache_path(split: str) -> Path:
    return PREDICTION_DIR / f"{split}_predictions.npz"


def load_prediction_cache(split: str, protocol_id: str):
    path = prediction_cache_path(split)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        cached_id = str(cache["protocol_id"].item())
        if cached_id != protocol_id:
            return None
        arrays = {
            name: cache[name]
            for name in ("target", "mask", "basin_index", "single_qr", "ensemble", "mc_dropout")
        }
        timings = json.loads(str(cache["timings_json"].item()))
    print(f"Loaded matching {split} prediction cache: {path}")
    return arrays, timings


def save_prediction_cache(
    split: str,
    protocol_id: str,
    arrays: dict[str, np.ndarray],
    timings: dict[str, float],
) -> None:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_cache_path(split),
        protocol_id=np.asarray(protocol_id),
        timings_json=np.asarray(json.dumps(timings, sort_keys=True)),
        **arrays,
    )


def target_dates(dataset: CamelsDataset) -> np.ndarray:
    dates = np.empty(len(dataset.indices), dtype="datetime64[D]")
    for sample, (basin_index, start) in enumerate(dataset.indices):
        dates[sample] = dataset.basin_data[basin_index]["dates"][start + dataset.seq_len]
    return dates


def pooled_nse(median: np.ndarray, target: np.ndarray) -> float:
    denominator = np.square(target - target.mean()).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(target - median).sum() / denominator)


def destandardize_log_scale(arrays: dict[str, np.ndarray], stats: dict) -> dict[str, np.ndarray]:
    output = {
        "mask": arrays["mask"],
        "basin_index": arrays["basin_index"],
        "target": arrays["target"] * stats["target_std"] + stats["target_mean"],
    }
    for method in ("single_qr", "ensemble", "mc_dropout"):
        output[method] = arrays[method] * stats["target_std"] + stats["target_mean"]
    return output


def basin_nse_values(
    median: np.ndarray,
    target: np.ndarray,
    basin_index: np.ndarray,
    n_basins: int,
) -> np.ndarray:
    values = np.full(n_basins, np.nan, dtype=np.float64)
    for index in range(n_basins):
        selected = basin_index == index
        if selected.sum() < 2:
            continue
        denominator = np.square(target[selected] - target[selected].mean()).sum()
        if denominator > 0:
            values[index] = 1.0 - np.square(target[selected] - median[selected]).sum() / denominator
    return values


def evaluate_variant(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    basin_index: np.ndarray,
    n_basins: int,
) -> dict:
    valid = mask > 0
    y = target[valid]
    med = median[valid]
    lo = lower[valid]
    hi = upper[valid]
    basins = basin_index[valid]
    uncertainty = compute_uncertainty_metrics(lo, hi, y, alpha=ALPHA)
    physical_y = np.maximum(np.expm1(y), 0.0)
    physical_median = np.maximum(np.expm1(med), 0.0)
    basin_nse = basin_nse_values(physical_median, physical_y, basins, n_basins)
    return {
        "n_valid": int(valid.sum()),
        "pooled_log_nse": pooled_nse(med, y),
        "median_basin_nse": float(np.nanmedian(basin_nse)),
        **uncertainty,
    }


def per_basin_sufficient(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    basin_index: np.ndarray,
    n_basins: int,
) -> dict[str, np.ndarray]:
    valid = mask > 0
    y = target[valid].astype(np.float64)
    med = median[valid].astype(np.float64)
    lo = lower[valid].astype(np.float64)
    hi = upper[valid].astype(np.float64)
    basins = basin_index[valid].astype(np.int64)
    width = hi - lo
    winkler = width + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - hi, 0)

    def sums(weights=None):
        return np.bincount(basins, weights=weights, minlength=n_basins).astype(np.float64)

    count = sums()
    y_sum = sums(y)
    y2_sum = sums(y * y)
    sse = sums(np.square(y - med))
    physical_y = np.maximum(np.expm1(y), 0.0)
    physical_median = np.maximum(np.expm1(med), 0.0)
    physical_y_sum = sums(physical_y)
    physical_y2_sum = sums(physical_y * physical_y)
    physical_sse = sums(np.square(physical_y - physical_median))
    denominator = physical_y2_sum - np.divide(
        physical_y_sum * physical_y_sum,
        count,
        out=np.zeros_like(count),
        where=count > 0,
    )
    basin_nse = np.full(n_basins, np.nan, dtype=np.float64)
    usable = denominator > 0
    basin_nse[usable] = 1.0 - physical_sse[usable] / denominator[usable]
    return {
        "count": count,
        "y_sum": y_sum,
        "y2_sum": y2_sum,
        "sse": sse,
        "covered": sums(((y >= lo) & (y <= hi)).astype(np.float64)),
        "width": sums(width),
        "winkler": sums(winkler),
        "crossing": sums((lo > hi).astype(np.float64)),
        "basin_nse": basin_nse,
    }


def bootstrap_metrics(
    variants: dict[str, dict[str, np.ndarray]],
    n_basins: int,
    repetitions: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_basins, size=(repetitions, n_basins))
    boot = {}
    summaries = {}

    for name, sufficient in variants.items():
        count = sufficient["count"][draws].sum(axis=1)
        y_sum = sufficient["y_sum"][draws].sum(axis=1)
        y2_sum = sufficient["y2_sum"][draws].sum(axis=1)
        sse = sufficient["sse"][draws].sum(axis=1)
        denominator = y2_sum - y_sum * y_sum / count
        values = {
            "pooled_log_nse": 1.0 - sse / denominator,
            "median_basin_nse": np.nanmedian(sufficient["basin_nse"][draws], axis=1),
            "picp": sufficient["covered"][draws].sum(axis=1) / count,
            "mpiw": sufficient["width"][draws].sum(axis=1) / count,
            "winkler_score": sufficient["winkler"][draws].sum(axis=1) / count,
            "crossing_rate": sufficient["crossing"][draws].sum(axis=1) / count,
        }
        boot[name] = values
        summaries[name] = {
            metric: {
                "low": float(np.percentile(samples, 2.5)),
                "high": float(np.percentile(samples, 97.5)),
            }
            for metric, samples in values.items()
        }

    pairs = {
        "mc_dropout_cqr_minus_raw": ("mc_dropout_cqr", "mc_dropout_raw"),
        "single_qr_cqr_minus_raw": ("single_qr_cqr", "single_qr_raw"),
        "ensemble_cqr_minus_raw": ("ensemble_cqr", "ensemble_raw"),
        "single_qr_cqr_minus_mc_dropout_cqr": ("single_qr_cqr", "mc_dropout_cqr"),
        "ensemble_cqr_minus_single_qr_cqr": ("ensemble_cqr", "single_qr_cqr"),
    }
    comparisons = {}
    for label, (left, right) in pairs.items():
        comparisons[label] = {}
        for metric in boot[left]:
            difference = boot[left][metric] - boot[right][metric]
            comparisons[label][metric] = {
                "low": float(np.percentile(difference, 2.5)),
                "high": float(np.percentile(difference, 97.5)),
            }
    return summaries, comparisons


def load_aridity(basin_ids: list[str]) -> np.ndarray:
    path = PROJECT_ROOT / "data" / "raw" / "camels_us" / "camels_clim.txt"
    frame = pd.read_csv(path, sep=";", dtype={"gauge_id": str})
    mapping = frame.set_index("gauge_id")["aridity"].to_dict()
    values = np.asarray([mapping.get(basin, np.nan) for basin in basin_ids], dtype=np.float64)
    if not np.isfinite(values).all():
        missing = [basin for basin, value in zip(basin_ids, values) if not np.isfinite(value)]
        raise RuntimeError(f"Missing aridity for {len(missing)} basins")
    return values


def climate_group_results(
    arrays: dict[str, np.ndarray],
    basin_ids: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[dict, list[dict]]:
    aridity = load_aridity(basin_ids)
    thresholds = np.quantile(aridity, [0.25, 0.50, 0.75])
    groups = np.digitize(aridity, thresholds, right=True)
    rows = []
    labels = ["Q1 most humid", "Q2 humid-intermediate", "Q3 dry-intermediate", "Q4 driest"]
    for group, label in enumerate(labels):
        basin_selection = np.flatnonzero(groups == group)
        sample_selection = np.isin(arrays["basin_index"], basin_selection)
        metrics = evaluate_variant(
            arrays["single_qr"][:, 1][sample_selection],
            lower[sample_selection],
            upper[sample_selection],
            arrays["target"][sample_selection],
            arrays["mask"][sample_selection],
            arrays["basin_index"][sample_selection],
            len(basin_ids),
        )
        rows.append({
            "group": label,
            "n_basins": int(len(basin_selection)),
            "aridity_min": float(aridity[basin_selection].min()),
            "aridity_max": float(aridity[basin_selection].max()),
            **metrics,
        })
    all_metrics = evaluate_variant(
        arrays["single_qr"][:, 1], lower, upper,
        arrays["target"], arrays["mask"], arrays["basin_index"], len(basin_ids),
    )
    rows.append({
        "group": "All CAMELS-US",
        "n_basins": len(basin_ids),
        "aridity_min": float(aridity.min()),
        "aridity_max": float(aridity.max()),
        **all_metrics,
    })
    metadata = {
        "definition": "Mutually exclusive quartiles over basin-level CAMELS aridity",
        "thresholds": [float(value) for value in thresholds],
        "calibration": "One global CQR adjustment fitted on all calibration samples",
    }
    return metadata, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-mc", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(20260711)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    checkpoint_paths = {
        "point_seed42": CHECKPOINT_DIR / "lpu_stream_best.pt",
        **{f"quantile_seed{seed}": checkpoint_path(seed) for seed in ENSEMBLE_SEEDS},
    }
    for path in checkpoint_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_hashes = {name: sha256(path) for name, path in checkpoint_paths.items()}
    normalization_path = PROJECT_ROOT / "data" / "metadata" / "normalization_stats.json"
    normalization_hash = sha256(normalization_path)
    protocol_id = protocol_identifier(
        args.n_mc,
        args.batch_size,
        checkpoint_hashes,
        normalization_hash,
    )

    basins = get_basin_list()
    if len(basins) != 671:
        raise RuntimeError(f"Expected 671 basins, found {len(basins)}")
    quantile_models = [load_quantile_model(seed, device) for seed in ENSEMBLE_SEEDS]
    point_model = load_point_model(device)

    split_configuration = {
        "calibration": (VAL_START, VAL_END, 20260712),
        "test": (TEST_START, TEST_END, 20260713),
    }
    predictions = {}
    split_timings = {}
    datasets = {}
    for split, (start, end, split_seed) in split_configuration.items():
        cached = None if args.force else load_prediction_cache(split, protocol_id)
        dataset, loader = make_dataset_and_loader(basins, start, end, args.batch_size)
        datasets[split] = dataset
        if cached is None:
            print(f"\nPredicting {split}: {start} to {end} ({len(dataset):,} samples)")
            arrays, timings = predict_split(
                loader, quantile_models, point_model, device, args.n_mc, split_seed,
            )
            save_prediction_cache(split, protocol_id, arrays, timings)
        else:
            arrays, timings = cached
        if len(arrays["target"]) != len(dataset):
            raise RuntimeError(f"{split} cache length does not match the current dataset")
        predictions[split] = arrays
        split_timings[split] = timings

    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    calibration = destandardize_log_scale(predictions["calibration"], normalization)
    test = destandardize_log_scale(predictions["test"], normalization)
    if datasets["calibration"].basin_ids != datasets["test"].basin_ids:
        raise RuntimeError("Calibration and test basin order differs")
    basin_ids = datasets["test"].basin_ids

    q_cal = {}
    variants = {}
    for method in ("mc_dropout", "single_qr", "ensemble"):
        valid_cal = calibration["mask"] > 0
        calibrator = CQRCalibrator(alpha=ALPHA)
        q_cal[method] = calibrator.fit(
            calibration[method][valid_cal, 0],
            calibration[method][valid_cal, 2],
            calibration["target"][valid_cal],
        )
        raw_lower = test[method][:, 0]
        raw_upper = test[method][:, 2]
        cal_lower, cal_upper = calibrator.calibrate(raw_lower, raw_upper)
        variants[f"{method}_raw"] = {
            "median": test[method][:, 1], "lower": raw_lower, "upper": raw_upper,
        }
        variants[f"{method}_cqr"] = {
            "median": test[method][:, 1], "lower": cal_lower, "upper": cal_upper,
        }

    metrics = {}
    sufficient = {}
    for name, prediction in variants.items():
        metrics[name] = evaluate_variant(
            prediction["median"], prediction["lower"], prediction["upper"],
            test["target"], test["mask"], test["basin_index"], len(basin_ids),
        )
        sufficient[name] = per_basin_sufficient(
            prediction["median"], prediction["lower"], prediction["upper"],
            test["target"], test["mask"], test["basin_index"], len(basin_ids),
        )

    confidence_intervals, difference_intervals = bootstrap_metrics(
        sufficient, len(basin_ids), args.bootstrap, seed=20260714,
    )

    test_costs = {
        "single_qr": split_timings["test"]["single_qr"],
        "ensemble": (
            split_timings["test"]["single_qr"]
            + split_timings["test"]["ensemble_additional"]
        ),
        "mc_dropout": split_timings["test"]["mc_dropout"],
    }
    forward_passes = {"single_qr": 1, "ensemble": 5, "mc_dropout": args.n_mc}
    for name in metrics:
        method = name.rsplit("_", 1)[0]
        metrics[name]["forward_passes_per_sample"] = forward_passes[method]
        metrics[name]["test_forward_seconds"] = float(test_costs[method])
        metrics[name]["relative_forward_time"] = float(test_costs[method] / test_costs["single_qr"])

    single_lower = variants["single_qr_cqr"]["lower"]
    single_upper = variants["single_qr_cqr"]["upper"]
    climate_metadata, climate_rows = climate_group_results(
        test, basin_ids, single_lower, single_upper,
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "corrected_full_671_uncertainty_comparison",
        "protocol_id": protocol_id,
        "protocol": {
            "n_basins": len(basin_ids),
            "sequence_length_days": SEQ_LEN,
            "calibration_targets": [VAL_START, VAL_END],
            "test_targets": [TEST_START, TEST_END],
            "target_coverage": 1 - ALPHA,
            "calibration_windows": int(len(calibration["mask"])),
            "calibration_observed_targets": int((calibration["mask"] > 0).sum()),
            "test_windows": int(len(test["mask"])),
            "test_observed_targets": int((test["mask"] > 0).sum()),
            "normalization": (
                "Inputs use stored statistics fitted on all 671 basins during 1980-10-01 to 1995-09-30; "
                "predictions and targets are de-standardized before evaluation"
            ),
            "aggregation": (
                "PICP, MPIW, and Winkler are pooled on the log1p(mm/day) scale; pooled NSE uses the same "
                "log scale; median basin NSE is computed after inversion to nonnegative physical flow"
            ),
            "bootstrap": f"{args.bootstrap} paired whole-basin resamples",
            "temperature_channel_note": (
                "CAMELS source files duplicate daily temperature in the Tmax and Tmin columns "
                "for Maurer and NLDAS; these supplied channels are retained for checkpoint compatibility"
            ),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "normalization_stats_sha256": normalization_hash,
        "prediction_sha256": {
            split: sha256(prediction_cache_path(split)) for split in split_configuration
        },
        "q_cal": {name: float(value) for name, value in q_cal.items()},
        "timings": split_timings,
        "metrics": metrics,
        "cluster_bootstrap_95ci": confidence_intervals,
        "paired_difference_95ci": difference_intervals,
        "climate_groups": {"metadata": climate_metadata, "rows": climate_rows},
    }
    with (TABLE_DIR / "method_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    metric_rows = []
    for name, values in metrics.items():
        row = {"method": name, **values}
        for metric, interval in confidence_intervals[name].items():
            row[f"{metric}_ci_low"] = interval["low"]
            row[f"{metric}_ci_high"] = interval["high"]
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(TABLE_DIR / "method_comparison.csv", index=False)
    pd.DataFrame(climate_rows).to_csv(TABLE_DIR / "climate_quartiles.csv", index=False)

    np.savez_compressed(
        PREDICTION_DIR / "evaluation_index.npz",
        basin_ids=np.asarray(basin_ids),
        calibration_dates=target_dates(datasets["calibration"]),
        test_dates=target_dates(datasets["test"]),
    )

    print("\nCorrected full-basin comparison")
    print("method                    NSE-med    PICP    MPIW  Winkler  cross  passes")
    for name, values in metrics.items():
        print(
            f"{name:<26}{values['median_basin_nse']:>7.3f}  {values['picp']:>7.3f}"
            f" {values['mpiw']:>7.3f} {values['winkler_score']:>8.3f}"
            f" {values['crossing_rate']:>6.3f} {values['forward_passes_per_sample']:>7d}"
        )
    print(f"Saved results to {TABLE_DIR}")


if __name__ == "__main__":
    main()
