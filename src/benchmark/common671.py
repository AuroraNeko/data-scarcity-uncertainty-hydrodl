"""Shared training and evaluation infrastructure for the 671-basin baselines.

Protocol mirrors the archived full-basin comparison: the train window uses
the original dataset path (365-day warm-up), calibration and test use the
corrected context-supplying datasets, batch 1024, Adam 1e-3, up to 30 epochs
with early stopping (patience 5), ReduceLROnPlateau(0.5, patience 3) on
training loss, best state = minimum training loss. All splits are held
GPU-resident (see gpu_loader) so the training stream never drains between
batches. Predictions are de-standardized with the stored global target
statistics before CQR and interval scoring.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# cudnn autotuning disabled on purpose: on this 8 GB WDDM card the
# benchmarking pass reserves multi-GB transient workspaces, the allocator
# cache keeps them, and once desktop contexts fill the remaining VRAM every
# big tensor spills to system RAM (~15-40x slowdown, observed 190->434 s
# per 300 batches mid-epoch). Fixed shapes make autotuning worthless anyway.
torch.backends.cudnn.benchmark = False

from src.data.data_preprocessing import TRAIN_END, TRAIN_START  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.corrections.common import ALPHA, conformal_radius  # noqa: E402
from src.benchmark.gpu_loader import GpuSplitData  # noqa: E402

SEQ_LEN = 365
BATCH = 1024
EPOCHS = 30
LR = 1e-3
SEED = 42
BOOT_SEED = 20260711
PRED_DIR = PROJECT_ROOT / "results" / "predictions" / "benchmark_671"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "benchmark_671"
VAL_START, VAL_END = "1995-10-01", "2000-09-30"
TEST_START, TEST_END = "2000-10-01", "2008-12-31"


def context_start(target_start: str) -> str:
    date = datetime.strptime(target_start, "%Y-%m-%d") - timedelta(days=SEQ_LEN)
    return date.strftime("%Y-%m-%d")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_loaders(device=None):
    """Load the GPU-resident mirrors produced by prepare_gpu.py.

    The 671-basin datasets must never be constructed inside a training
    process on this workstation: doing so leaves CUDA permanently ~14x
    slower for the rest of the process lifetime. prepare_gpu.py builds
    them in a separate process and stores the mirrors under
    results/cache/benchmark_gpu/.
    """
    if device is None:
        device = get_device()
    cache = PROJECT_ROOT / "results" / "cache" / "benchmark_gpu"
    gpu = {
        split: GpuSplitData.load(cache / f"{split}.npz", device, SEQ_LEN)
        for split in ("train", "calibration", "test")
    }
    basins = gpu["train"].basin_ids

    class _Meta:
        pass

    datasets = {split: _Meta() for split in gpu}
    for split in datasets:
        datasets[split].basin_ids = gpu[split].basin_ids
    return basins, datasets, gpu


def train_model(model, loss_fn, gpu_train, device, log_path: Path,
                epochs: int = EPOCHS, selection_fn=None):
    """Up to 30 epochs with early stopping (patience 5); the returned state
    is the one with the minimum training (or selection) score."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    best_score, best_state, history = float("inf"), None, []
    patience, patience_counter = 5, 0  # matches the archived 671 protocol
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device)
        batch_mark = time.time()
        for batch_number, (dynamic, static, target, mask) in enumerate(
                gpu_train.batches(BATCH, shuffle=True, seed=SEED), 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(dynamic, static), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = mask.sum()
            total = total + loss.detach() * n
            count = count + n
            if batch_number % 300 == 0:
                now = time.time()
                free_b, _total_b = torch.cuda.mem_get_info()
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"  batch {batch_number} +{now - batch_mark:.0f}s "
                        f"reserved={torch.cuda.memory_reserved()/2**20:.0f}M "
                        f"free={free_b/2**20:.0f}M\n")
                batch_mark = now
        train_loss = float(total.item()) / max(float(count.item()), 1.0)
        history.append(train_loss)
        # keep the allocator honest: return unused cached blocks to the
        # driver so desktop contexts never push training tensors off-chip
        torch.cuda.empty_cache()
        scheduler.step(train_loss)
        score = train_loss
        if selection_fn is not None:
            score = selection_fn(model)
        if score < best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        with open(log_path, "a", encoding="utf-8") as handle:
            free_b, total_b = torch.cuda.mem_get_info()
            handle.write(
                f"epoch {epoch:02d}/{epochs} loss={train_loss:.6f} "
                f"select={score:.6f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"time={time.time()-started:.0f}s "
                f"vram alloc={torch.cuda.memory_allocated()/2**20:.0f}M "
                f"reserved={torch.cuda.memory_reserved()/2**20:.0f}M "
                f"free={free_b/2**20:.0f}M\n")
        if patience_counter >= patience:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"early stopping at epoch {epoch}\n")
            break
    if best_state is None:
        raise RuntimeError(
            "no epoch produced a score below the initial best "
            f"(last score {score:.6f}); training diverged")
    model.load_state_dict(best_state)
    return history, float(best_score)


@torch.no_grad()
def gpu_masked_mse(model, gpu_split, device):
    """Selection score over a GPU-resident split (for point reselection)."""
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for dynamic, static, target, mask in gpu_split.batches(BATCH):
        pred = model(dynamic, static)
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        total = total + (loss * mask).sum()
        count = count + mask.sum()
    return float(total.item() / count.item())


def target_stats():
    stats = json.loads(
        (PROJECT_ROOT / "data" / "metadata" / "normalization_stats.json").read_text())
    return float(stats["target_mean"]), float(stats["target_std"])


def interval_metrics(lower, upper, y):
    """Pooled interval scores; bounds are scored exactly as passed (the
    corrections path additionally projects empty intervals, this one does
    not, matching the archived benchmark tables)."""
    penalty = 2.0 / ALPHA
    width = upper - lower
    winkler = (
        width
        + np.where(y < lower, penalty * (lower - y), 0.0)
        + np.where(y > upper, penalty * (y - upper), 0.0)
    )
    return {
        "picp": float(((y >= lower) & (y <= upper)).mean()),
        "mpiw": float(width.mean()),
        "winkler": float(winkler.mean()),
    }


def per_basin_nse(y_log, q50_log, basin):
    observed = np.maximum(np.expm1(y_log), 0.0)
    predicted = np.maximum(np.expm1(q50_log), 0.0)
    values = []
    for b in np.unique(basin):
        sel = basin == b
        denom = np.square(observed[sel] - observed[sel].mean()).sum()
        if denom > 0:
            values.append(1.0 - np.square(observed[sel] - predicted[sel]).sum() / denom)
    return np.asarray(values)


def basin_sufficient(lower, upper, y, basin):
    penalty = 2.0 / ALPHA
    width = upper - lower
    winkler = width + np.maximum(lower - y, 0) * penalty + np.maximum(y - upper, 0) * penalty
    covered = ((y >= lower) & (y <= upper)).astype(np.float64)
    rows = []
    for b in np.unique(basin):
        sel = basin == b
        rows.append((str(b), int(sel.sum()), float(covered[sel].sum()),
                     float(width[sel].sum()), float(winkler[sel].sum())))
    return rows


def cluster_ci(rows, n_boot=1000):
    arr = np.asarray([[r[2], r[3], r[4], r[1]] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    samples = np.empty((n_boot, 3))
    for i in range(n_boot):
        pick = rng.integers(0, arr.shape[0], size=arr.shape[0])
        tot = arr[pick].sum(axis=0)
        samples[i] = [tot[0] / tot[3], tot[1] / tot[3], tot[2] / tot[3]]
    tot = arr.sum(axis=0)
    point = [tot[0] / tot[3], tot[1] / tot[3], tot[2] / tot[3]]
    return {
        name: {"value": point[j],
               "ci95": [float(np.quantile(samples[:, j], 0.025)),
                        float(np.quantile(samples[:, j], 0.975))]}
        for j, name in enumerate(("picp", "mpiw", "winkler"))
    }


def evaluate_and_save(name, model, datasets, gpu, device, mu, sigma,
                      quantile_fn, checkpoint_path, log_path):
    """quantile_fn(model, dynamic, static) -> (batch, 3) tensor on device."""
    arrays = {}
    model.eval()
    for split in ("calibration", "test"):
        preds, targets, masks = [], [], []
        with torch.no_grad():
            for dynamic, static, target, mask in gpu[split].batches(BATCH):
                q = quantile_fn(model, dynamic, static)
                preds.append(q.cpu().numpy())
                targets.append(target[:, 0].cpu().numpy())
                masks.append(mask[:, 0].cpu().numpy())
        arrays[split] = (
            np.concatenate(preds).astype(np.float64) * sigma + mu,
            np.concatenate(targets).astype(np.float64) * sigma + mu,
            np.concatenate(masks),
        )

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cal_q, cal_y, cal_m = arrays["calibration"]
    obs = cal_m > 0
    q_cal = conformal_radius(np.maximum(
        cal_q[obs, 0] - cal_y[obs], cal_y[obs] - cal_q[obs, 2]))

    test_q, test_y, test_m = arrays["test"]
    obs = test_m > 0
    q05, q50, q95 = test_q[obs, 0], test_q[obs, 1], test_q[obs, 2]
    y = test_y[obs]
    basin = gpu["test"].sample_basin_array()[obs]

    nse = per_basin_nse(y, q50, basin)
    result = {
        "protocol_version": "benchmark_671_v1",
        "method": name,
        "q_cal_log": float(q_cal),
        "raw": interval_metrics(q05, q95, y),
        "cqr": interval_metrics(q05 - q_cal, q95 + q_cal, y),
        "nse": {
            "n_basins": int(nse.size),
            "median": float(np.median(nse)),
            "mean": float(np.mean(nse)),
            "fraction_above_zero": float((nse > 0).mean()),
        },
        "pooled_log_nse": float(
            1.0 - np.square(y - q50).sum() / np.square(y - y.mean()).sum()),
        "cluster_bootstrap_ci95": {
            "raw": cluster_ci(basin_sufficient(q05, q95, y, basin)),
            "cqr": cluster_ci(basin_sufficient(q05 - q_cal, q95 + q_cal, y, basin)),
        },
        "n_test_observed": int(obs.sum()),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_log": str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "timestamp": datetime.now().isoformat(),
    }

    npz_path = PRED_DIR / f"{name}_predictions.npz"
    temp = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp,
        basin_ids=np.asarray(datasets["test"].basin_ids),
        calibration_dates=gpu["calibration"].target_dates,
        test_dates=gpu["test"].target_dates,
        calibration_mask=arrays["calibration"][2].astype(np.uint8),
        test_mask=arrays["test"][2].astype(np.uint8),
        target_calibration=arrays["calibration"][1].astype(np.float32),
        target_test=arrays["test"][1].astype(np.float32),
        calibration_quantiles_log=arrays["calibration"][0].astype(np.float32),
        test_quantiles_log=arrays["test"][0].astype(np.float32),
    )
    temp.replace(npz_path)
    result["predictions"] = str(npz_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    result["predictions_sha256"] = sha256(npz_path)

    out = TABLE_DIR / f"{name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{name}] raw picp={result['raw']['picp']:.4f} "
          f"mpiw={result['raw']['mpiw']:.4f} winkler={result['raw']['winkler']:.4f} | "
          f"cqr picp={result['cqr']['picp']:.4f} winkler={result['cqr']['winkler']:.4f} | "
          f"NSEmed={result['nse']['median']:.4f}", flush=True)
    print(f"wrote {out}", flush=True)
    return result
