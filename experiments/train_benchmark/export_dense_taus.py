"""Dense 13-tau grids and multi-level reliability for the dense 671 heads.

Regenerates the 13-tau calibration/test prediction grids for the MDN,
Gaussian-quantile ('umal'), and MC-dropout+noise heads from their
checkpoints, checks them against the archived three-tau predictions
(max abs diff < 1e-4 on the shared float32 grid), and writes both the
dense grids consumed by evaluate/distribution_scores.py and the
multi-level reliability table consumed by
evaluate/method_comparison.py (timing block).

The timing block measures this workstation's GPU; the archived copy in the
frozen results tree records the reference hardware, so a rerun may update
those numbers without touching any deterministic metric.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.common671 import BATCH, build_loaders, target_stats  # noqa: E402
from src.corrections.common import conformal_radius, metrics as interval_metrics  # noqa: E402
from src.heads.dense_heads import (  # noqa: E402
    GaussianQuantileStream,
    HeadStreamModel,
    MDNStream,
    NoiseStream,
    PointChunkedStream,
)
from src.utils import get_device, set_seed, sha256_file  # noqa: E402

CKPT_DIR = ROOT / "results" / "checkpoints" / "benchmark_671"
PRED_DIR = ROOT / "results" / "predictions" / "benchmark_671"
TABLE_DIR = ROOT / "results" / "tables" / "benchmark_671"
CKPT_ROOT = ROOT / "results" / "checkpoints"

TAUS = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50,
        0.75, 0.90, 0.95, 0.975, 0.99, 0.995]
LEVELS = [0.50, 0.80, 0.90, 0.95, 0.99]
for _LEVEL in LEVELS:
    # A level needs both tail taus present on the grid; a nearest-neighbour
    # lookup can silently alias 70% onto the 80% interval.
    for _T in ((1 - _LEVEL) / 2, 1 - (1 - _LEVEL) / 2):
        assert any(abs(_T - t) < 1e-12 for t in TAUS), (_LEVEL, _T)
MC_N, MC_SEED_EVAL = 50, 20260713
TIME_BATCHES, WARMUP = 100, 5
ARCH_KEYS = {0.05: 0, 0.50: 1, 0.95: 2}
MODEL_CLASSES = {
    "mdn": MDNStream,
    "umal": GaussianQuantileStream,
    "mc_dropout_noise": NoiseStream,
}


def tau_index(tau):
    """Nearest-tau lookup; alpha/2 floats like 0.15000000000000002 break ==."""
    return min(range(len(TAUS)), key=lambda i: abs(TAUS[i] - tau))


def load_dense_model(name: str):
    ck = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    state = ck.get("best_state") if ck.get("procedure") == "train_loss" else None
    model = MODEL_CLASSES[name](
        n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3)
    model.load_state_dict(state or ck["model_state_dict"])
    return model


def dense_quantile_fn(name, model):
    if name in ("mdn", "umal"):
        return lambda d, s: model.predict_quantiles(model(d, s), taus=TAUS)
    if name == "mc_dropout_noise":
        def fn(d, s):
            model.train()
            with torch.no_grad():
                mus, sigmas = [], []
                for _ in range(MC_N):
                    raw = model(d, s)
                    mu_b, sigma_b = model.split(raw)
                    mus.append(mu_b)
                    sigmas.append(sigma_b)
            model.eval()
            return NoiseStream.mixture_quantiles(
                torch.cat(mus, dim=1), torch.cat(sigmas, dim=1), taus=TAUS)
        return fn
    raise KeyError(name)


@torch.no_grad()
def eval_grid(name, model, gpu, device, mu, sigma):
    """Predictions at TAUS for both splits; RNG order mirrors the archived export."""
    if name == "mc_dropout_noise":
        set_seed(MC_SEED_EVAL)
    grids, targets, masks, n_batches = {}, {}, {}, {}
    model.eval()
    for split in ("calibration", "test"):
        fn = dense_quantile_fn(name, model)
        preds, tgts, msks = [], [], []
        count = 0
        for dynamic, static, target, mask in gpu[split].batches(BATCH):
            q = fn(dynamic, static)
            preds.append(q.cpu().numpy())
            tgts.append(target[:, 0].cpu().numpy())
            msks.append(mask[:, 0].cpu().numpy())
            count += 1
        n_batches[split] = count
        grids[split] = np.concatenate(preds).astype(np.float64) * sigma + mu
        targets[split] = np.concatenate(tgts).astype(np.float64) * sigma + mu
        masks[split] = np.concatenate(msks)
    return grids, targets, masks, n_batches


def consistency_check(name, grids, masks):
    archived = np.load(PRED_DIR / f"{name}_predictions.npz", allow_pickle=True)
    worst = 0.0
    for split in ("calibration", "test"):
        key = f"{split}_quantiles_log"
        obs = masks[split] > 0
        fresh = grids[split][obs]
        for tau, col in ARCH_KEYS.items():
            ref = archived[key][obs, col].astype(np.float64)
            worst = max(worst, float(np.abs(fresh[:, tau_index(tau)] - ref).max()))
    assert worst < 1e-4, (name, worst)
    return worst


def reliability_entry(grids, targets, masks):
    entry = {"quantile_coverage": {}, "interval_levels": {}}
    obs_t = masks["test"] > 0
    y = targets["test"][obs_t]
    obs_c = masks["calibration"] > 0
    y_cal = targets["calibration"][obs_c]
    q_cal = grids["calibration"][obs_c]
    q_test = grids["test"][obs_t]
    for i, tau in enumerate(TAUS):
        entry["quantile_coverage"][str(tau)] = float((y <= q_test[:, i]).mean())
    for level in LEVELS:
        alpha = 1.0 - level
        lo_tau, hi_tau = alpha / 2.0, 1.0 - alpha / 2.0
        lo = q_test[:, tau_index(lo_tau)]
        hi = q_test[:, tau_index(hi_tau)]
        scores = np.maximum(q_cal[:, tau_index(lo_tau)] - y_cal,
                            y_cal - q_cal[:, tau_index(hi_tau)])
        radius = conformal_radius(scores, alpha=alpha)
        entry["interval_levels"][str(level)] = {
            "raw": interval_metrics(lo, hi, y, alpha=alpha),
            "cqr": interval_metrics(lo - radius, hi + radius, y, alpha=alpha),
            "radius": float(radius),
        }
    return entry


def load_lpu_quantile(ckpt_name):
    ckpt = torch.load(CKPT_ROOT / ckpt_name, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    model = HeadStreamModel(
        n_out=3, n_dynamic=config["n_dynamic"], n_static=config["n_static"],
        hidden_size=config["hidden_size"], embed_dim=config["embed_dim"],
        dropout=config.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def time_procedure(fn, batches, device):
    it = iter(batches)
    with torch.no_grad():
        for _ in range(WARMUP):
            dynamic, static, _, _ = next(it)
            fn(dynamic, static)
        torch.cuda.synchronize()
        samples = 0
        started = time.perf_counter()
        for _ in range(TIME_BATCHES):
            dynamic, static, _, _ = next(it)
            fn(dynamic, static)
            samples += int(dynamic.shape[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {"seconds_per_million": elapsed / samples * 1e6,
            "ms_per_batch_1024": elapsed / TIME_BATCHES * 1e3,
            "timed_batches": TIME_BATCHES, "timed_samples": samples,
            "total_seconds": elapsed}


def timing_suite(gpu, device):
    results = {}

    def run(name, fn):
        torch.cuda.empty_cache()
        results[name] = time_procedure(fn, gpu["test"].batches(BATCH), device)
        print(f"[timing] {name}: {results[name]['ms_per_batch_1024']:.1f} ms/batch, "
              f"{results[name]['seconds_per_million']:.1f} s per 10^6", flush=True)

    single = load_lpu_quantile("lpu_stream_quantile_best.pt").to(device).eval()
    with torch.no_grad():
        run("single_qr", lambda d, s: single(d, s))

    members = [single] + [
        load_lpu_quantile(f"ensemble_seed{seed}.pt").to(device).eval()
        for seed in (123, 456, 789, 999)]

    def ensemble_fn(d, s):
        stack = torch.stack([m(d, s) for m in members], dim=0)
        return stack.mean(dim=0)

    run("ensemble", ensemble_fn)

    ckpt = torch.load(CKPT_ROOT / "lpu_stream_best.pt", map_location="cpu",
                      weights_only=False)
    point = PointChunkedStream(
        n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3
    ).to(device)
    point.load_state_dict(ckpt["model_state_dict"])

    def mc_fn(d, s):
        point.train()
        stack = torch.stack([point(d, s).squeeze(-1) for _ in range(MC_N)], dim=0)
        point.eval()
        return torch.quantile(
            stack,
            torch.tensor([0.05, 0.5, 0.95], device=d.device, dtype=stack.dtype),
            dim=0).T

    run("mc_dropout", mc_fn)
    del single, members, point
    torch.cuda.empty_cache()

    for name in ("mdn", "umal"):
        model = load_dense_model(name).to(device)
        model.eval()
        run(name, (lambda mm: lambda d, s: mm.predict_quantiles(mm(d, s)))(model))
        del model
        torch.cuda.empty_cache()

    noise = load_dense_model("mc_dropout_noise").to(device)

    def noise_fn(d, s):
        noise.train()
        with torch.no_grad():
            mus, sigmas = [], []
            for _ in range(MC_N):
                raw = noise(d, s)
                mu_b, sigma_b = noise.split(raw)
                mus.append(mu_b)
                sigmas.append(sigma_b)
        noise.eval()
        return NoiseStream.mixture_quantiles(
            torch.cat(mus, dim=1), torch.cat(sigmas, dim=1))

    run("mc_dropout_noise", noise_fn)
    return results


def archived_reference():
    """Three-point quantile coverage + archived timings of the original suite."""
    out = {"quantile_coverage": {}}
    d = np.load(ROOT / "results" / "predictions" / "corrected_full_671" / "test_predictions.npz",
                allow_pickle=True)
    mask = d["mask"].astype(bool)
    y = d["target"][mask].astype(np.float64)
    for method in ("single_qr", "ensemble", "mc_dropout"):
        q = d[method][mask].astype(np.float64)
        out["quantile_coverage"][method] = {
            "0.05": float((y <= q[:, 0]).mean()),
            "0.5": float((y <= q[:, 1]).mean()),
            "0.95": float((y <= q[:, 2]).mean()),
        }
    # Durations of the original three-tau grid export on this workstation
    # (archived 2026-08-22); this script no longer reruns that suite, so the
    # historical measurements are carried forward instead of re-measured.
    out["archived_full_grid_seconds"] = {
        "single_qr": 69.91682039991429,
        "ensemble": 346.5725617998978,
        "mc_dropout": 3809.0574479002025,
    }
    return out


# Published per-method inference cost of the timing suite below, archived
# 2026-08-22 on the recording workstation (RTX 5060 Ti, batch 1024,
# GPU-resident inputs). The manuscript's inference-cost prose and figure
# quote these values, so they stay fixed; each rerun's fresh measurement is
# recorded alongside in timing_reproduction_check instead of replacing them
# (a rerun right after hours of sustained grid regeneration measures a
# throttled GPU and shifts every entry by double-digit percentages).
ARCHIVED_TIMING = {
    "single_qr": {
        "seconds_per_million": 35.77158691413729,
        "ms_per_batch_1024": 36.63010500007658,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 3.6630105000076583,
    },
    "ensemble": {
        "seconds_per_million": 177.77577636707065,
        "ms_per_batch_1024": 182.04239499988034,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 18.204239499988034,
    },
    "mc_dropout": {
        "seconds_per_million": 1918.6230136718052,
        "ms_per_batch_1024": 1964.6699659999285,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 196.46699659999285,
    },
    "mdn": {
        "seconds_per_million": 77.85639550775159,
        "ms_per_batch_1024": 79.72494899993762,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 7.972494899993762,
    },
    "umal": {
        "seconds_per_million": 35.95005859381217,
        "ms_per_batch_1024": 36.812860000063665,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 3.6812860000063665,
    },
    "mc_dropout_noise": {
        "seconds_per_million": 1971.8453955077566,
        "ms_per_batch_1024": 2019.1696849999428,
        "timed_batches": 100,
        "timed_samples": 102400,
        "total_seconds": 201.91696849999425,
    },
}


def main():
    force = "--force" in sys.argv
    reliability_path = TABLE_DIR / "dense_reliability.json"
    have_grids = all(
        (PRED_DIR / f"dense_{name}_taus.npz").exists() for name in MODEL_CLASSES
    )
    if have_grids and reliability_path.exists() and not force:
        print("SKIP completed dense tau grids and reliability table", flush=True)
        return

    device = get_device()
    basins, datasets, gpu = build_loaders(device)
    mu, sigma = target_stats()
    print(f"GPU: {torch.cuda.get_device_name(device)}, "
          f"{len(basins)} basins, taus={len(TAUS)}, levels={LEVELS}", flush=True)

    out = {
        "protocol_version": "dense_reliability_v1",
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "taus": TAUS,
        "levels": LEVELS,
        "timestamp": datetime.now().isoformat(),
        "methods": {},
        "reference": archived_reference(),
        "timing": {},
    }

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_CLASSES:
        model = load_dense_model(name).to(device)
        grids, targets, masks, n_batches = eval_grid(name, model, gpu, device, mu, sigma)
        worst = consistency_check(name, grids, masks)
        entry = reliability_entry(grids, targets, masks)
        entry["consistency_max_abs_diff"] = worst
        entry["test_batches"] = n_batches["test"]
        out["methods"][name] = entry
        print(f"[{name}] consistency {worst:.2e} | "
              f"tau0.95 cov={entry['quantile_coverage']['0.95']:.4f} | "
              f"L90 raw={entry['interval_levels']['0.9']['raw']['picp']:.4f} "
              f"cqr={entry['interval_levels']['0.9']['cqr']['picp']:.4f}",
              flush=True)
        np.savez_compressed(
            PRED_DIR / f"dense_{name}_taus.npz",
            taus=np.asarray(TAUS, dtype=np.float64),
            calibration=grids["calibration"].astype(np.float32),
            test=grids["test"].astype(np.float32),
            calibration_mask=masks["calibration"].astype(np.uint8),
            test_mask=masks["test"].astype(np.uint8),
        )
        del model
        torch.cuda.empty_cache()

    fresh = timing_suite(gpu, device)
    out["timing"] = ARCHIVED_TIMING
    out["timing_note"] = (
        "timing holds the archived 2026-08-22 measurements quoted by the "
        "manuscript; this run's fresh measurement of the identical suite is "
        "stored in timing_reproduction_check (measured immediately after the "
        "full grid regeneration, i.e. on a heat-soaked GPU, and is therefore "
        "not comparable to the archived cold-state reference).")
    out["timing_reproduction_check"] = fresh
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    reliability_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", reliability_path, flush=True)
    for name in MODEL_CLASSES:
        print(f"dense_{name}_taus.npz sha256 {sha256_file(PRED_DIR / f'dense_{name}_taus.npz')}",
              flush=True)
    print(f"done {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
