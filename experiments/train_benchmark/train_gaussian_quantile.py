"""Gaussian location-scale quantile baseline (archived artifact id ``umal``).

A single Gaussian head predicts mu and sigma; quantiles come from
mu + sigma * Phi^-1(tau). Despite the artifact id it is not a mixture of
asymmetric Laplacians.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_device, set_seed  # noqa: E402
from src.benchmark.common671 import (  # noqa: E402
    BATCH, LR, SEED, SEQ_LEN, build_loaders, evaluate_and_save, target_stats,
    train_model)
from src.heads.dense_heads import GaussianQuantileStream  # noqa: E402

CKPT_DIR = Path(__file__).resolve().parents[2] / "results/checkpoints/benchmark_671"
NAME = "umal"


def main():
    set_seed(SEED)
    device = get_device()
    basins, datasets, gpu = build_loaders(device)
    model = GaussianQuantileStream(
        n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{NAME}: {n_params:,} parameters, {len(basins)} basins", flush=True)

    log_path = Path(__file__).resolve().parents[2] / 'results/logs/train_gaussian_quantile.log'
    history, best = train_model(model, model.pinball, gpu["train"], device, log_path)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{NAME}.pt"
    config = {
        "architecture": "gaussian_location_scale_quantiles", "legacy_artifact_id": "umal", "n_dynamic": 15, "n_static": 13,
        "hidden_size": 128, "embed_dim": 32, "dropout": 0.3, "n_tau_train": 8,
        "seq_len": SEQ_LEN, "batch_size": BATCH, "learning_rate": LR,
        "epochs": 30, "seed": SEED, "alpha": 0.1,
    }
    torch.save({
        "config": config, "seed": SEED, "procedure": "train_loss",
        "model_state_dict": model.state_dict(),
        "best_state": model.state_dict(),
        "best_loss": best, "completed": True, "epoch": len(history),
    }, ckpt_path)

    mu, sigma = target_stats()
    evaluate_and_save(
        NAME, model, datasets, gpu, device, mu, sigma,
        lambda m, d, s: m.predict_quantiles(m(d, s)), ckpt_path, log_path)
    print(f"done {NAME} {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
