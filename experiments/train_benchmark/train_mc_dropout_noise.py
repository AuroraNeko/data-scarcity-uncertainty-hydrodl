"""MC Dropout with an explicit aleatoric (observation-noise) term."""
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
from src.heads.dense_heads import NoiseStream  # noqa: E402

CKPT_DIR = Path(__file__).resolve().parents[2] / "results/checkpoints/benchmark_671"
NAME = "mc_dropout_noise"
N_MC = 50


def main():
    set_seed(SEED)
    device = get_device()
    basins, datasets, gpu = build_loaders(device)
    model = NoiseStream(
        n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{NAME}: {n_params:,} parameters, {len(basins)} basins", flush=True)

    log_path = Path(__file__).resolve().parents[2] / 'results/logs/train_mc_dropout_noise.log'
    history, best = train_model(model, model.nll, gpu["train"], device, log_path)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{NAME}.pt"
    config = {
        "architecture": "noise_stream", "n_dynamic": 15, "n_static": 13,
        "hidden_size": 128, "embed_dim": 32, "dropout": 0.3, "n_mc": N_MC,
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
    set_seed(20260713)  # fixed inference seed, as in the archived comparison

    def quantile_fn(m, d, s):
        m.train()
        with torch.no_grad():
            mus = []
            sigmas = []
            for _ in range(N_MC):
                raw = m(d, s)
                mu_b, sigma_b = m.split(raw)
                mus.append(mu_b)
                sigmas.append(sigma_b)
        m.eval()
        return NoiseStream.mixture_quantiles(
            torch.cat(mus, dim=1), torch.cat(sigmas, dim=1))

    evaluate_and_save(
        NAME, model, datasets, gpu, device, mu, sigma,
        quantile_fn, ckpt_path, log_path)
    print(f"done {NAME} {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
