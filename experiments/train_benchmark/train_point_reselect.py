"""Point-model retraining with checkpoint selection on its training-period tail.

The archived point checkpoint (MC Dropout base) was selected by validation
MSE on 1995-2000, which the CQR calibration later reuses. This retraining
selects on the training-period tail 1992-10-01..1995-09-30 instead, so no
1995-2000 information influences checkpoint selection; the training window
matches the archived period. This implementation schedules learning rate on
training loss; the archived point-model entry point schedules on validation
loss. The comparison therefore changes both selection data and optimization.
It estimates sensitivity across these fixed fits. MC Dropout
intervals (50 passes, empirical 5th/50th/95th percentiles) are then
re-evaluated with the reselected checkpoint.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import get_device, set_seed  # noqa: E402
from src.benchmark.common671 import (  # noqa: E402
    BATCH, LR, SEED, SEQ_LEN, TABLE_DIR, build_loaders,
    evaluate_and_save, gpu_masked_mse, target_stats, train_model)
from src.benchmark.gpu_loader import GpuSplitData  # noqa: E402
from src.heads.dense_heads import PointChunkedStream  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CKPT_DIR = Path(__file__).resolve().parents[2] / "results/checkpoints/benchmark_671"
NAME = "mc_dropout_reselect"
N_MC = 50
SELECT_START, SELECT_END = "1992-10-01", "1995-09-30"


def main():
    set_seed(SEED)
    device = get_device()
    basins, datasets, gpu = build_loaders(device)

    gpu_selection = GpuSplitData.load(
        PROJECT_ROOT / "results" / "cache" / "benchmark_gpu" / "selection.npz",
        device, SEQ_LEN)

    model = PointChunkedStream(
        n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3,
        quantiles=None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{NAME}: {n_params:,} parameters, {len(basins)} basins", flush=True)

    def masked_mse(pred, target, mask):
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def selection_fn(m):
        return gpu_masked_mse(m, gpu_selection, device)

    log_path = Path(__file__).resolve().parents[2] / 'results/logs/train_point_reselect.log'
    history, best = train_model(
        model, masked_mse, gpu["train"], device, log_path,
        selection_fn=selection_fn)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "point_reselect.pt"
    config = {
        "architecture": "lpu_stream_point", "n_dynamic": 15, "n_static": 13,
        "hidden_size": 128, "embed_dim": 32, "dropout": 0.3,
        "seq_len": SEQ_LEN, "batch_size": BATCH, "learning_rate": LR,
        "epochs": 30, "seed": SEED, "alpha": 0.1,
        "checkpoint_selection": "mse_on_training_tail_1992-10-01_1995-09-30",
    }
    torch.save({
        "config": config, "seed": SEED, "procedure": "tail_selection",
        "model_state_dict": model.state_dict(),
        "best_state": model.state_dict(),
        "best_selection_loss": best, "completed": True, "epoch": len(history),
    }, ckpt_path)

    mu, sigma = target_stats()
    set_seed(20260713)

    def quantile_fn(m, d, s):
        m.train()
        with torch.no_grad():
            passes = [m(d, s).squeeze(-1) for _ in range(N_MC)]
        m.eval()
        stack = torch.stack(passes, dim=0)  # (n_mc, batch)
        taus = torch.tensor([0.05, 0.50, 0.95], device=stack.device)
        return torch.quantile(stack, taus, dim=0).transpose(0, 1)

    evaluate_and_save(
        NAME, model, datasets, gpu, device, mu, sigma,
        quantile_fn, ckpt_path, log_path)

    result_path = TABLE_DIR / f"{NAME}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["selection"] = {
        "segment": [SELECT_START, SELECT_END],
        "note": "checkpoint selected on training-period tail; 1995-2000 never "
                "used for selection, leaving the CQR calibration window clean",
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"done {NAME} {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
