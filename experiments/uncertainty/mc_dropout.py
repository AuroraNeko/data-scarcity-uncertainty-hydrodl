"""
mc_dropout.py  -  MC Dropout uncertainty baseline.

Runs 50 stochastic forward passes with dropout enabled at inference time,
forms a 90% prediction interval from the 5th/95th percentiles, and compares
the result against the CQR-calibrated model.

Usage:
    python experiments/uncertainty/mc_dropout.py

Requires a trained point-prediction checkpoint at
``results/checkpoints/lpu_stream_best.pt`` (produced by the baseline trainer).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.cqr import compute_uncertainty_metrics
from src.utils import get_device

CKPT_DIR = PROJECT_ROOT / "results" / "checkpoints"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

device = get_device()

# Load Phase 2 point model
ckpt = torch.load(CKPT_DIR / "lpu_stream_best.pt", weights_only=False)
cfg = ckpt["config"]
model = LPUStreamModel(
    n_dynamic=cfg["n_dynamic"], n_static=cfg["n_static"],
    hidden_size=cfg["hidden_size"], embed_dim=cfg["embed_dim"],
    dropout=cfg["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.train()  # Enable Dropout at inference time

print("Loading test data...", flush=True)
_, _, test_loader = create_dataloaders(
    seq_len=cfg["seq_len"], batch_size=1024, basin_list=get_basin_list())

N_MC = 50
print(f"Running {N_MC} MC passes...", flush=True)

all_preds = []
all_targets = []
all_masks = []

with torch.no_grad():
    for i in range(N_MC):
        preds_b = []
        for dynamic, static, target, mask, _, _ in test_loader:
            dynamic, static = dynamic.to(device), static.to(device)
            pred = model(dynamic, static)
            preds_b.append(pred.cpu().numpy())
            # Collect ground truth only once (identical across MC passes)
            if i == 0:
                all_targets.append(target.numpy())
                all_masks.append(mask.numpy())
        all_preds.append(np.concatenate(preds_b))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_MC}", flush=True)

all_targets = np.concatenate(all_targets)
all_masks = np.concatenate(all_masks)

# Stack + compute intervals
mc = np.stack(all_preds, axis=0).squeeze(-1)
q05 = np.percentile(mc, 5, axis=0)
q95 = np.percentile(mc, 95, axis=0)
q50 = np.percentile(mc, 50, axis=0)

valid = all_masks.flatten() > 0
y = all_targets.flatten()[valid]
ss_res = np.sum((y - q50[valid]) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
nse = float(1 - ss_res / ss_tot)

raw = compute_uncertainty_metrics(q05[valid], q95[valid], y, alpha=0.1)

# Compare with CQR
with open(TABLES_DIR / "lpu_stream_quantile_results.json") as f:
    cqr_r = json.load(f)["test_calibrated"]

print()
print("=" * 60)
print("MC Dropout vs CQR (Test Set)")
print("=" * 60)
print(f'{"":20} {"MC Dropout":>12} {"CQR":>12}')
print(f'{"NSE":20} {nse:>12.4f} {cqr_r.get("nse", 0):>12}')
print(f'{"PICP":20} {raw["picp"]:>12.4f} {cqr_r["picp"]:>12.4f}')
print(f'{"MPIW":20} {raw["mpiw"]:>12.4f} {cqr_r["mpiw"]:>12.4f}')
print(f'{"Winkler":20} {raw["winkler_score"]:>12.4f} {cqr_r["winkler_score"]:>12.4f}')
print(f'{"Cov. low":20} {raw["coverage_low_flow"]:>12.4f} {cqr_r["coverage_low_flow"]:>12.4f}')
print(f'{"Cov. normal":20} {raw["coverage_normal_flow"]:>12.4f} {cqr_r["coverage_normal_flow"]:>12.4f}')
print(f'{"Cov. high":20} {raw["coverage_high_flow"]:>12.4f} {cqr_r["coverage_high_flow"]:>12.4f}')

results = {
    "model": "mc_dropout", "n_mc_samples": N_MC,
    "test_nse": nse, "test_metrics": raw,
    "vs_cqr": {
        "picp_diff": round(raw["picp"] - cqr_r["picp"], 4),
        "mpiw_diff": round(raw["mpiw"] - cqr_r["mpiw"], 4),
    },
}
TABLES_DIR.mkdir(parents=True, exist_ok=True)
with open(TABLES_DIR / "mc_dropout_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved!")
