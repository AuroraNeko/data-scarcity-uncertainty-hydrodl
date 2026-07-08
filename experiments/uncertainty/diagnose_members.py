"""diagnose_members.py  -  per-member NSE/PICP on full 671 test set.

Reports whether the 5 ensemble members are individually comparable to the
single-model reference run (seed 42 = 0.844).
"""
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.cqr import compute_uncertainty_metrics
from src.utils import get_device

device = get_device()
QUANTILES = [0.05, 0.5, 0.95]


def load_q(seed):
    p = PROJECT_ROOT / "results/checkpoints" / (
        "lpu_stream_quantile_best.pt" if seed == 42 else f"ensemble_seed{seed}.pt")
    c = torch.load(p, weights_only=False)
    m = LPUStreamModel(quantiles=QUANTILES).to(device)
    m.load_state_dict(c["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def pred(m, loader):
    ps, ts, ms = [], [], []
    for d, s, t, mk, _, _ in loader:
        ps.append(m(d.to(device), s.to(device)).cpu().numpy())
        ts.append(t.numpy())
        ms.append(mk.numpy())
    return np.concatenate(ps), np.concatenate(ts), np.concatenate(ms)


basins = get_basin_list()
_, _, test_loader = create_dataloaders(seq_len=365, batch_size=1024, basin_list=basins)
print(f"\nPer-member diagnostics on {len(basins)} basins:")
print(f"{'seed':>6}{'NSE':>9}{'PICP':>9}{'MPIW':>9}")
nses = []
for seed in [42, 123, 456, 789, 999]:
    m = load_q(seed)
    p, t, mk = pred(m, test_loader)
    v = mk.flatten() > 0
    y = t.flatten()[v]
    med = p[v, 1]
    nse = float(1 - np.sum((y - med) ** 2) / np.sum((y - y.mean()) ** 2))
    raw = compute_uncertainty_metrics(p[v, 0], p[v, 2], y, alpha=0.1)
    nses.append(nse)
    print(f"{seed:>6}{nse:>9.4f}{raw['picp']:>9.4f}{raw['mpiw']:>9.4f}")
nses = np.array(nses)
print(f"\nmean NSE = {nses.mean():.4f}   std = {nses.std():.4f}   "
      f"range = [{nses.min():.4f}, {nses.max():.4f}]")
