"""eval_point_perbasin.py — Standard per-basin NSE (raw flow) for the
point-prediction models, for an honest, literature-comparable Table 1.

Computes per-basin NSE on raw streamflow (mm/day) and reports the median and
mean across basins with cluster-bootstrap 95% CIs.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'baseline'))

from src.data.dataset import create_dataloaders, get_basin_list
from train_model import build_model

STATS = json.load(open(PROJECT_ROOT / 'data' / 'metadata' / 'normalization_stats.json'))
MEAN, STD = STATS['target_mean'], STATS['target_std']


def to_raw(x_norm):
    return np.expm1(x_norm * STD + MEAN)


@torch.no_grad()
def collect(model, loader, device):
    preds, tgts, masks, bidxs = [], [], [], []
    for d, s, t, m, bi, _ in loader:
        preds.append(model(d.to(device), s.to(device)).cpu().numpy())
        tgts.append(t.numpy()); masks.append(m.numpy()); bidxs.append(bi.numpy())
    return (np.concatenate(preds), np.concatenate(tgts),
            np.concatenate(masks), np.concatenate(bidxs))


def per_basin(pred, tgt, mask, bidx, B=1000, seed=0):
    med = []
    for b in np.unique(bidx):
        m = (bidx == b) & (mask.flatten() > 0)
        if m.sum() < 5:
            continue
        y = to_raw(tgt.flatten()[m]); p = to_raw(pred[m, 0])
        ss = np.sum((y - y.mean()) ** 2)
        if ss > 0:
            med.append(1 - np.sum((y - p) ** 2) / ss)
    med = np.array(med)
    rng = np.random.RandomState(seed); n = len(med)
    boot_med = [np.nanmedian(med[rng.randint(0, n, n)]) for _ in range(B)]
    boot_mean = [np.nanmean(med[rng.randint(0, n, n)]) for _ in range(B)]
    return (float(np.nanmedian(med)), float(np.nanmean(med)),
            (float(np.percentile(boot_med, 2.5)), float(np.percentile(boot_med, 97.5))),
            (float(np.percentile(boot_mean, 2.5)), float(np.percentile(boot_mean, 97.5))))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basins = get_basin_list()
    print(f"Loading test data ({len(basins)} basins)...", flush=True)
    _, _, test_loader = create_dataloaders(seq_len=365, batch_size=1024, basin_list=basins)

    models = [
        ('lpu_stream', 'lpu_stream_best', 'LPU-Stream (MSE point, Part C)'),
        ('lstm', 'lstm_best', 'LSTM'),
        ('ea_lstm', 'ea_lstm_best', 'EA-LSTM'),
        ('transformer', 'transformer_best', 'Transformer'),
    ]
    out = {}
    print(f"\n{'Model':<34}{'median NSE':>12}{'mean NSE':>11}  95% CI (median)")
    print("-" * 80)
    for name, ck, disp in models:
        c = torch.load(PROJECT_ROOT / 'results' / 'checkpoints' / f'{ck}.pt',
                       weights_only=False, map_location='cpu')
        model = build_model(name, c['config']).to(device)
        model.load_state_dict(c['model_state_dict']); model.eval()
        pred, tgt, mask, bidx = collect(model, test_loader, device)
        med, mean, cimed, cimean = per_basin(pred, tgt, mask, bidx)
        print(f"{disp:<34}{med:>12.4f}{mean:>11.4f}  [{cimed[0]:.4f}, {cimed[1]:.4f}]")
        out[ck] = {'display': disp, 'nse_median': med, 'nse_mean': mean,
                   'ci_median': list(cimed), 'ci_mean': list(cimean)}
    with open(PROJECT_ROOT / 'results' / 'tables' / 'perbasin_nse_point.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved -> results/tables/perbasin_nse_point.json")


if __name__ == '__main__':
    main()
