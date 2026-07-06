"""eval_raw.py — Standard-metric evaluation pipeline (Phase 2).

Evaluates a quantile model in RAW flow units (mm/day), reporting the metrics a
hydrology reviewer expects:
  * per-basin NSE on raw flow (median + mean across basins) — the CAMELS convention
  * pooled PICP / MPIW on raw flow (PICP is invariant under the monotonic inverse
    transform, so it matches the normalized-scale PICP; MPIW is now in mm/day)
  * bootstrap 95% CIs (cluster bootstrap by basin, to respect within-basin correlation)

Inverse transform:  flow_mm = expm1(target_norm * std + mean), where mean/std are
the pooled training-set statistics of log1p(flow).

Usage:
    python experiments/analysis/eval_raw.py <ckpt_name> [--basins all|50]
    e.g. python experiments/analysis/eval_raw.py lpu_stream_quantile_best
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
from src.losses.cqr import CQRCalibrator

META = PROJECT_ROOT / 'data' / 'metadata'
STATS = json.load(open(META / 'normalization_stats.json'))
MEAN, STD = STATS['target_mean'], STATS['target_std']
QUANTILES = [0.05, 0.5, 0.95]


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


def basin_raw_metrics(pred, tgt, mask, bidx):
    """Return per-basin arrays (NSE, PICP, MPIW on raw flow) keyed by basin."""
    basins = np.unique(bidx)
    rec = []
    for b in basins:
        m = (bidx == b) & (mask.flatten() > 0)
        if m.sum() < 5:
            continue
        y = to_raw(tgt.flatten()[m])
        p50 = to_raw(pred[m, 1])
        L = to_raw(pred[m, 0]); U = to_raw(pred[m, 2])
        ss_tot = np.sum((y - y.mean()) ** 2)
        nse = float(1 - np.sum((y - p50) ** 2) / ss_tot) if ss_tot > 0 else np.nan
        rec.append({'bidx': int(b), 'n': int(m.sum()),
                    'nse': nse,
                    'picp': float(((y >= L) & (y <= U)).mean()),
                    'mpiw': float((U - L).mean())})
    return rec


def pooled_from_basins(rec, weights=None):
    """Pooled PICP/MPIW (sample-weighted across basins) + median/mean NSE."""
    n = np.array([r['n'] for r in rec], float)
    w = n / n.sum()
    picp = np.sum([r['picp'] for r in rec] * w)
    mpiw = np.sum([r['mpiw'] for r in rec] * w)
    nses = np.array([r['nse'] for r in rec])
    return dict(picp=float(picp), mpiw=float(mpiw),
                nse_median=float(np.nanmedian(nses)),
                nse_mean=float(np.nanmean(nses)))


def cluster_bootstrap(rec, B=1000, seed=0):
    """Cluster bootstrap by basin: resample basin records with replacement."""
    rng = np.random.RandomState(seed)
    n = len(rec)
    nses = np.array([r['nse'] for r in rec])
    ns = np.array([r['n'] for r in rec], float)
    w = ns / ns.sum()
    pics = np.array([r['picp'] for r in rec])
    mws = np.array([r['mpiw'] for r in rec])
    med, mn, pi, mw = [], [], [], []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        ww = ns[idx] / ns[idx].sum()
        med.append(np.nanmedian(nses[idx]))
        mn.append(np.nanmean(nses[idx]))
        pi.append(np.sum(pics[idx] * ww))
        mw.append(np.sum(mws[idx] * ww))
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {'nse_median_ci': ci(med), 'nse_mean_ci': ci(mn),
            'picp_ci': ci(pi), 'mpiw_ci': ci(mw)}


def main():
    ckpt_name = sys.argv[1] if len(sys.argv) > 1 else 'lpu_stream_quantile_best'
    basins_arg = sys.argv[2] if len(sys.argv) > 2 else 'all'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(PROJECT_ROOT / 'results' / 'checkpoints' / f'{ckpt_name}.pt',
                      weights_only=False, map_location='cpu')
    model = LPUStreamModel(quantiles=QUANTILES).to(device)
    model.load_state_dict(ckpt['best_state'] if ckpt.get('procedure') == 'train_loss'
                          else ckpt['model_state_dict'])
    model.eval()

    basins = get_basin_list() if basins_arg == 'all' else get_basin_list()[:50]
    print(f"Evaluating {ckpt_name} on {len(basins)} basins (RAW units)", flush=True)
    _, val_loader, test_loader = create_dataloaders(seq_len=365, batch_size=1024, basin_list=basins)

    # CQR calibrate on val (normalized space, as trained)
    vp, vt, vm, vb = collect(model, val_loader, device)
    vv = vm.flatten() > 0
    cal = CQRCalibrator(0.1)
    q_cal = cal.fit(vp[vv, 0], vp[vv, 2], vt.flatten()[vv])

    # Test predictions (apply CQR in normalized space, THEN invert)
    tp, tt, tm, tb = collect(model, test_loader, device)
    tv = tm.flatten() > 0
    L_n, U_n = cal.calibrate(tp[tv, 0], tp[tv, 2])
    tp_cal = tp.copy()
    tp_cal[tv, 0] = L_n
    tp_cal[tv, 2] = U_n

    rec = basin_raw_metrics(tp_cal, tt, tm, tb)
    pooled = pooled_from_basins(rec)
    ci = cluster_bootstrap(rec)

    print("\n" + "=" * 60)
    print(f"RAW-FLOW METRICS — {ckpt_name} ({len(rec)} basins, CQR-calibrated)")
    print("=" * 60)
    print(f"Per-basin NSE  median = {pooled['nse_median']:.4f}  "
          f"95% CI [{ci['nse_median_ci'][0]:.4f}, {ci['nse_median_ci'][1]:.4f}]")
    print(f"Per-basin NSE  mean   = {pooled['nse_mean']:.4f}  "
          f"95% CI [{ci['nse_mean_ci'][0]:.4f}, {ci['nse_mean_ci'][1]:.4f}]")
    print(f"Pooled PICP (raw)     = {pooled['picp']:.4f}  "
          f"95% CI [{ci['picp_ci'][0]:.4f}, {ci['picp_ci'][1]:.4f}]")
    print(f"Pooled MPIW (mm/day)  = {pooled['mpiw']:.4f}  "
          f"95% CI [{ci['mpiw_ci'][0]:.4f}, {ci['mpiw_ci'][1]:.4f}]")
    print(f"(CQR q_cal in normalized units = {q_cal:.4f})")
    print(f"(reference: pooled-normalized NSE was 0.844)")
    print("=" * 60)

    out = {'ckpt': ckpt_name, 'n_basins': len(rec), 'q_cal_norm': float(q_cal),
           **pooled, **{k: list(v) for k, v in ci.items()}}
    name = ckpt_name.replace('_best', '') + ('_rawall' if basins_arg == 'all' else '_raw50')
    with open(PROJECT_ROOT / 'results' / 'tables' / f'raw_metrics_{name}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> results/tables/raw_metrics_{name}.json")


if __name__ == '__main__':
    main()
