"""eval_perbasin_ckpt.py  -  Evaluate the best_state saved in a per-basin-norm
checkpoint (e.g. the epoch-9 best) for per-basin raw NSE, without waiting for
the full 30-epoch run to finish.
"""
import sys, json, numpy as np, torch
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list

PB = json.load(open(PROJECT_ROOT / 'data/metadata/per_basin_target_stats.json'))
CKPT = PROJECT_ROOT / 'results/checkpoints/lpu_stream_mse_perbasin.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c = torch.load(CKPT, weights_only=False, map_location='cpu')
print(f"checkpoint best_epoch={c.get('best_ep')} best_val_mse={c.get('best_vl'):.4f}")
model = LPUStreamModel(15, 13, 128, 32, 0.3).to(device)
model.load_state_dict(c['best_state']); model.eval()

basin_list = get_basin_list()
_, _, test_loader = create_dataloaders(seq_len=365, batch_size=1024, basin_list=basin_list, per_basin_stats=PB)

preds, tgts, masks, bidxs = [], [], [], []
with torch.no_grad():
    for d, s, t, m, b, _ in test_loader:
        preds.append(model(d.to(device), s.to(device)).cpu().numpy())
        tgts.append(t.numpy()); masks.append(m.numpy()); bidxs.append(b.numpy())
pred = np.concatenate(preds); tgt = np.concatenate(tgts)
mask = np.concatenate(masks); bidx = np.concatenate(bidxs)

med = []
for b in np.unique(bidx):
    m = (bidx == b) & (mask.flatten() > 0)
    if m.sum() < 5:
        continue
    bid = basin_list[int(b)]; st = PB.get(bid)
    if not st:
        continue
    y = np.expm1(tgt.flatten()[m] * st['std'] + st['mean'])
    p = np.expm1(pred[m, 0] * st['std'] + st['mean'])
    ss = np.sum((y - y.mean()) ** 2)
    if ss > 0:
        med.append(1 - np.sum((y - p) ** 2) / ss)
med = np.array(med)
# bootstrap CI
rng = np.random.RandomState(0); n = len(med)
boot = [np.nanmedian(med[rng.randint(0, n, n)]) for _ in range(1000)]
ci = np.percentile(boot, [2.5, 97.5])
print(f"\n=== PER-BASIN NSE (per-basin norm, best epoch {c.get('best_ep')}) ===")
print(f"median NSE = {np.nanmedian(med):.4f}  (95% CI [{ci[0]:.4f}, {ci[1]:.4f}])")
print(f"mean NSE   = {np.nanmean(med):.4f}")
print(f"(global-norm reference: median 0.4625; CAMELS literature ~0.55-0.71)")
