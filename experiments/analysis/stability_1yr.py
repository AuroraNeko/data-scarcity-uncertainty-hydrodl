"""stability_1yr.py  -  True initialization stability of the 1-year scarcity model.

Fixes the 50 basins (seed-42 selection, i.e. the Table 2 1-year basin set) and
varies ONLY the model initialization across seeds 42/123/456, reporting NSE and
CQR-calibrated PICP per seed. Saves results/tables/stability_1yr.json so Fig 6a
can be sourced from an artifact.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments' / 'scarce'))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import CamelsDataset, get_basin_list
from src.data.data_preprocessing import VAL_START, VAL_END, TEST_START, TEST_END
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.utils import get_device, set_seed
from train_data_scarce import train_model, predict, compute_nse

QUANTILES = [0.05, 0.5, 0.95]
device = get_device()

# fixed basin selection (seed 42)  -  identical 50 basins across all init seeds
all_basins = get_basin_list()
selected = np.random.RandomState(42).choice(all_basins, 50, replace=False).tolist()
seq_len, train_start, train_end = 30, "1980-10-01", "1981-09-30"
print(f"Fixed {len(selected)} basins; varying only init seed", flush=True)

train_ds = CamelsDataset(selected, train_start, train_end, seq_len)
val_ds = CamelsDataset(selected, VAL_START, VAL_END, seq_len)
test_ds = CamelsDataset(selected, TEST_START, TEST_END, seq_len)
bs = 512
tl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
vl = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)
tel = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, pin_memory=True)

INIT_SEEDS = [42, 123, 456]
rows = []
for s in INIT_SEEDS:
    set_seed(s)  # only initialization + shuffle order vary; basins are fixed
    m = LPUStreamModel(quantiles=QUANTILES).to(device)
    train_model(m, tl, vl, device, epochs=30)
    vp, vt, vm = predict(m, vl, device)
    tp, tt, tm = predict(m, tel, device)
    nse = compute_nse(tp, tt, tm)
    vv = vm.flatten() > 0
    tv = tm.flatten() > 0
    cal = CQRCalibrator(0.1)
    cal.fit(vp[vv, 0], vp[vv, 2], vt.flatten()[vv])
    lo, hi = cal.calibrate(tp[tv, 0], tp[tv, 2])
    picp = compute_uncertainty_metrics(lo, hi, tt.flatten()[tv], alpha=0.1)['picp']
    rows.append({'seed': s, 'nse': float(nse), 'picp_cal': float(picp)})
    print(f"seed {s}: NSE={nse:.4f}  PICP(cal)={picp:.4f}", flush=True)

nses = np.array([r['nse'] for r in rows])
picps = np.array([r['picp_cal'] for r in rows])
out = {'experiment': '1yr_init_stability', 'n_basins': 50,
       'basins': 'fixed (seed-42 selection)', 'init_seeds': INIT_SEEDS,
       'runs': rows,
       'nse_mean': float(nses.mean()), 'nse_std': float(nses.std()),
       'picp_mean': float(picps.mean()), 'picp_std': float(picps.std())}
with open(PROJECT_ROOT / 'results' / 'tables' / 'stability_1yr.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\n" + json.dumps(out, indent=2))
