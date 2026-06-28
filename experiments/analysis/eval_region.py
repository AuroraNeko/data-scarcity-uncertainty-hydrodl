"""Evaluate a single climate region. Accepts group name as argument."""
import sys, json, time, numpy as np, torch, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, PROCESSED_DIR
from src.losses.cqr import compute_uncertainty_metrics
from src.utils import get_device

CKPT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "lpu_stream_quantile_best.pt"
LOG_PATH = PROJECT_ROOT / "eval_region.log"

QUANTILES = [0.05, 0.5, 0.95]
_orig_print = print
results_file = PROJECT_ROOT / "results" / "tables" / "cross_region_results.json"
def log(msg):
    _orig_print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n"); f.flush()

group = sys.argv[1] if len(sys.argv) > 1 else "humid"

# Basin grouping
all_csvs = sorted(PROCESSED_DIR.glob("*.csv"))
aridity = {}
for csv in all_csvs:
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    if "aridity" in df.columns:
        aridity[csv.stem] = float(df["aridity"].iloc[0])

ar_vals = np.array(list(aridity.values()))
q25, q50, q75 = np.percentile(ar_vals, [25, 50, 75])

group_map = {
    "humid": lambda a: a <= q50,
    "dry": lambda a: a > q50,
    "very_humid": lambda a: a <= q25,
    "transitional": lambda a: q25 < a <= q75,
    "very_dry": lambda a: a > q75,
    "all": lambda a: True,
}
select_fn = group_map.get(group, group_map["all"])
basins = sorted([b for b, a in aridity.items() if select_fn(a)])

log(f"Group: {group}, Basins: {len(basins)}, Aridity: [{ar_vals.min():.3f}, {ar_vals.max():.3f}]")

device = get_device()
ckpt = torch.load(CKPT_PATH, weights_only=False)
model = LPUStreamModel(quantiles=QUANTILES).to(device)
model.load_state_dict(ckpt["model_state_dict"])

t0 = time.time()
_, _, test_loader = create_dataloaders(seq_len=365, batch_size=1024, basin_list=basins)

model.eval()
preds, targets, masks = [], [], []
with torch.no_grad():
    for dynamic, static, target, mask, _, _ in test_loader:
        dynamic = dynamic.to(device); static = static.to(device)
        pred = model(dynamic, static)
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy()); masks.append(mask.numpy())

preds = np.concatenate(preds); targets = np.concatenate(targets); masks = np.concatenate(masks)
valid = masks.flatten() > 0
y = targets.flatten()[valid]
q05, q50, q95 = preds[valid, 0], preds[valid, 1], preds[valid, 2]

ss_res = np.sum((y - q50)**2); ss_tot = np.sum((y - y.mean())**2)
nse = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
uq = compute_uncertainty_metrics(q05, q95, y, alpha=0.1)

log(f"NSE: {nse:.4f}, PICP: {uq['picp']:.4f}, MPIW: {uq['mpiw']:.4f}, Winkler: {uq['winkler_score']:.4f}")
log(f"Coverage low/normal/high: {uq['coverage_low_flow']:.4f}/{uq['coverage_normal_flow']:.4f}/{uq['coverage_high_flow']:.4f}")

# Save
result = {"group": group, "n_basins": len(basins), "nse": float(nse),
          "picp": float(uq['picp']), "mpiw": float(uq['mpiw']),
          "winkler": float(uq['winkler_score']),
          "coverage_low": float(uq['coverage_low_flow']),
          "coverage_normal": float(uq['coverage_normal_flow']),
          "coverage_high": float(uq['coverage_high_flow']),
          "time_s": time.time()-t0}

if results_file.exists():
    all_r = json.loads(results_file.read_text())
else:
    all_r = {}
all_r[group] = result
results_file.write_text(json.dumps(all_r, indent=2))
log(f"\nSaved to {results_file}")
