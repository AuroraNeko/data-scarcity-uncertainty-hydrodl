"""
confidence_levels.py  -  Evaluate CQR calibration at 90%, 95%, and 99% confidence levels.
Uses the pre-trained quantile model and CQR calibrator with different alpha values.
"""

import sys, json, time, numpy as np, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.losses.pinball_loss import PinballLoss
from src.utils import get_device

CKPT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "lpu_stream_quantile_best.pt"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LOG_PATH = PROJECT_ROOT / "confidence_levels.log"

QUANTILES = [0.05, 0.5, 0.95]

_orig_print = print
def log(msg):
    _orig_print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n"); f.flush()

def predict(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for dynamic, static, target, mask, _, _ in loader:
            dynamic = dynamic.to(device); static = static.to(device)
            pred = model(dynamic, static)
            preds.append(pred.cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
    return (np.concatenate(preds), np.concatenate(targets), np.concatenate(masks))

device = get_device()

# Load model
log("Loading model...")
ckpt = torch.load(CKPT_PATH, weights_only=False)
model = LPUStreamModel(quantiles=QUANTILES).to(device)
model.load_state_dict(ckpt["model_state_dict"])
log(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Get predictions
log("Loading data and predicting...")
t0 = time.time()
train_loader, val_loader, test_loader = create_dataloaders(
    seq_len=365, batch_size=1024, basin_list=get_basin_list())

vp, vt, vm = predict(model, val_loader, device)
tp, tt, tm = predict(model, test_loader, device)
log(f"Predictions done: {time.time()-t0:.0f}s")

valid_v, valid_t = vm.flatten() > 0, tm.flatten() > 0
vy_val = vt.flatten()[valid_v]
vy_test = tt.flatten()[valid_t]
vq05, vq95 = vp[valid_v, 0], vp[valid_v, 2]

tq05_test, tq95_test = tp[valid_t, 0], tp[valid_t, 2]

# Evaluate at different confidence levels
alphas = [0.10, 0.05, 0.01]  # 90%, 95%, 99%
results = {}

log(f"\n{'='*70}")
log(f"{'CQR Calibration at Multiple Confidence Levels':^70}")
log(f"{'='*70}")
log(f"{'Confidence':<15} {'Alpha':>8} {'q_cal':>10} {'PICP':>8} {'MPIW':>8} {'Winkler':>8} {'Low Cov':>8} {'High Cov':>8}")
log(f"{'-'*70}")

for alpha in alphas:
    conf = f"{(1-alpha)*100:.0f}%"
    calibrator = CQRCalibrator(alpha=alpha)
    q_cal = calibrator.fit(vq05, vq95, vy_val)
    cal_low, cal_high = calibrator.calibrate(tq05_test, tq95_test)
    metrics = compute_uncertainty_metrics(cal_low, cal_high, vy_test, alpha=alpha)
    
    results[conf] = {
        "alpha": alpha, "q_cal": float(q_cal),
        "picp": float(metrics["picp"]),
        "mpiw": float(metrics["mpiw"]),
        "winkler": float(metrics["winkler_score"]),
        "coverage_low": float(metrics["coverage_low_flow"]),
        "coverage_normal": float(metrics["coverage_normal_flow"]),
        "coverage_high": float(metrics["coverage_high_flow"]),
    }
    
    log(f"{conf:<15} {alpha:>8.2f} {q_cal:>10.4f} {metrics['picp']:>8.4f} "
        f"{metrics['mpiw']:>8.4f} {metrics['winkler_score']:>8.4f} "
        f"{metrics['coverage_low_flow']:>8.4f} {metrics['coverage_high_flow']:>8.4f}")

log(f"{'='*70}")

# Also show uncalibrated at 90%
log(f"\n{'For comparison (uncalibrated 90% interval):':^70}")
raw_picp = ((vy_test >= tq05_test) & (vy_test <= tq95_test)).mean()
raw_mpiw = (tq95_test - tq05_test).mean()
log(f"{'Uncalibrated 90%':<15} {'':>8} {'':>10} {raw_picp:>8.4f} {raw_mpiw:>8.4f}")

# Save
results["note"] = "CQR calibration at different confidence levels using pre-trained Q0.05/Q0.5/Q0.95 model"
out_path = TABLES_DIR / "confidence_level_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\n[OK] Saved to {out_path}")
log(f"Done! Confidence level comparison complete.")
