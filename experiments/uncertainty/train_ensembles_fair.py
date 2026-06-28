"""
train_ensembles_fair.py — Fair Deep Ensembles training for CQR comparison.

Strategy:
  - Ensemble member 0: existing lpu_stream_quantile_best.pt (seed=42)
  - Train members 1-4: seeds 123, 456, 789, 999 with IDENTICAL config
  - Evaluate: mean ensemble predictions, CQR calibration, compare with CQR/MC Dropout

Usage:
    python experiments/uncertainty/train_ensembles_fair.py
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Direct file logging (survives w/ pythonw.exe, no console dependency)
_log_file = PROJECT_ROOT / "training_output.log"
_orig_print = print  # save original print
def log(msg: str):
    """Write to both stdout and log file."""
    _orig_print(msg)
    try:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
            f.flush()
    except Exception:
        pass

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.pinball_loss import PinballLoss
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.utils import set_seed

QUANTILES = [0.05, 0.5, 0.95]
BASE_CONFIG = {
    "n_dynamic": 5, "n_static": 13, "hidden_size": 128,
    "embed_dim": 32, "dropout": 0.3, "quantiles": QUANTILES,
    "seq_len": 365, "batch_size": 1024, "learning_rate": 1e-3,
    "epochs": 30, "alpha": 0.1,
}
NEW_SEEDS = [123, 456, 789, 999]   # Train these
ALL_SEEDS = [42] + NEW_SEEDS        # Full ensemble


def train_model(seed: int, device: torch.device) -> dict:
    """Train a single quantile model. Saves checkpoint after EVERY epoch for crash recovery."""
    cfg = {**BASE_CONFIG, "seed": seed}
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ensemble_seed{seed}.pt"

    # Check if already trained
    if ckpt_path.exists():
        log(f"  Seed {seed} checkpoint already exists at {ckpt_path}. Skipping training.")
        ckpt = torch.load(ckpt_path, weights_only=False)
        return {"seed": seed, "best_epoch": ckpt["epoch"], "val_loss": ckpt["val_loss"], "ckpt": str(ckpt_path)}

    set_seed(seed)

    log(f"\n{'='*60}")
    log(f"Training ensemble member — seed={seed}")
    log(f"{'='*60}")

    train_loader, val_loader, _ = create_dataloaders(
        seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
        basin_list=get_basin_list(),
    )

    model = LPUStreamModel(
        n_dynamic=cfg["n_dynamic"], n_static=cfg["n_static"],
        hidden_size=cfg["hidden_size"], embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"], quantiles=QUANTILES,
    ).to(device)

    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss, best_state, best_epoch = float("inf"), None, 0
    patience, patience_counter = 5, 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss, n = 0.0, 0
        for dynamic, static, target, mask, _, _ in train_loader:
            dynamic, static = dynamic.to(device), static.to(device)
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(dynamic, static)
            loss = criterion(pred, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
        train_loss = total_loss / max(n, 1.0)

        # Validate
        model.eval()
        val_total, vn = 0.0, 0
        with torch.no_grad():
            for dynamic, static, target, mask, _, _ in val_loader:
                dynamic, static = dynamic.to(device), static.to(device)
                target, mask = target.to(device), mask.to(device)
                pred = model(dynamic, static)
                loss = criterion(pred, target, mask)
                val_total += loss.item() * mask.sum().item()
                vn += mask.sum().item()
        val_loss = val_total / max(vn, 1.0)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        log(f"  Seed {seed:3d} | Epoch {epoch:2d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | {elapsed:.0f}s")

        is_best = False
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, is_best = val_loss, epoch, True
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"  Seed {seed} early stopping at epoch {epoch}")
                # Save final (best) checkpoint before breaking
                torch.save({
                    "seed": seed, "epoch": best_epoch, "val_loss": float(best_val_loss),
                    "model_state_dict": best_state, "config": cfg,
                }, ckpt_path)
                log(f"  Seed {seed} best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
                return {"seed": seed, "best_epoch": best_epoch, "val_loss": float(best_val_loss), "ckpt": str(ckpt_path)}

        # Save checkpoint after every epoch so partial progress is never lost
        current_state = model.state_dict()
        save_state = best_state if is_best else current_state
        save_epoch = best_epoch if is_best else epoch
        save_loss = best_val_loss if is_best else val_loss
        torch.save({
            "seed": seed, "epoch": save_epoch, "val_loss": float(best_val_loss),
            "model_state_dict": save_state, "config": cfg,
        }, ckpt_path)
        log(f"    -> Checkpoint saved (best epoch so far: {best_epoch})")

    # Final save if loop completes without early stopping
    torch.save({
        "seed": seed, "epoch": best_epoch, "val_loss": float(best_val_loss),
        "model_state_dict": best_state, "config": cfg,
    }, ckpt_path)
    log(f"  Seed {seed} best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    return {"seed": seed, "best_epoch": best_epoch, "val_loss": float(best_val_loss), "ckpt": str(ckpt_path)}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    for dynamic, static, target, mask, _, _ in loader:
        dynamic = dynamic.to(device)
        static = static.to(device)
        pred = model(dynamic, static)
        preds.append(pred.cpu().numpy())
        targets.append(target.numpy())
        masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)


def load_model(seed: int, device: torch.device) -> LPUStreamModel:
    """Load existing checkpoint."""
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    seed42_path = ckpt_dir / "lpu_stream_quantile_best.pt"
    other_path = ckpt_dir / f"ensemble_seed{seed}.pt"
    ckpt_path = seed42_path if seed == 42 else other_path
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = LPUStreamModel(quantiles=QUANTILES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    log("=" * 60)
    log("Deep Ensembles — Fair Re-training")
    log("=" * 60)
    log(f"Member 0:  existing lpu_stream_quantile_best.pt (seed=42)")
    log(f"Members 1-4: training seeds {NEW_SEEDS}")
    log(f"Config: 15yr, 671 basins, pinball loss, early stopping")
    log(f"Total ensemble: {len(ALL_SEEDS)} models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ─── Phase 1: Train new ensemble members ────────────────────────────
    train_infos = []
    for seed in NEW_SEEDS:
        info = train_model(seed, device)
        train_infos.append(info)
        log(f"  Done: seed={seed}, best_epoch={info['best_epoch']}")

    # ─── Phase 2: Load all models and get predictions ───────────────────
    log(f"\n{'='*60}")
    log("Loading all ensemble members...")
    log(f"{'='*60}")

    _, val_loader, test_loader = create_dataloaders(
        seq_len=BASE_CONFIG["seq_len"],
        batch_size=BASE_CONFIG["batch_size"],
        basin_list=get_basin_list(),
    )

    # Collect predictions from each member
    member_test_preds = []
    member_val_preds = []
    test_targets = test_masks = None
    val_targets = val_masks = None

    for seed in ALL_SEEDS:
        model = load_model(seed, device)
        vp, vt, vm = predict(model, val_loader, device)
        tp, tt, tm = predict(model, test_loader, device)
        member_val_preds.append(vp)
        member_test_preds.append(tp)
        if val_targets is None:
            val_targets, val_masks = vt, vm
            test_targets, test_masks = tt, tm
        log(f"  Seed {seed}: loaded")

    # ─── Phase 3: Ensemble aggregation ──────────────────────────────────
    # Average quantile predictions across ensemble members
    val_ensemble = np.mean(np.stack(member_val_preds, axis=0), axis=0)
    test_ensemble = np.mean(np.stack(member_test_preds, axis=0), axis=0)

    valid_test = test_masks.flatten() > 0
    test_y = test_targets.flatten()[valid_test]
    valid_val = val_masks.flatten() > 0
    val_y = val_targets.flatten()[valid_val]

    # NSE from ensemble median
    ens_median = test_ensemble[valid_test, 1]
    ss_res = np.sum((test_y - ens_median) ** 2)
    ss_tot = np.sum((test_y - test_y.mean()) ** 2)
    ens_nse = float(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)

    # Uncalibrated ensemble metrics
    ens_q05 = test_ensemble[valid_test, 0]
    ens_q95 = test_ensemble[valid_test, 2]
    ens_raw = compute_uncertainty_metrics(ens_q05, ens_q95, test_y, alpha=0.1)

    # CQR calibration on ensemble
    calibrator = CQRCalibrator(alpha=0.1)
    q_cal = calibrator.fit(val_ensemble[valid_val, 0], val_ensemble[valid_val, 2], val_y)
    cal_lower, cal_upper = calibrator.calibrate(ens_q05, ens_q95)
    ens_cal = compute_uncertainty_metrics(cal_lower, cal_upper, test_y, alpha=0.1)

    log(f"\nEnsemble NSE: {ens_nse:.4f}")
    log(f"q_cal: {q_cal:.4f}")
    log(f"Uncalibrated: PICP={ens_raw['picp']:.4f}, MPIW={ens_raw['mpiw']:.4f}")
    log(f"CQR Calibrated: PICP={ens_cal['picp']:.4f}, MPIW={ens_cal['mpiw']:.4f}")

    # ─── Phase 4: Load comparison methods ───────────────────────────────
    cqr_path = PROJECT_ROOT / "results" / "tables" / "lpu_stream_quantile_results.json"
    with open(cqr_path) as f:
        cqr = json.load(f)
    mc_path = PROJECT_ROOT / "results" / "tables" / "mc_dropout_results.json"
    with open(mc_path) as f:
        mc = json.load(f)

    cqr_cal = cqr["test_calibrated"]
    mc_met = mc["test_metrics"]

    # ─── Print fair comparison table ────────────────────────────────────
    log(f"\n{'='*75}")
    log(f"{'FAIR Uncertainty Method Comparison (all methods: 15yr, 671 basins)':^75}")
    log(f"{'='*75}")
    hdr = f"{'Method':<25} {'NSE':>8} {'PICP':>8} {'MPIW':>8} {'Winkler':>8} {'Inference':>10}"
    log(hdr)
    log("-" * 75)
    log(f"{'MC Dropout':<25} {mc.get('test_nse',0):>8.4f} {mc_met['picp']:>8.4f} {mc_met['mpiw']:>8.4f} {mc_met['winkler_score']:>8.4f} {'50×':>10}")
    log(f"{'Deep Ensembles (raw)':<25} {ens_nse:>8.4f} {ens_raw['picp']:>8.4f} {ens_raw['mpiw']:>8.4f} {ens_raw['winkler_score']:>8.4f} {f'{len(ALL_SEEDS)}×':>10}")
    log(f"{'Deep Ensembles + CQR':<25} {ens_nse:>8.4f} {ens_cal['picp']:>8.4f} {ens_cal['mpiw']:>8.4f} {ens_cal['winkler_score']:>8.4f} {f'{len(ALL_SEEDS)}×':>10}")
    log(f"{'CQR (single model)':<25} {cqr.get('test_nse',0):>8.4f} {cqr_cal['picp']:>8.4f} {cqr_cal['mpiw']:>8.4f} {cqr_cal['winkler_score']:>8.4f} {'1×':>10}")
    log(f"{'='*75}")

    # ─── Save results ───────────────────────────────────────────────────
    results = {
        "experiment": "deep_ensembles_fair",
        "n_ensemble": len(ALL_SEEDS),
        "seeds": ALL_SEEDS,
        "config": BASE_CONFIG,
        "member_info": train_infos,
        "ensemble": {
            "nse": ens_nse, "q_cal": float(q_cal),
            "uncalibrated": ens_raw, "calibrated": ens_cal,
        },
        "comparison": {
            "mc_dropout": {"nse": mc.get("test_nse",0), "metrics": mc_met},
            "cqr_single": {"nse": cqr.get("test_nse",0), "metrics": cqr_cal},
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_path = PROJECT_ROOT / "results" / "tables" / "deep_ensembles_fair_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Update final_comparison
    final_path = PROJECT_ROOT / "results" / "tables" / "final_comparison.json"
    if final_path.exists():
        with open(final_path) as f:
            final = json.load(f)
        final["deep_ensembles_fair"] = {
            "n_seeds": len(ALL_SEEDS),
            "nse": ens_nse,
            "uncalibrated": ens_raw,
            "calibrated": ens_cal,
        }
        with open(final_path, "w") as f:
            json.dump(final, f, indent=2)

    log(f"\nResults saved to:")
    log(f"  {out_path}")
    log(f"  {final_path}")
    log(f"\nDone!")


if __name__ == "__main__":
    main()
