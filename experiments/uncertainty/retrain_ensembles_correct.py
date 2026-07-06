"""retrain_ensembles_correct.py — Retrain the 4 Deep Ensemble members with the
CORRECT training procedure, then rebuild the fair 671-basin comparison.

Correct procedure = train-loss checkpoint selection (effectively train the full
30 epochs), matching train_quantile.py and the main model. Verified stable:
seed 42 -> 0.844, seed 123 -> 0.843. The old members (val-loss early stopping)
gave 0.57-0.68 because validation pinball loss bottoms out around epoch 8 while
median NSE keeps improving to epoch 30.

RESUMABLE (crash-safe): each member's checkpoint is overwritten every epoch
with the FULL training state (model + optimizer + scheduler + RNG + best
state + epoch + completed flag). On restart:
  - completed=True            -> skip
  - exists, completed=False   -> resume from next epoch (optimizer/sched/RNG restored)
  - missing                   -> train from scratch
Worst-case loss on a crash is one epoch (~7 min), not a full retrain.

Usage:
    python -u experiments/uncertainty/retrain_ensembles_correct.py
"""
import sys
import json
import time
import random
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.pinball_loss import PinballLoss
from src.losses.cqr import CQRCalibrator, compute_uncertainty_metrics
from src.utils import set_seed

QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.1
TRAIN_SEEDS = [123, 456, 789, 999]
ALL_SEEDS = [42] + TRAIN_SEEDS
CFG = dict(n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3,
           seq_len=365, batch_size=1024, learning_rate=1e-3, epochs=30)

CKPT = PROJECT_ROOT / "results" / "checkpoints"
TABLES = PROJECT_ROOT / "results" / "tables"
LOG_PATH = PROJECT_ROOT / "retrain_ensembles.log"


def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def ckpt_path(seed: int) -> Path:
    return CKPT / ("lpu_stream_quantile_best.pt" if seed == 42
                   else f"ensemble_seed{seed}.pt")


def get_rng_state():
    st = {"torch": torch.get_rng_state(), "np": np.random.get_state(),
          "py": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st):
    try:
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["np"])
        random.setstate(st["py"])
        if torch.cuda.is_available() and "cuda" in st:
            torch.cuda.set_rng_state(st["cuda"])
    except Exception:
        pass


def compute_nse(pred, target, mask):
    med = pred[:, 1:2]
    v = mask.flatten() > 0
    p, t = med.flatten()[v], target.flatten()[v]
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > 0 else 0.0


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ts, ms = [], [], []
    for d, s, t, m, _, _ in loader:
        ps.append(model(d.to(device), s.to(device)).cpu().numpy())
        ts.append(t.numpy())
        ms.append(m.numpy())
    return np.concatenate(ps), np.concatenate(ts), np.concatenate(ms)


def status_report():
    """Print which seeds are done / in-progress / todo."""
    log("\n--- checkpoint status ---")
    for seed in TRAIN_SEEDS:
        p = ckpt_path(seed)
        if not p.exists():
            log(f"  seed {seed}: TODO (no checkpoint)")
            continue
        try:
            c = torch.load(p, weights_only=False, map_location="cpu")
            proc = c.get("procedure")
            done = c.get("completed", False)
            ep = c.get("epoch", "?")
            if proc == "train_loss" and done:
                log(f"  seed {seed}: DONE (epoch {ep})")
            elif proc == "train_loss":
                log(f"  seed {seed}: IN-PROGRESS (resume after epoch {ep})")
            else:
                log(f"  seed {seed}: STALE/buggy ({proc!r}, epoch {ep}) -> will retrain")
        except Exception as e:
            log(f"  seed {seed}: UNREADABLE ({e}) -> will retrain")


def train_member(seed, train_loader, device):
    """Train one member with full per-epoch resumable checkpointing."""
    path = ckpt_path(seed)
    resume = None
    if path.exists():
        try:
            c = torch.load(path, weights_only=False, map_location="cpu")
            if c.get("procedure") == "train_loss" and c.get("completed", False):
                log(f"  seed {seed}: already complete (epoch {c.get('epoch')}), skip")
                return
            if c.get("procedure") == "train_loss":
                resume = c
                log(f"  seed {seed}: RESUMING after epoch {c.get('epoch')}")
        except Exception as e:
            log(f"  seed {seed}: unreadable checkpoint ({e}), starting fresh")

    set_seed(seed)
    model = LPUStreamModel(n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32,
                           dropout=0.3, quantiles=QUANTILES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=3)
    crit = PinballLoss(QUANTILES)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    start_epoch = 1

    if resume is not None:
        model.load_state_dict(resume["model_state_dict"])
        opt.load_state_dict(resume["optim_state"])
        sched.load_state_dict(resume["sched_state"])
        best_loss = resume["best_loss"]
        best_epoch = resume["best_epoch"]
        best_state = resume["best_state"]
        start_epoch = resume["epoch"] + 1
        set_rng_state(resume.get("rng", {}))
        log(f"  seed {seed}: restored optim/sched/rng; best={best_loss:.4f}@{best_epoch}")

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, nv = 0.0, 0
        for d, s, t, m, _, _ in train_loader:
            d, s, t, m = d.to(device), s.to(device), t.to(device), m.to(device)
            opt.zero_grad()
            pred = model(d, s)
            loss = crit(pred, t, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * m.sum().item()
            nv += m.sum().item()
        tl = tot / max(nv, 1.0)
        sched.step(tl)
        if tl < best_loss:
            best_loss, best_epoch = tl, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        completed = (epoch >= CFG["epochs"])
        # --- resumable checkpoint: full state, every epoch ---
        torch.save({
            "seed": seed, "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state": opt.state_dict(),
            "sched_state": sched.state_dict(),
            "best_state": best_state,
            "best_loss": float(best_loss), "best_epoch": best_epoch,
            "last_train_loss": float(tl),
            "config": {**CFG, "seed": seed},
            "procedure": "train_loss", "completed": completed,
            "rng": get_rng_state(),
        }, path)
        log(f"  seed {seed} ep{epoch:2d}/{CFG['epochs']} "
            f"loss={tl:.4f} best={best_loss:.4f}@{best_epoch} "
            f"{'DONE' if completed else ''} {time.time()-t0:.0f}s")

    log(f"  seed {seed} TRAINING COMPLETE (best_epoch {best_epoch}, loss {best_loss:.4f})")


def aggregate_and_compare(device, val_loader, test_loader):
    """Load all 5 (correct) members, build ensemble, compute fair comparison."""
    log("\n" + "=" * 60)
    log("Aggregating 5-member ensemble + fair comparison (671 basins)")
    log("=" * 60)
    val_mem, test_mem = [], []
    tt = tm = vt = vm = None
    per_member = []
    for seed in ALL_SEEDS:
        c = torch.load(ckpt_path(seed), weights_only=False, map_location="cpu")
        m = LPUStreamModel(quantiles=QUANTILES).to(device)
        m.load_state_dict(c["best_state"] if c.get("procedure") == "train_loss"
                          else c["model_state_dict"])
        m.eval()
        vp, v_t, v_m = predict(m, val_loader, device)
        tp, t_t, t_m = predict(m, test_loader, device)
        val_mem.append(vp)
        test_mem.append(tp)
        if tt is None:
            tt, tm, vt, vm = t_t, t_m, v_t, v_m
        nse_i = compute_nse(tp, tt, tm)
        per_member.append({"seed": seed, "nse": nse_i})
        log(f"  member seed {seed}: NSE={nse_i:.4f}")

    val_ens = np.mean(np.stack(val_mem, 0), 0)
    test_ens = np.mean(np.stack(test_mem, 0), 0)
    tv = tm.flatten() > 0
    vv = vm.flatten() > 0
    test_y = tt.flatten()[tv]
    val_y = vt.flatten()[vv]

    ens_nse = compute_nse(test_ens, tt, tm)
    ens_raw = compute_uncertainty_metrics(test_ens[tv, 0], test_ens[tv, 2], test_y, alpha=ALPHA)
    cal = CQRCalibrator(ALPHA)
    q_cal = cal.fit(val_ens[vv, 0], val_ens[vv, 2], val_y)
    lo, hi = cal.calibrate(test_ens[tv, 0], test_ens[tv, 2])
    ens_cal = compute_uncertainty_metrics(lo, hi, test_y, alpha=ALPHA)

    nses = np.array([p["nse"] for p in per_member])
    ens_result = {
        "n_members": len(ALL_SEEDS), "seeds": ALL_SEEDS,
        "member_nse": per_member,
        "member_nse_mean": float(nses.mean()), "member_nse_std": float(nses.std()),
        "nse": ens_nse, "q_cal": float(q_cal),
        "raw": ens_raw, "cal": ens_cal,
    }

    # Reuse single-CQR + MC from the prior fair run (they don't change)
    cqr_single = mc = None
    prior = TABLES / "fair_comparison_671.json"
    if prior.exists():
        p = json.load(open(prior))
        cqr_single = p.get("cqr_single")
        mc = p.get("mc_dropout")

    # Save
    out = {"experiment": "deep_ensembles_fair_correct", "n_basins": 671,
           "procedure": "train_loss_selection", "ensemble": ens_result,
           "cqr_single": cqr_single, "mc_dropout": mc,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(TABLES / "deep_ensembles_fair_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # Pretty fair table
    log("\n" + "=" * 72)
    log(f"FAIR comparison (ALL 671 basins) — corrected ensemble")
    log("=" * 72)
    log(f"{'Method':<24}{'NSE':>8}{'PICP':>9}{'MPIW':>9}{'Winkler':>10}")
    log("-" * 72)
    if mc:
        log(f"{'MC Dropout':<24}{mc['nse']:>8.3f}{mc['raw']['picp']:>9.3f}"
            f"{mc['raw']['mpiw']:>9.3f}{mc['raw']['winkler_score']:>10.3f}")
    log(f"{'Deep Ensembles (5)':<24}{ens_nse:>8.3f}{ens_raw['picp']:>9.3f}"
        f"{ens_raw['mpiw']:>9.3f}{ens_raw['winkler_score']:>10.3f}")
    log(f"{'Deep Ensembles+CQR':<24}{ens_nse:>8.3f}{ens_cal['picp']:>9.3f}"
        f"{ens_cal['mpiw']:>9.3f}{ens_cal['winkler_score']:>10.3f}")
    if cqr_single:
        log(f"{'CQR (single)':<24}{cqr_single['nse']:>8.3f}{cqr_single['cal']['picp']:>9.3f}"
            f"{cqr_single['cal']['mpiw']:>9.3f}{cqr_single['cal']['winkler_score']:>10.3f}")
    log("=" * 72)
    log(f"Member NSE: mean={nses.mean():.4f} std={nses.std():.4f} "
        f"range=[{nses.min():.4f},{nses.max():.4f}]")
    log(f"Ensemble q_cal={q_cal:.4f}")
    log(f"Saved -> {TABLES/'deep_ensembles_fair_results.json'}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    log("\n" + "#" * 60)
    log(f"# retrain_ensembles_correct  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("#" * 60)
    log(f"Device: {device}")
    status_report()

    log("\nLoading data (once)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=CFG["seq_len"], batch_size=CFG["batch_size"],
        basin_list=get_basin_list())

    for seed in TRAIN_SEEDS:
        log(f"\n=== member seed {seed} ===")
        train_member(seed, train_loader, device)

    aggregate_and_compare(device, val_loader, test_loader)
    log("\nALL DONE.")


if __name__ == "__main__":
    main()
