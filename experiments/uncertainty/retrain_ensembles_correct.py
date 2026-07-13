"""Train the four additional members of the full-basin quantile ensemble.

This script uses the same train-loss checkpoint selection as train_quantile.py.
The corrected, common-grid comparison is performed separately by
evaluate_corrected_methods.py.

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
from src.utils import set_seed

QUANTILES = [0.05, 0.5, 0.95]
TRAIN_SEEDS = [123, 456, 789, 999]
CFG = dict(n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32, dropout=0.3,
           seq_len=365, batch_size=1024, learning_rate=1e-3, epochs=30)

CKPT = PROJECT_ROOT / "results" / "checkpoints"
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


def status_report():
    """Print which seeds are done / in-progress / todo."""
    log("\n--- checkpoint status ---")
    for seed in TRAIN_SEEDS:
        p = ckpt_path(seed)
        if not p.exists():
            log(f"  seed {seed}: PENDING (no checkpoint)")
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
                log(f"  seed {seed}: STALE/legacy ({proc!r}, epoch {ep}) -> will retrain")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    log("\n" + "#" * 60)
    log(f"# retrain_ensembles_correct  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("#" * 60)
    log(f"Device: {device}")
    status_report()

    log("\nLoading data (once)...")
    train_loader, _, _ = create_dataloaders(
        seq_len=CFG["seq_len"], batch_size=CFG["batch_size"],
        basin_list=get_basin_list())

    for seed in TRAIN_SEEDS:
        log(f"\n=== member seed {seed} ===")
        train_member(seed, train_loader, device)

    log("\nALL ENSEMBLE CHECKPOINTS READY.")
    log("Run evaluate_corrected_methods.py for the common-grid comparison.")


if __name__ == "__main__":
    main()
