"""train_multiseed_trainloss.py — Train LPU-Stream quantile model on several
seeds using the SAME checkpoint-selection procedure as train_quantile.py
(best-by-TRAINING-loss), to test whether seed 42's 0.844 NSE is
procedure-stable or a lucky outlier.

For each seed we report test NSE. If the extra seeds also reach ~0.84, the
train-loss-selection procedure is stable and 0.844 is legitimate (and the
ensemble script's val-loss selection was the bug that undertrained members
1-4). If they land near 0.57-0.68, seed 42 is an outlier and the headline NSE
must be revised.

Usage:
    python experiments/uncertainty/train_multiseed_trainloss.py
"""
import sys
import json
import time
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
SEEDS = [123, 456, 789]
CFG = dict(n_dynamic=15, n_static=13, hidden_size=128, embed_dim=32,
           dropout=0.3, seq_len=365, batch_size=1024, learning_rate=1e-3,
           epochs=30, alpha=0.1)


def compute_nse(pred, target, mask):
    med = pred[:, 1:2]
    v = mask.flatten() > 0
    p, t = med.flatten()[v], target.flatten()[v]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ts, ms = [], [], []
    for d, s, t, m, _, _ in loader:
        ps.append(model(d.to(device), s.to(device)).cpu().numpy())
        ts.append(t.numpy())
        ms.append(m.numpy())
    return np.concatenate(ps), np.concatenate(ts), np.concatenate(ms)


def run_seed(seed, train_loader, val_loader, test_loader, device):
    set_seed(seed)
    model = LPUStreamModel(
        n_dynamic=CFG["n_dynamic"], n_static=CFG["n_static"],
        hidden_size=CFG["hidden_size"], embed_dim=CFG["embed_dim"],
        dropout=CFG["dropout"], quantiles=QUANTILES,
    ).to(device)
    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # Mirror train_quantile.py: select best by TRAINING loss
    best_train_loss, best_state, best_epoch = float("inf"), None, 0
    patience, ctr = 5, 0
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot, nv = 0.0, 0
        for d, s, t, m, _, _ in train_loader:
            d, s, t, m = d.to(device), s.to(device), t.to(device), m.to(device)
            optimizer.zero_grad()
            pred = model(d, s)
            loss = criterion(pred, t, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item() * m.sum().item()
            nv += m.sum().item()
        train_loss = tot / max(nv, 1.0)
        scheduler.step(train_loss)  # train_quantile steps on train_loss
        improved = train_loss < best_train_loss
        if improved:
            best_train_loss, best_epoch = train_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1
            if ctr >= patience:
                break
        print(f"  seed {seed} ep{epoch:2d} train_loss={train_loss:.4f} "
              f"{'*best*' if improved else ''} {time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    _, _, _ = predict(model, val_loader, device)
    tp, tt, tm = predict(model, test_loader, device)
    nse = compute_nse(tp, tt, tm)
    print(f"==> seed {seed}: test NSE = {nse:.4f} (best_epoch {best_epoch})",
          flush=True)
    return {"seed": seed, "test_nse": nse, "best_epoch": best_epoch,
            "best_train_loss": float(best_train_loss)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print("Loading data...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=CFG["seq_len"], batch_size=CFG["batch_size"],
        basin_list=get_basin_list())

    results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nTraining seed {seed} (train-loss selection)\n{'='*60}",
              flush=True)
        results.append(run_seed(seed, train_loader, val_loader, test_loader, device))

    nses = np.array([r["test_nse"] for r in results])
    print("\n" + "=" * 60)
    print("MULTI-SEED (train-loss selection) SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  seed {r['seed']}: NSE={r['test_nse']:.4f}")
    print(f"  mean={nses.mean():.4f}  std={nses.std():.4f}  "
          f"range=[{nses.min():.4f},{nses.max():.4f}]")
    print(f"  (seed 42 reference = 0.8442)")
    print("\nInterpretation:")
    if nses.mean() > 0.80:
        print("  >> Train-loss selection is STABLE: extra seeds ~0.84.")
        print("     => 0.844 is legitimate; ensemble script (val-loss) was the bug.")
    else:
        print("  >> Extra seeds stay LOW (~0.6-0.7): seed 42 is an OUTLIER.")
        print("     => headline NSE 0.844 not robust; revise NSE claims down.")

    out = {"procedure": "train_loss_selection", "seeds": SEEDS,
           "results": results, "mean_nse": float(nses.mean()),
           "std_nse": float(nses.std()), "seed42_reference": 0.8442}
    out_path = PROJECT_ROOT / "results" / "tables" / "multiseed_trainloss.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
