"""
Train the seed-42 full-basin quantile checkpoint.

Usage:
    python experiments/train_benchmark/train_single_qr.py

The common-grid raw/CQR comparison is performed by
evaluate_corrected_methods.py after all checkpoints are available.
"""

import sys
import hashlib
import json
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.losses.pinball_loss import PinballLoss
from src.utils import set_seed

# --- Config ----------------------------------------------------------------
# Training hyperparameters. Early stopping / LR scheduling monitor the
# *training* loss (patience 5, max 30 epochs); batch size 1024; dropout 0.3.
QUANTILES = [0.05, 0.5, 0.95]
CONFIG = {
    "n_dynamic": 15, "n_static": 13, "hidden_size": 128,
    "embed_dim": 32, "dropout": 0.3, "quantiles": QUANTILES,
    "seq_len": 365, "batch_size": 1024, "learning_rate": 1e-3,
    "epochs": 30, "seed": 42, "alpha": 0.1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_valid = 0.0, 0

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(dynamic, static)  # (batch, n_quantiles)

        loss = criterion(pred, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_valid = mask.sum().item()
        total_loss += loss.item() * batch_valid
        n_valid += batch_valid

    return total_loss / max(n_valid, 1.0)


@torch.no_grad()
def predict(model, loader, device):
    """Return predictions for all quantiles + targets + masks."""
    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        pred = model(dynamic, static)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    preds = np.concatenate(all_preds)      # (N, n_quantiles)
    targets = np.concatenate(all_targets)   # (N, 1)
    masks = np.concatenate(all_masks)       # (N, 1)
    return preds, targets, masks


def compute_nse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Compute NSE using median (index 1 = Q_0.5)."""
    median = pred[:, 1:2]
    valid = mask.flatten() > 0
    p, t = median.flatten()[valid], target.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main():
    cfg = {**CONFIG}
    print("=" * 60)
    print("LPU-Stream Full-Basin Quantile Checkpoint Training")
    print("=" * 60)
    print(f"Quantiles: {QUANTILES}")
    print(f"Target coverage: {1 - cfg['alpha']:.0%}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
        basin_list=get_basin_list(),
    )

    # Model
    model = LPUStreamModel(
        n_dynamic=cfg["n_dynamic"], n_static=cfg["n_static"],
        hidden_size=cfg["hidden_size"], embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"], quantiles=QUANTILES,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # Loss
    criterion = PinballLoss(QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # Training
    print(f"\nTraining for {cfg['epochs']} epochs...")
    print("-" * 60)

    best_training_loss, best_epoch = float("inf"), 0
    patience, patience_counter = 5, 0
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lpu_stream_quantile_best.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation: predict + compute NSE from median
        val_preds, val_targets, val_masks = predict(model, val_loader, device)
        val_nse = compute_nse(val_preds, val_targets, val_masks)
        # Early stopping / LR scheduling use the *training* loss below (a full
        # validation pinball pass is comparatively expensive). val_nse is
        # reported for monitoring only.
        scheduler.step(train_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | "
              f"Val NSE: {val_nse:.4f} | "
              f"LR: {lr:.1e} | {elapsed:.0f}s")

        if train_loss < best_training_loss:
            best_training_loss, best_epoch = train_loss, epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_nse": val_nse, "training_loss": float(train_loss),
                "checkpoint_selection": "minimum_training_pinball_loss",
                "config": cfg,
            }, best_path)
            print(f"  -> Saved best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    results = {
        "model": "lpu_stream_quantile",
        "model_seed": cfg["seed"],
        "config": cfg,
        "n_parameters": n_params,
        "best_epoch": best_epoch,
        "best_training_loss": float(best_training_loss),
        "checkpoint_selection": "minimum_training_pinball_loss",
        "training_run_complete": True,
        "checkpoint": best_path.relative_to(PROJECT_ROOT).as_posix(),
        "checkpoint_sha256": sha256(best_path),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "full_checkpoint_training"
        / "quantile_seed42.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nCheckpoint metadata saved to {results_path}")


if __name__ == "__main__":
    main()
