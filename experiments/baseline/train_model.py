"""
train_model.py — Unified training script for DL baselines.

Usage:
    python experiments/baseline/train_model.py --model lstm
    python experiments/baseline/train_model.py --model ea_lstm
    python experiments/baseline/train_model.py --model tcn
    python experiments/baseline/train_model.py --model transformer

Paper reference: Section 8.3-8.6
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import LSTMModel
from src.models.ea_lstm import EALSTMModel
from src.models.tcn import TCNModel
from src.models.transformer import TransformerModel
from src.models.lpu_stream import LPUStreamModel
from src.data.dataset import create_dataloaders, get_basin_list
from src.utils import set_seed

# ─── Default configs per model (paper Sections 8.3-8.6) ─────────────────────
MODEL_CONFIGS = {
    "lstm": {
        "hidden_size": 128, "num_layers": 1, "dropout": 0.3,
        "batch_size": 256, "learning_rate": 1e-3, "epochs": 30,
    },
    "ea_lstm": {
        "hidden_size": 128, "dropout": 0.3,
        "batch_size": 1024, "learning_rate": 1e-3, "epochs": 30,
    },
    "tcn": {
        "channels": 64, "kernel_size": 3, "n_blocks": 6, "dropout": 0.2,
        "batch_size": 256, "learning_rate": 1e-3, "epochs": 30,
    },
    "transformer": {
        "d_model": 128, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 256, "dropout": 0.2,
        "batch_size": 128, "learning_rate": 5e-4, "epochs": 30,
    },
    "lpu_stream": {
        "hidden_size": 128, "embed_dim": 32, "dropout": 0.3,
        "batch_size": 1024, "learning_rate": 1e-3, "epochs": 30,
    },
}

COMMON_CONFIG = {
    "seq_len": 365,
    "n_dynamic": 5,
    "n_static": 13,
    "seed": 42,
}


def build_model(model_name: str, cfg: dict) -> nn.Module:
    nd, ns = cfg["n_dynamic"], cfg["n_static"]
    if model_name == "lstm":
        return LSTMModel(nd, ns, cfg["hidden_size"], cfg.get("num_layers", 1), cfg["dropout"])
    elif model_name == "ea_lstm":
        return EALSTMModel(nd, ns, cfg["hidden_size"], cfg["dropout"])
    elif model_name == "tcn":
        return TCNModel(nd, ns, cfg["channels"], cfg["kernel_size"], cfg["n_blocks"], cfg["dropout"])
    elif model_name == "transformer":
        return TransformerModel(nd, ns, cfg["d_model"], cfg["nhead"], cfg["num_layers"],
                                cfg["dim_feedforward"], cfg["dropout"])
    elif model_name == "lpu_stream":
        return LPUStreamModel(nd, ns, cfg["hidden_size"],
                              cfg["embed_dim"], cfg["dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_valid = 0

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(dynamic, static)

        loss = (criterion(pred, target) * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        n_valid += mask.sum().item()

    return total_loss / max(n_valid, 1.0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_valid = 0
    all_preds, all_targets, all_masks = [], [], []

    for dynamic, static, target, mask, _, _ in loader:
        dynamic, static = dynamic.to(device), static.to(device)
        target, mask = target.to(device), mask.to(device)

        pred = model(dynamic, static)
        total_loss += (criterion(pred, target) * mask).sum().item()
        n_valid += mask.sum().item()

        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    avg_loss = total_loss / max(n_valid, 1.0)
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    return avg_loss, preds, targets, masks


def compute_nse(preds, targets, masks):
    valid = masks.flatten() > 0
    if valid.sum() == 0:
        return 0.0
    p, t = preds.flatten()[valid], targets.flatten()[valid]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "ea_lstm", "tcn", "transformer", "lpu_stream"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_name = args.model
    if args.seed is not None:
        COMMON_CONFIG["seed"] = args.seed
    cfg = {**COMMON_CONFIG, **MODEL_CONFIGS[model_name]}
    if args.epochs:
        cfg["epochs"] = args.epochs

    print("=" * 60)
    print(f"{model_name.upper()} Training — CAMELS-US Task 1")
    print("=" * 60)
    print(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        seq_len=cfg["seq_len"], batch_size=cfg["batch_size"],
        basin_list=get_basin_list(),
    )

    # Model
    model = build_model(model_name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # Training
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"\nTraining for {cfg['epochs']} epochs...")
    print("-" * 60)

    best_val_loss, best_epoch = float("inf"), 0
    patience = 5
    patience_counter = 0
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{cfg['seed']}" if args.seed is not None else ""
    best_path = ckpt_dir / f"{model_name}{seed_tag}_best.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, vp, vt, vm = evaluate(model, val_loader, criterion, device)
        val_nse = compute_nse(vp, vt, vm)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"NSE: {val_nse:.4f} | LR: {lr:.1e} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "val_nse": val_nse, "config": cfg,
            }, best_path)
            print(f"  -> Saved best (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Test
    print("\n" + "=" * 60)
    print("Test Evaluation")
    print("=" * 60)

    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, tp, tt, tm = evaluate(model, test_loader, criterion, device)
    test_nse = compute_nse(tp, tt, tm)

    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test NSE: {test_nse:.4f}")

    # Save results
    results = {
        "model": model_name, "config": cfg,
        "best_epoch": best_epoch, "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss), "test_nse": float(test_nse),
        "n_params": n_params, "timestamp": datetime.now().isoformat(),
    }
    results_path = PROJECT_ROOT / "results" / "tables" / f"{model_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
