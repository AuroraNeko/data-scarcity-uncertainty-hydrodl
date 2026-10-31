"""Training-loss trajectories for the equal-update sensitivity.

Produces paper/figures/figS8_learning_curves.pdf:
  (a) masked pinball training loss vs cumulative optimizer updates for the
      15-year fixed-epoch run (16,050 updates), the 15-year equal-update run
      (1,080 updates), and the 1-year equal-update run (1,080 updates);
  (b) raw test PICP at the three stopping configurations;
  (c) raw test MPIW at the three stopping configurations.
All values are seed means with seed SD bands/bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "figS8_learning_curves.pdf"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GRAY = "#62676D"
INK = "#202428"

SEEDS = (42, 123, 456)


def load_loss_and_steps(path: Path) -> tuple[np.ndarray, int, int]:
    d = json.load(open(path, encoding="utf-8"))
    loss = np.asarray(d["training_loss"], dtype=float)
    steps_per = d.get("steps_per_round") or d.get("steps_per_epoch")
    if steps_per is None:
        import math
        steps_per = math.ceil(d["train_windows"] / d["batch_size"])
    total = d.get("optimizer_steps") or steps_per * len(loss)
    return loss, int(steps_per), int(total)


def load_raw_metrics(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    r = d["raw_interval_metrics"]
    return {"picp": r["picp"], "mpiw": r["mpiw"], "winkler": r["winkler_score"]}

def series(paths):
    losses, totals, mets = [], [], []
    for p in paths:
        loss, per, total = load_loss_and_steps(p)
        losses.append(loss)
        totals.append((per, total))
        mets.append(load_raw_metrics(p))
    return np.stack(losses), totals, mets




def main() -> None:
    eq_dir = ROOT / "results" / "tables" / "scarcity_equal_updates"
    co_dir = ROOT / "results" / "tables" / "corrected_scarcity"

    eq15 = series([eq_dir / f"scarcity_15yr_subsetprimary_seed42_modelseed{s}.json" for s in SEEDS])
    eq1 = series([eq_dir / f"scarcity_1yr_subsetprimary_seed42_modelseed{s}.json" for s in SEEDS])
    fe15 = series([co_dir / f"scarcity_15yr_basinseed42_modelseed{s}.json" for s in SEEDS])

    for name, pack in [("eq15", eq15), ("eq1", eq1), ("fe15", fe15)]:
        per, total = pack[1][0]
        print(name, "steps/round:", per, "total:", total,
              "final loss: %.4f +/- %.4f" % (pack[0][:, -1].mean(), pack[0][:, -1].std()))
        m = pack[2]
        print("   raw picp %.3f  mpiw %.3f  winkler %.3f" % (
            np.mean([x["picp"] for x in m]),
            np.mean([x["mpiw"] for x in m]),
            np.mean([x["winkler"] for x in m])))

    mpl.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
    })

    figure, axes = plt.subplots(
        1, 3, figsize=(10.0, 3.1), constrained_layout=True, width_ratios=[1.7, 1.0, 1.0])

    ax = axes[0]
    for pack, color, label in [
        (fe15, VERMILLION, "15 yr, fixed epochs (16,050 updates)"),
        (eq15, BLUE, "15 yr, equal updates (1,080)"),
        (eq1, GREEN, "1 yr, equal updates (1,080)"),
    ]:
        losses, totals, _ = pack
        per, _ = totals[0]
        x = per * np.arange(1, losses.shape[1] + 1)
        mean = losses.mean(axis=0)
        sd = losses.std(axis=0, ddof=1)
        ax.plot(x, mean, color=color, marker="o", markersize=2.6, linewidth=1.3, label=label)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, linewidth=0)
    ax.axvline(1080, color=GRAY, linestyle="--", linewidth=1.0)
    ax.text(1080 * 1.15, ax.get_ylim()[1] * 0.55, "1,080 updates", color=GRAY, fontsize=7.6, rotation=90, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative optimizer updates (log scale)")
    ax.set_ylabel("Training pinball loss")
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    labels = ["1 yr\n1,080 upd.", "15 yr\n1,080 upd.", "15 yr\n16,050 upd."]
    colors = [GREEN, BLUE, VERMILLION]
    packs = [eq1, eq15, fe15]

    ax = axes[1]
    picp = [[m["picp"] for m in p[2]] for p in packs]
    means = [np.mean(v) for v in picp]
    sds = [np.std(v, ddof=1) for v in picp]
    ax.bar(range(3), means, yerr=sds, color=colors, width=0.62, capsize=3, error_kw={"elinewidth": 0.9})
    ax.axhline(0.90, color=GRAY, linestyle="--", linewidth=1.0)
    ax.text(2.42, 0.902, "nominal", color=GRAY, fontsize=7.2, va="bottom", ha="right")
    ax.set_xticks(range(3), labels, fontsize=7.4)
    ax.set_ylabel("Raw test PICP")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    ax.text(-0.20, 1.04, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    ax = axes[2]
    mpiw = [[m["mpiw"] for m in p[2]] for p in packs]
    means = [np.mean(v) for v in mpiw]
    sds = [np.std(v, ddof=1) for v in mpiw]
    ax.bar(range(3), means, yerr=sds, color=colors, width=0.62, capsize=3, error_kw={"elinewidth": 0.9})
    ax.set_xticks(range(3), labels, fontsize=7.4)
    ax.set_ylabel("Raw test MPIW (log scale units)")
    ax.set_ylim(0.0, 0.66)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    ax.text(-0.20, 1.04, "c", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    figure.savefig(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
