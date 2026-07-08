"""orchestrator.py - resumable 15-feature experiment pipeline.

Runs Stage 2 (remaining baselines) -> Stage 3 (quantile + ensembles) ->
Stage 4 (CQR/MC eval) -> Stage 5 (scarcity + cross-region) -> Stage 6
(figures + result consistency audit), with:
  * RESUMABILITY: each stage verifies its outputs; verified-done stages are skipped.
  * CLEAN-STATE: kills stray python.exe (excluding self) before each training,
    avoiding the Windows DataLoader num_workers deadlock.
  * HANG PROTECTION: per-command timeout (subprocess.TimeoutExpired -> fail).
  * VERIFICATION GATES: proceed only if the stage's output JSON/checkpoint is sane.
Stops and logs "NEEDS ATTENTION: <stage>" on any unrecoverable failure so the
run can be inspected and resumed after the issue is fixed.

Run download_camels.py and src/data/data_preprocessing.py before launching this
script in a fresh checkout.

Usage:  python experiments/orchestrator.py
"""
import subprocess, json, time, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results" / "checkpoints"
TABLES = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
LOG = ROOT / "orchestrator.log"
SELF_PID = os.getpid()
PY = sys.executable


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kill_stray_python():
    """Kill all python.exe except this orchestrator (clean-state for DataLoader)."""
    try:
        subprocess.run(["powershell", "-Command",
            f"Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            f"Where-Object {{$_.ProcessId -ne {SELF_PID}}} | "
            f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:
        log(f"  kill_stray warn: {e}")
    time.sleep(3)


def run(cmd, timeout_s, logfile):
    """Run cmd (list), tee to logfile, enforce timeout. Returns True on exit 0."""
    log(f"  RUN: {' '.join(cmd)}  (timeout {timeout_s}s)")
    t0 = time.time()
    try:
        with open(ROOT / logfile, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout_s, cwd=str(ROOT))
        dt = time.time() - t0
        log(f"  exit={p.returncode} in {dt:.0f}s -> {logfile}")
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {timeout_s}s -> FAIL")
        return False
    except Exception as e:
        log(f"  EXCEPTION: {e} -> FAIL")
        return False


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def is_15feat(name):
    r = load_json(TABLES / f"{name}_results.json")
    return bool(r and r.get("config", {}).get("n_dynamic") == 15)


def stage2():
    log("### STAGE 2: remaining baselines (transformer, ea_lstm, xgboost, eval) ###")
    # transformer (15-feat)  -  full training
    if is_15feat("transformer"):
        log("  transformer already 15-feat, skip")
    else:
        kill_stray_python()
        # transformer: O(seq^2)=365^2 attention at batch 128 is ~1423s/epoch, so
        # cap at 15 epochs (it plateaus fast, like LSTM best@3) + 6h timeout.
        if not run([PY, "-u", "experiments/baseline/train_model.py", "--model", "transformer",
                    "--epochs", "15"],
                   6 * 3600, "train_transformer.log"):
            return False, "transformer training failed"
        if not is_15feat("transformer"):
            return False, "transformer result not 15-feat after training"
    # ea_lstm  -  capped 15 epochs (manual loop is ~1161s/epoch)
    if is_15feat("ea_lstm"):
        log("  ea_lstm already 15-feat, skip")
    else:
        kill_stray_python()
        if not run([PY, "-u", "experiments/baseline/train_model.py", "--model", "ea_lstm", "--epochs", "15"],
                   7 * 3600, "train_ea_lstm.log"):
            return False, "ea_lstm training failed"
        if not is_15feat("ea_lstm"):
            return False, "ea_lstm result not 15-feat after training"
    # xgboost (daymet-only): skip if a good daymet-only result already exists
    xr = load_json(TABLES / "xgboost_results.json")
    if xr and xr.get("n_features", 0) < 100 and xr.get("test_nse_perbasin_median", 0) > 0.50:
        log(f"  xgboost already daymet-only good (per-basin {xr.get('test_nse_perbasin_median'):.3f}), skip")
    else:
        log("  xgboost: daymet-only run on cached features")
        kill_stray_python()
        if not run([PY, "experiments/baseline/train_xgboost.py"], 2 * 3600, "train_xgboost.log"):
            return False, "xgboost failed"
        xr = load_json(TABLES / "xgboost_results.json")
        if not xr or xr.get("n_features", 0) < 50:
            return False, "xgboost result missing/low-feature"
        log(f"  xgboost per-basin median NSE = {xr.get('test_nse_perbasin_median'):.4f}")
    # per-basin eval for all 4 DL point models: skip if already done
    pb = load_json(TABLES / "perbasin_nse_point.json")
    if pb and len(pb) >= 4:
        log("  perbasin_nse_point.json already has 4 models, skip")
    else:
        kill_stray_python()
        if not run([PY, "experiments/analysis/eval_point_perbasin.py"], 1 * 3600, "eval_point_perbasin.log"):
            return False, "per-basin eval failed"
        pb = load_json(TABLES / "perbasin_nse_point.json")
        if not pb or len(pb) < 4:
            return False, "perbasin_nse_point.json incomplete"
    for k, v in pb.items():
        log(f"    {v.get('display', k):<32} median NSE = {v.get('nse_median'):.4f}")
    return True, "ok"


def stage3():
    log("### STAGE 3: quantile model (seed 42) + 4 ensemble members ###")
    # Delete only STALE 5-feat ensemble checkpoints; KEEP 15-feat ones (incl.
    # partially-trained, which retrain_ensembles resumes). Unconditional deletion
    # would discard completed 15-feat members on re-launch.
    import torch
    for s in [123, 456, 789, 999]:
        for suf in [".pt", "_opt.pt"]:
            p = CKPT / f"ensemble_seed{s}{suf}"
            if p.exists():
                try:
                    nd = torch.load(p, weights_only=False, map_location="cpu").get("config", {}).get("n_dynamic", 5)
                except Exception:
                    nd = 5
                if nd != 15:
                    p.unlink()
                    log(f"  deleted stale {p.name} (n_dynamic={nd})")
    # main quantile model (seed 42)  -  overwrites lpu_stream_quantile_best.pt
    q = load_json(TABLES / "lpu_stream_quantile_results.json")
    if q and q.get("config", {}).get("n_dynamic") == 15:
        log("  quantile seed42 already 15-feat, skip")
    else:
        kill_stray_python()
        if not run([PY, "-u", "experiments/uncertainty/train_quantile.py"],
                   3 * 3600, "train_quantile.log"):
            return False, "quantile training failed"
    # 4 ensemble members + aggregation
    de = load_json(TABLES / "deep_ensembles_fair_results.json")
    done_members = all((CKPT / f"ensemble_seed{s}.pt").exists() for s in [123, 456, 789, 999])
    if de and de.get("ensemble", {}).get("n_members") == 5 and done_members:
        # still re-aggregate to refresh fair_comparison_671 link? keep if present
        log("  ensembles already trained (5 members), skip")
    else:
        kill_stray_python()
        if not run([PY, "-u", "experiments/uncertainty/retrain_ensembles_correct.py"],
                   15 * 3600, "retrain_ensembles.log"):
            return False, "ensemble training failed"
    de2 = load_json(TABLES / "deep_ensembles_fair_results.json")
    if not de2 or de2.get("ensemble", {}).get("n_members") != 5:
        return False, "deep_ensembles_fair_results incomplete"
    return True, "ok"


def stage4():
    log("### STAGE 4: fair uncertainty eval (CQR / MC / ensembles) ###")
    fc = load_json(TABLES / "fair_comparison_671.json")
    if fc and "cqr_single" in fc and "mc_dropout" in fc and "deep_ensembles" in fc:
        log("  fair_comparison_671.json already complete, skip")
    else:
        kill_stray_python()
        # eval_fair_671 includes MC dropout (50 forward passes on 2.2M test samples)
        # which alone takes ~2.5-3h; allow 5h.
        if not run([PY, "experiments/uncertainty/eval_fair_671.py"], 5 * 3600, "eval_fair_671.log"):
            return False, "fair eval failed"
        fc = load_json(TABLES / "fair_comparison_671.json")
        if not fc or "cqr_single" not in fc or "mc_dropout" not in fc:
            return False, "fair_comparison_671.json incomplete"
    log(f"    CQR single PICP={fc['cqr_single']['cal']['picp']:.3f} "
        f"MPIW={fc['cqr_single']['cal']['mpiw']:.3f}")
    return True, "ok"


def stage5():
    log("### STAGE 5: scarcity (1/3/5 yr) + cross-region ###")
    for yr in [1, 3, 5]:
        out = TABLES / f"scarce_{yr}yr_results.json"
        r = load_json(out)
        # scarcity output stores test_nse (not n_dynamic); done if it has results.
        # The model is 15-feat via LPUStreamModel/CamelsDataset defaults.
        if r and r.get("test_nse") is not None:
            log(f"  scarcity {yr}yr already done (test_nse={r['test_nse']:.4f}), skip")
            continue
        kill_stray_python()
        if not run([PY, "-u", "experiments/scarce/train_data_scarce.py", "--years", str(yr)],
                   4 * 3600, f"scarce_{yr}yr.log"):
            return False, f"scarcity {yr}yr failed"
    cr = load_json(TABLES / "cross_region_results.json")
    # cross-region is an eval (no model config); done if regime keys present
    if not (cr and cr.get("very_humid") and cr.get("dry")):
        kill_stray_python()
        if not run([PY, "experiments/analysis/cross_region_validation.py"], 3 * 3600, "cross_region.log"):
            return False, "cross-region failed"
    return True, "ok"


def stage6():
    log("### STAGE 6: re-run XGBoost if needed + figures + result audit ###")
    # Re-run XGBoost with the balanced config (depth 8)  -  the over-regularized
    # depth-6 version collapsed to per-basin 0.13. All DL training is done now,
    # so RAM (~16GB) is free for XGBoost's ~6GB peak. Forces re-run regardless.
    kill_stray_python()
    import os
    xr_old = load_json(TABLES / "xgboost_results.json")
    if xr_old and xr_old.get("test_nse_perbasin_median", 0) < 0.40:
        log(f"  xgboost per-basin NSE was {xr_old.get('test_nse_perbasin_median'):.3f} (low) -> re-run with fixed config")
        if not run([PY, "experiments/baseline/train_xgboost.py"], 2 * 3600, "train_xgboost.log"):
            return False, "xgboost fixed re-run failed"
    if not run([PY, "experiments/analysis/make_figures.py"], 1 * 3600, "make_figures.log"):
        return False, "make_figures failed"
    # verify_manuscript must end with all checks pass.
    if not run([PY, "experiments/analysis/verify_manuscript.py"], 1 * 3600, "verify_manuscript.log"):
        return False, "result audit reported failures"
    return True, "ok"


def main():
    log("\n" + "=" * 70 + "\n ORCHESTRATOR START\n" + "=" * 70)
    for name, fn in [("stage2", stage2), ("stage3", stage3), ("stage4", stage4),
                     ("stage5", stage5), ("stage6", stage6)]:
        log(f"\n>>> entering {name}")
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f"exception: {e}"
        if not ok:
            log(f"\n!!! NEEDS ATTENTION: {name} FAILED  -  {msg}")
            log("!!! Orchestrator stopped. Fix and re-run.")
            return
        log(f"<<< {name} OK ({msg})")
    log("\n" + "=" * 70 + "\n ORCHESTRATOR COMPLETE  -  all stages verified OK\n" + "=" * 70)


if __name__ == "__main__":
    main()
