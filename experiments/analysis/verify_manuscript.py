"""Cross-check reported manuscript numbers against stored result JSONs.

If paper/manuscript.tex is available, selected text strings are also checked.
In a code-only checkout, those text checks are skipped.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'results' / 'tables'
MAN_PATH = ROOT / 'paper' / 'manuscript.tex'
MAN = MAN_PATH.read_text(encoding='utf-8') if MAN_PATH.exists() else ''


def J(name):
    return json.load(open(T / name))


def rnd(x, d=3):
    return round(x, d)


fails = []
def chk(label, stated, actual, d=3):
    ok = abs(stated - round(actual, d)) < 0.5 * 10 ** -d + 1e-9
    if not ok:
        fails.append(f"FAIL {label}: manuscript={stated} vs json={actual:.6f} (round{d}={round(actual,d)})")
    else:
        print(f"  ok  {label}: {stated} (json {round(actual,d)})")


def has(text):
    if not MAN:
        print(f"  skip text check (manuscript missing): {text[:60]!r}")
        return
    ok = text in MAN
    if not ok:
        fails.append(f"FAIL text-not-found: {text[:60]!r}")
    else:
        print(f"  ok  text present: {text[:60]!r}")


# ---- sources ----
s1, s3, s5 = J('scarce_1yr_results.json'), J('scarce_3yr_results.json'), J('scarce_5yr_results.json')
q = J('lpu_stream_quantile_results.json')
pt = J('lpu_stream_results.json')
lstm = J('lstm_results.json'); tr = J('transformer_results.json'); ea = J('ea_lstm_results.json')
xgb = J('xgboost_results.json'); clim = J('climatology_results.json'); pers = J('persistence_results.json')
mc = J('mc_dropout_results.json'); de = J('deep_ensembles_fair_results.json'); fair = J('fair_comparison_671.json')
cr = J('cross_region_results.json'); conf = J('confidence_level_results.json')
diag = J('diagnosis_1yr.json'); stab = J('stability_1yr.json')
enh = J('enhanced_robustness.json')
s1ns, s5ns = J('scarce_1yr_nostatic_results.json'), J('scarce_5yr_nostatic_results.json')
s15 = J('scarce_15yr_results.json')  # 50-basin matched 15yr (clean scarcity gradient)

print("== Table 1 (point baselines) ==")
chk("Climatology", 0.04, clim['mean_nse'], 2)
chk("Persistence", 0.47, pers['median_nse'], 2)
chk("LSTM NSE", 0.838, lstm['test_nse'])
chk("LSTM params(83K)", 83, round(lstm['n_params']/1000))
chk("Transformer NSE", 0.794, tr['test_nse'])
chk("EA-LSTM NSE", 0.865, ea['test_nse'])
chk("LPU-Stream NSE", 0.878, pt['test_nse'])
chk("LPU-Stream epoch", 17, pt['best_epoch'], 0)
chk("XGBoost NSE", 0.866, xgb['test_nse'])

print("\n== Table 2 (scarcity) ==")
for tag, s, ns in [("1yr", s1, 0.719), ("3yr", s3, 0.813), ("5yr", s5, 0.843), ("15yr", s15, 0.884)]:
    chk(f"{tag} NSE", ns, s['test_nse'])
for tag, s, pc in [("1yr", s1, 0.883), ("3yr", s3, 0.869), ("5yr", s5, 0.866), ("15yr", s15, 0.876)]:
    chk(f"{tag} PICP(CQR)", pc, s['test_calibrated']['picp'])
for tag, s, qc in [("1yr", s1, 0.341), ("3yr", s3, 0.155), ("5yr", s5, 0.096), ("15yr", s15, 0.042)]:
    chk(f"{tag} q_cal", qc, s['q_cal'])
for tag, s, mw in [("1yr", s1, 1.298), ("3yr", s3, 0.968), ("5yr", s5, 0.833), ("15yr", s15, 0.662)]:
    chk(f"{tag} MPIW", mw, s['test_calibrated']['mpiw'])
chk("1yr samples", 16750, s1['train_samples'], 0)
chk("3yr samples", 50250, s3['train_samples'], 0)
chk("5yr samples", 82300, s5['train_samples'], 0)

print("\n== Table 3 (UQ comparison, all 671) ==")
chk("MC NSE", 0.878, fair['mc_dropout']['nse'])
chk("MC PICP", 0.635, fair['mc_dropout']['raw']['picp'])
chk("MC MPIW", 0.403, fair['mc_dropout']['raw']['mpiw'])
chk("MC Winkler", 2.103, fair['mc_dropout']['raw']['winkler_score'])
chk("Ens NSE", 0.892, de['ensemble']['nse'])
chk("Ens PICP", 0.838, de['ensemble']['raw']['picp'])
chk("Ens MPIW", 0.651, de['ensemble']['raw']['mpiw'])
chk("Ens Winkler", 1.235, de['ensemble']['raw']['winkler_score'])
chk("Ens+CQR PICP", 0.847, de['ensemble']['cal']['picp'])
chk("Ens+CQR MPIW", 0.662, de['ensemble']['cal']['mpiw'])
chk("CQR NSE", 0.876, fair['cqr_single']['nse'])
chk("CQR PICP", 0.848, fair['cqr_single']['cal']['picp'])
chk("CQR MPIW", 0.713, fair['cqr_single']['cal']['mpiw'])
chk("CQR Winkler", 1.362, fair['cqr_single']['cal']['winkler_score'])

print("\n== Confidence table ==")
chk("90% PICP", 0.848, conf['90%']['picp']); chk("90% MPIW", 0.713, conf['90%']['mpiw'])
chk("95% q_cal", 0.110, conf['95%']['q_cal']); chk("95% PICP", 0.916, conf['95%']['picp'])
chk("95% MPIW", 0.879, conf['95%']['mpiw']); chk("95% Winkler", 1.800, conf['95%']['winkler'])
chk("99% q_cal", 0.432, conf['99%']['q_cal']); chk("99% PICP", 0.979, conf['99%']['picp'])
chk("99% MPIW", 1.524, conf['99%']['mpiw']); chk("99% Winkler", 3.426, conf['99%']['winkler'])

print("\n== Table 4 (cross-region) ==")
chk("V.Humid NSE", 0.875, cr['very_humid']['nse']); chk("V.Humid PICP", 0.843, cr['very_humid']['picp'])
chk("V.Humid MPIW", 0.960, cr['very_humid']['mpiw']); chk("V.Humid N", 168, cr['very_humid']['n_basins'], 0)
chk("Trans NSE", 0.794, cr['transitional']['nse']); chk("Trans PICP", 0.774, cr['transitional']['picp'])
chk("Trans N", 335, cr['transitional']['n_basins'], 0)
chk("Dry NSE", 0.812, cr['dry']['nse']); chk("Dry PICP", 0.812, cr['dry']['picp'])
chk("Dry MPIW", 0.479, cr['dry']['mpiw']); chk("Dry N", 335, cr['dry']['n_basins'], 0)
chk("V.Dry NSE", 0.820, cr['very_dry']['nse']); chk("V.Dry PICP", 0.853, cr['very_dry']['picp'])
chk("V.Dry MPIW", 0.347, cr['very_dry']['mpiw']); chk("V.Dry N", 168, cr['very_dry']['n_basins'], 0)

print("\n== Table 5 (ablation) ==")
chk("1yr with", 0.719, s1['test_nse']); chk("1yr without", 0.543, s1ns['test_nse'])
chk("5yr with", 0.843, s5['test_nse']); chk("5yr without", 0.778, s5ns['test_nse'])
chk("1yr improve%", 32.5, (s1['test_nse']-s1ns['test_nse'])/s1ns['test_nse']*100, 1)
chk("5yr improve%", 8.4, (s5['test_nse']-s5ns['test_nse'])/s5ns['test_nse']*100, 1)

print("\n== Diagnosis (1yr) ==")
chk("width_ratio overall", 0.350, diag['width_ratio_overall'])
chk("high-flow PICP", 0.504, diag['width_ratio_by_regime']['high']['picp'])
chk("low-flow PICP", 0.91, diag['width_ratio_by_regime']['low']['picp'], 2)
chk("high-flow width_ratio", 0.404, diag['width_ratio_by_regime']['high']['width_ratio'])

print("\n== CQR per-regime (1yr model) ==")
chk("CQR-global high-flow", 0.736, diag['coverage_by_regime_cqr_global']['high'], 3)
chk("CQR-predregime overall", 0.903, diag['cqr_predregime_overall']['picp'], 3)
chk("CQR-predregime high-flow", 0.791, diag['coverage_by_regime_cqr_predregime']['high'], 3)
chk("CQR-predregime normal-flow", 0.925, diag['coverage_by_regime_cqr_predregime']['normal'], 3)
chk("CQR-predregime low-flow", 0.991, diag['coverage_by_regime_cqr_predregime']['low'], 3)
chk("CQR-perregime high-flow", 0.888, diag['coverage_by_regime_cqr_perregime']['high'], 3)
chk("CQR-perregime low-flow", 0.885, diag['coverage_by_regime_cqr_perregime']['low'], 3)
chk("CQR q_cal high", 0.844, diag['cqr_perregime_q_cal']['high'], 3)
chk("CQR-global low-flow", 0.996, diag['coverage_by_regime_cqr_global']['low'], 3)
chk("CQR-global normal-flow", 0.938, diag['coverage_by_regime_cqr_global']['normal'], 3)
chk("CQR-perregime normal-flow", 0.876, diag['coverage_by_regime_cqr_perregime']['normal'], 3)
chk("CQR q_cal low", -0.034, diag['cqr_perregime_q_cal']['low'], 3)
chk("CQR q_cal normal", 0.257, diag['cqr_perregime_q_cal']['normal'], 3)
chk("15yr samples (50b)", 255650, s15['train_samples'], 0)

print("\n== Enhanced robustness checks ==")
chk("1yr calibration-window PICP", 0.853, enh['calibration_window_sensitivity']['1']['picp'], 3)
chk("5yr calibration-window PICP", 0.848, enh['calibration_window_sensitivity']['5']['picp'], 3)
chk("1yr calibration-window Winkler", 1.358, enh['calibration_window_sensitivity']['1']['winkler'], 3)
chk("5yr calibration-window Winkler", 1.362, enh['calibration_window_sensitivity']['5']['winkler'], 3)
chk("bootstrap PICP CI low", 0.842, enh['cluster_bootstrap_95ci']['picp']['ci95_low'], 3)
chk("bootstrap PICP CI high", 0.854, enh['cluster_bootstrap_95ci']['picp']['ci95_high'], 3)
chk("bootstrap MPIW CI low", 0.693, enh['cluster_bootstrap_95ci']['mpiw']['ci95_low'], 3)
chk("bootstrap MPIW CI high", 0.734, enh['cluster_bootstrap_95ci']['mpiw']['ci95_high'], 3)
chk("bootstrap Winkler CI low", 1.321, enh['cluster_bootstrap_95ci']['winkler']['ci95_low'], 3)
chk("bootstrap Winkler CI high", 1.403, enh['cluster_bootstrap_95ci']['winkler']['ci95_high'], 3)

print("\n== Stability ==")
chk("1yr NSE mean", 0.722, stab['nse_mean']); chk("1yr NSE std", 0.0017, stab['nse_std'], 4)
chk("1yr PICP mean", 0.890, stab['picp_mean']); chk("1yr PICP std", 0.0014, stab['picp_std'], 4)
chk("15yr NSE mean", 0.878, de['ensemble']['member_nse_mean'])
chk("15yr NSE std", 0.0015, de['ensemble']['member_nse_std'], 4)

print("\n== Arithmetic / text checks ==")
chk("PICP drop 16pp", 16, (s15['test_uncalibrated']['picp']-s1['test_uncalibrated']['picp'])*100, 0)
chk("NSE relative decline%", 19, (s15['test_nse']-s1['test_nse'])/s15['test_nse']*100, 0)
chk("overconfidence 65%", 65, (1-diag['width_ratio_overall'])*100, 0)
chk("high-flow overconfidence 60%", 60, (1-diag['width_ratio_by_regime']['high']['width_ratio'])*100, 0)
has("aridity-based diagnostic subsets")
has("validation loss for the point-prediction baselines")
has("predicted-regime CQR variant")
has("PICP 95\\% CI: 0.842--0.854")
has("mean NSE of $0.878")
has("RTX~5060 Ti")

print("\n" + "=" * 50)
if fails:
    print(f"!!! {len(fails)} FAILURES !!!")
    for f in fails:
        print(" ", f)
else:
    print("ALL CHECKS PASSED [OK]")
