"""Rebuild quality-flag, direct-contrast, coverage and area diagnostics."""
from pathlib import Path
from collections import Counter
import csv
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.protocol import interval_arrays, project_nonempty, dump


def main():
    tables = ROOT / 'results/tables'
    raw = ROOT / 'data/raw/camels_us/basin_dataset_public_v1p2'
    effects = pd.read_csv(tables / 'paired_basin_effects.csv', dtype={'basin': str})
    primary = set(effects.basin)
    names = pd.read_csv(ROOT / 'data/raw/camels_us/camels_name.txt', sep=';', dtype=str).set_index('gauge_id')
    with np.load(ROOT / 'results/predictions/corrected_full_671/evaluation_index.npz') as index:
        full_basins = set(index['basin_ids'].tolist())
    counts = {k: Counter() for k in ['primary_cal', 'primary_test', 'full_cal', 'full_test']}
    flags = {}
    for path in sorted((raw / 'usgs_streamflow').rglob('*_streamflow_qc.txt')):
        basin = path.name.split('_')[0]
        if basin not in full_basins:
            continue
        for line in path.read_bytes().splitlines():
            cells = line.split()
            if len(cells) < 6:
                continue
            date = b'-'.join(cells[1:4]).decode()
            period = 'cal' if '1995-10-01' <= date <= '2000-09-30' else (
                'test' if '2000-10-01' <= date <= '2008-12-31' else None)
            if period is None:
                continue
            flag = cells[5].decode()
            counts['full_' + period][flag] += 1
            if basin in primary:
                counts['primary_' + period][flag] += 1
                if period == 'test':
                    flags[basin, date] = flag
    qc_scores = []
    for years in [1, 15]:
        values = {'all': [], 'A_only': []}
        for seed in [42, 123, 456]:
            path = ROOT / f'results/predictions/corrected_scarcity/scarcity_{years}yr_basinseed42_modelseed{seed}.npz'
            with np.load(path) as z:
                valid = z['test_mask'] > 0
                labels = np.array([flags[str(b), str(d)] for b, d in zip(z['test_basin_ids'], z['test_target_dates'])])
                assert np.array_equal(labels == 'M', ~valid)
                q = z['test_prediction_log'][valid].astype(float)
                y = z['test_target_log'][valid].astype(float)
                arr = interval_arrays(q[:, 0], q[:, 2], y)
                for label, selected in [('all', np.ones(len(y), bool)), ('A_only', labels[valid] == 'A')]:
                    values[label].append([arr['covered'][selected].mean(), arr['winkler'][selected].mean(), selected.sum()])
        for label, rows in values.items():
            rate, score, n = np.mean(rows, axis=0)
            qc_scores.append({'years': years, 'test_flags': label, 'raw_picp': rate, 'raw_winkler': score, 'observations_per_seed': int(n)})
    frame = pd.read_csv(tables / 'basin_year_sufficient_statistics.csv', dtype={'basin': str})
    frame = frame.query('year == 1 and method == "predicted" and group == "predicted_high"').copy()
    frame['huc2'] = frame.basin.map(names.huc_02)
    absolute = {}
    for cluster in ['basin', 'huc2', 'water_year']:
        v = frame.groupby(['seed', cluster])[['hits', 'n']].sum()
        hits, den = v.hits.unstack().values, v.n.unstack().values
        rng = np.random.default_rng(20261102)
        draw = rng.integers(hits.shape[1], size=(5000, hits.shape[1]))
        dist = (hits[:, draw].sum(-1) / den[:, draw].sum(-1)).mean(0)
        absolute[cluster] = {'estimate': float((hits.sum(1) / den.sum(1)).mean()),
                             'ci95': np.quantile(dist, [.025, .975]), 'clusters': hits.shape[1]}
    arch = ROOT / 'results/predictions/corrected_full_671'
    methods = {m['id']: m for m in json.loads((tables / 'method_comparison.json').read_text())['methods']}
    norm = json.loads((ROOT / 'data/metadata/normalization_stats.json').read_text())
    with np.load(arch / 'test_predictions.npz') as z:
        valid = z['mask'] > 0
        basin = z['basin_index'][valid]
        y = z['target'][valid].astype(float) * norm['target_std'] + norm['target_mean']
        spread = z['mc_dropout'][valid].astype(float) * norm['target_std'] + norm['target_mean']
    with np.load(arch / 'evaluation_index.npz') as z:
        regions = names.loc[z['basin_ids'], 'huc_02'].values
    _, region_index = np.unique(regions, return_inverse=True)
    n = np.bincount(basin)
    sufficient = {}
    for method in ['mc_dropout', 'mc_dropout_noise', 'mc_dropout_reselect']:
        if method == 'mc_dropout':
            q = spread
        else:
            with np.load(ROOT / f'results/predictions/benchmark_671/{method}_predictions.npz') as z:
                assert np.array_equal(z['test_mask'] > 0, valid)
                assert np.max(np.abs(z['target_test'][valid] - y)) < 1e-6
                q = z['test_quantiles_log'][valid].astype(float)
        radius = methods[method]['radius_log']
        lo, hi, _ = project_nonempty(q[:, 0] - radius, q[:, 2] + radius)
        score = interval_arrays(lo, hi, y)['winkler']
        assert abs(score.mean() - methods[method]['cqr']['winkler']) < 1e-10
        sufficient[method] = np.bincount(basin, weights=score)
    rng = np.random.default_rng(20261102)
    draw = rng.integers(len(n), size=(2000, len(n)))
    nr = int(region_index.max() + 1)
    spatial = rng.integers(nr, size=(2000, nr))
    rn = np.bincount(region_index, weights=n)
    contrasts = {}
    for method in ['mc_dropout_noise', 'mc_dropout_reselect']:
        delta = sufficient[method] - sufficient['mc_dropout']
        rd = np.bincount(region_index, weights=delta)
        contrasts[method] = {'winkler_difference': float(delta.sum() / n.sum()),
            'basin_ci95': np.quantile(delta[draw].sum(1) / n[draw].sum(1), [.025, .975]),
            'huc2_ci95': np.quantile(rd[spatial].sum(1) / rn[spatial].sum(1), [.025, .975])}
    area = {m.group(1) for m in re.finditer(r'^\d{2}\s+(\d{8})\s+', (raw / 'basin_size_errors_10_percent.txt').read_text(), re.M)}
    area_results = {}
    for label, v in [('all_50', effects), ('excluding_listed', effects.loc[~effects.basin.isin(area)])]:
        area_results[label] = {'n_basins': len(v), 'median_nse_delta': v.raw_nse.median(),
            'mean_raw_picp_delta': v.raw_picp.mean(), 'mean_raw_winkler_delta': v.raw_winkler.mean()}
    result = {'random_seed': 20261102, 'bootstrap_conditioning': 'fixed fitted models and historical observations',
        'absolute_coverage_draws': 5000, 'method_contrast_draws': 2000, 'quality_counts': counts,
        'quality_test_only_sensitivity': qc_scores, 'absolute_predicted_high_coverage': absolute,
        'direct_mc_contrasts': contrasts, 'area_exclusion': area_results,
        'area_primary_listed_basins': sorted(primary & area)}
    dump(tables / 'submission_diagnostics.json', result)
    generated = ROOT / 'results/drafts/generated'
    for zh in [False, True]:
        lines = [r'\section*{S25 ' + ('交付补充核验与固定模型敏感性' if zh else 'Additional fixed-model diagnostics') + '}']
        lines += [('下列诊断保留冻结模型和原校准，质量标记筛选仅改变测试评分样本，所有区间以固定模型和历史时期为条件。'
                  if zh else 'These diagnostics retain the frozen predictors and calibration, with the quality-flag restriction changing test scoring only; all intervals condition on the fitted models and historical periods.')]
        lines += [r'\begin{center}\small\begin{tabular}{lrr}\toprule',
                  ('数据与时期 & A & A:e\\' if zh else 'Cohort and period & A & A:e\\'), r'\midrule']
        for key, label in [('primary_cal', '50 cal.'), ('primary_test', '50 test'), ('full_cal', '671 cal.'), ('full_test', '671 test')]:
            lines.append(f"{label} & {counts[key]['A']:,} & {counts[key]['A:e']:,}"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{center}',
                  ('全流域测试另有17个A:<标记，均属09480000站点；按上游小于限定值记录保留其数值。主样本校准无缺测，测试缺1461日；全流域校准缺1日，测试缺1464日。'
                   if zh else 'The full test set also contains 17 A:< records at gauge 09480000, retained at the archived value with its less-than qualifier. Missing counts are 0/1,461 in primary calibration/test and 1/1,464 in full calibration/test.'),
                  r'\begin{center}\small\begin{tabular}{llrr}\toprule',
                  ('训练年 & 测试样本 & Raw PICP & Winkler\\' if zh else 'Train yr & Test flags & Raw PICP & Winkler\\'), r'\midrule']
        for row in qc_scores:
            lines.append(f"{row['years']} & {row['test_flags'].replace('_', ' ')} & {row['raw_picp']:.4f} & {row['raw_winkler']:.4f}"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{center}']
        ci = {c: tuple(row['ci95']) for c, row in absolute.items()}
        cluster_label = {'basin': ('流域', 'basin'), 'huc2': ('HUC2 区域', 'HUC2 region'),
                         'water_year': ('同步水文年', 'synchronous water year')}
        est = absolute['basin']['estimate']
        segs = [f'[{lo:.4f}, {hi:.4f}]（{cluster_label[c][0]}）' if zh else
                f'[{lo:.4f}, {hi:.4f}] by {cluster_label[c][1]}' for c, (lo, hi) in ci.items()]
        if zh:
            lines.append(f'一年高预测流量组校准后覆盖率为 {est:.4f}，95\\% 区间为 ' + '、'.join(segs) + '。')
        else:
            lines.append(f'One-year calibrated high predicted-flow coverage is {est:.4f}, with 95\\% intervals '
                         + ', '.join(segs[:-1]) + ', and ' + segs[-1] + '.')
        lines += [('末水文年仅含三个月。'
                   if zh else 'The final water year contains three months.'),
                  r'\begin{center}\footnotesize\begin{tabular}{lrrr}\toprule',
                  (r'MC差值 & Winkler差 & Basin 95\% CI & HUC2 95\% CI\\' if zh else r'MC contrast & Winkler difference & Basin 95\% CI & HUC2 95\% CI\\'), r'\midrule']
        for method, row in contrasts.items():
            label = 'Noise minus spread' if method == 'mc_dropout_noise' else 'Tail minus spread'
            lo, hi = row['basin_ci95']; a, b = row['huc2_ci95']
            lines.append(f"{label} & {row['winkler_difference']:.5f} & [{lo:.5f}, {hi:.5f}] & [{a:.5f}, {b:.5f}]"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{center}',
                  ('直接差值使用配对流域/区域重采样。尾段方案同时改变选模数据和学习率调度依据，其差值反映两者的综合变化。面积敏感性排除上游分区面积差异名单中的5个主流域，保留原模型和单位换算。'
                   if zh else 'Direct contrasts use paired basin/region draws. The tail-selected procedure changes both selection data and the learning-rate scheduling signal, so its difference reflects the combined change. The area sensitivity excludes five primary basins on the upstream delineation-area discrepancy list, retaining their original fitted models and conversion rule.'),
                  r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',
                  (r'样本 & Basins & $\Delta$NSE & $\Delta$PICP & $\Delta$Winkler\\' if zh else r'Sample & Basins & $\Delta$NSE & $\Delta$PICP & $\Delta$Winkler\\'), r'\midrule']
        for label, row in area_results.items():
            lines.append(f"{label.replace('_', ' ')} & {row['n_basins']} & {row['median_nse_delta']:.4f} & {row['mean_raw_picp_delta']:.4f} & {row['mean_raw_winkler_delta']:.4f}"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{center}',
                  (r'差值为15减1年；NSE为流域配对差中位数，覆盖/评分为均值，换算采用area\_gages2。逐项数字及生成命令随源码结果包提供。'
                   if zh else r'Differences are 15 minus 1 year; NSE uses the median paired basin change, coverage/score use the mean, and conversion uses area\_gages2. Machine-readable values and the generating command accompany the source/results archive.')]
        lines = [line.replace('A:<', r'\texttt{A:<}').replace('tail-selected flow changes', 'tail-selected procedure changes') for line in lines]
        lines = [line + '\\' if '&' in line and line.endswith('\\') and not line.endswith('\\\\') else line for line in lines]
        (generated / f'revision_closure_{"zh" if zh else "en"}.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Submission diagnostics rebuilt')


if __name__ == '__main__':
    main()
