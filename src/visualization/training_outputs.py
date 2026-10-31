"""Render publication tables and figures for completed, predeclared sensitivity fits.

Shared by the figure stage; consumes the frozen training summary and cache
and writes through injected ``rows``/``macros``/``save`` callbacks so callers
keep control of output locations.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results'

def num(v, d=3):
    return f'{v:.{d}f}'

def ci(v, d=3):
    return '[' + ', '.join(num(x, d) for x in v) + ']'

def pm(mean, sd):
    return num(mean) + r' $\pm$ ' + num(sd)


def training_outputs(tr, macros, rows, save):
    f = pd.DataFrame(tr['conditions']); assert not tr['missing'] or tr.get('preview', False)
    def condition(group, variant='final', level=.9, method='raw'):
        z = f[(f.group == group) & (f.variant == variant) & np.isclose(f.nominal, level) & (f.method == method)]
        assert len(z) == 1 and int(z.iloc[0].n_seeds) == 3, (group, variant, level, method)
        return z.iloc[0]
    temporal = []; window_years = [1985, 1990, 1991, 1992, 1993, 1994, 1995]
    raw_cache = np.load(BASE / 'cache/primary_raw.npz'); dates = raw_cache['dates']
    for year in window_years:
        raw = condition(f'temporal_wy{year}'); cal = condition(f'temporal_wy{year}', method='global')
        effect = tr['archived_15yr_minus_alternative_year'][str(year)]['nse']
        selected = (dates >= np.datetime64(f'{year-1}-10-01')) & (dates <= np.datetime64(f'{year}-09-30'))
        valid_labels = int(raw_cache['mask'][:, selected].sum())
        temporal.append([year, f'{valid_labels:,}', pm(raw.median_basin_nse_mean, raw.median_basin_nse_sd), pm(raw.picp_mean, raw.picp_sd),
                         pm(cal.picp_mean, cal.picp_sd), num(raw.winkler_mean), num(effect['estimate']) + ' ' + ci(effect['basin_ci95'])])
    rows('revision_temporal_rows.tex', temporal)
    vals = f[(f.family == 'temporal') & (f.variant == 'final') & np.isclose(f.nominal, .9) & (f.method == 'raw')]
    for suffix, col, op in [('NSEMin', 'median_basin_nse_mean', 'min'), ('NSEMax', 'median_basin_nse_mean', 'max'),
                            ('PICPMin', 'picp_mean', 'min'), ('PICPMax', 'picp_mean', 'max')]:
        macros['RevisionTemporal' + suffix] = num(getattr(vals[col], op)())
    for family, prefix in [('validation', 'RevisionValidation'), ('validation_equal', 'RevisionValidationEqual'), ('tcn', 'RevisionTCN')]:
        variant = 'selected' if family.startswith('validation') else 'final'
        v = tr['endpoint_comparisons'][f'{family}_{variant}_0.9_raw_nse']
        macros[prefix + 'NSEDelta'] = num(v['estimate']); macros[prefix + 'NSECI'] = ci(v['basin_ci95'])
    valid = []
    for group, years, budget in [('validation_1yr', 1, '30 epochs'), ('validation_15yr', 15, '30 epochs'), ('validation_equal_15yr', 15, '1,080 updates')]:
        for variant, label in [('final', 'Final'), ('selected', 'Validation minimum')]:
            raw = condition(group, variant); cal = condition(group, variant, method='global')
            valid.append([years, budget, label, pm(raw.median_basin_nse_mean, raw.median_basin_nse_sd),
                          pm(raw.picp_mean, raw.picp_sd), pm(cal.picp_mean, cal.picp_sd), num(cal.winkler_mean)])
    rows('revision_validation_rows.tex', valid)
    multi = []
    for level in [.5, .8, .9, .95, .99]:
        for years in [1, 15]:
            raw = condition(f'multilevel_{years}yr', level=level); cal = condition(f'multilevel_{years}yr', level=level, method='global')
            multi.append([years, int(level * 100), pm(raw.picp_mean, raw.picp_sd), pm(cal.picp_mean, cal.picp_sd), num(raw.winkler_mean), num(cal.winkler_mean)])
    rows('revision_multilevel_rows.tex', multi)
    tcn = []
    for years in [1, 15]:
        raw = condition(f'tcn_{years}yr'); cal = condition(f'tcn_{years}yr', method='global')
        tcn.append([years, pm(raw.median_basin_nse_mean, raw.median_basin_nse_sd), pm(raw.picp_mean, raw.picp_sd), num(raw.mpiw_mean), num(raw.winkler_mean),
                    pm(cal.picp_mean, cal.picp_sd), num(cal.winkler_mean)])
    rows('revision_tcn_rows.tex', tcn)
    if tr.get('preview', False):
        rows('revision_memory_rows.tex', [[r'\multicolumn{6}{c}{LAYOUT PREVIEW: memory fits still running}']])
        for name in ['RevisionMemoryShortDelta', 'RevisionMemoryShortCI', 'RevisionMemoryLongDelta', 'RevisionMemoryLongCI']:
            macros[name] = 'PENDING'
    else:
        memory_rows = []
        for years, label in [(1, 'Short record'), (15, 'Common long record')]:
            for seq in [180, 365]:
                group = f'memory{seq}_{years}yr' if (years == 15 or seq == 365) else 'temporal_wy1995'
                raw = condition(group); cal = condition(group, method='global')
                memory_rows.append([label, seq, pm(raw.median_basin_nse_mean, raw.median_basin_nse_sd), pm(raw.picp_mean, raw.picp_sd),
                                    pm(cal.picp_mean, cal.picp_sd), num(cal.winkler_mean)])
        rows('revision_memory_rows.tex', memory_rows)
        for years, prefix in [(1, 'RevisionMemoryShort'), (15, 'RevisionMemoryLong')]:
            effect = tr['matched_target_memory_effects_365_minus_180'][f'{years}yr_raw']['nse']
            macros[prefix + 'Delta'] = num(effect['estimate'], 4); macros[prefix + 'CI'] = ci(effect['basin_ci95'], 4)
    # Time-window composition.
    cf = pd.read_csv(BASE / 'tables/coverage_cells.csv'); bf = pd.read_csv(BASE / 'tables/basin_metrics.csv', dtype={'basin': str})
    baseline = [bf[(bf.year == 15) & (bf.method == 'raw')].groupby('seed').nse.median().mean(),
                cf[(cf.year == 15) & (cf.method == 'raw') & (cf.dimension == 'pooled')].picp.mean(),
                cf[(cf.year == 15) & (cf.method == 'raw') & (cf.dimension == 'pooled')].winkler.mean()]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, col, sdcol, title, ref in zip(axes, ['median_basin_nse_mean', 'picp_mean', 'winkler_mean'],
        ['median_basin_nse_sd', 'picp_sd', 'winkler_sd'], ['Median basin NSE', 'Raw coverage', 'Raw Winkler (log-flow)'], baseline):
        z = [condition(f'temporal_wy{year}') for year in window_years]
        ax.errorbar(range(7), [r[col] for r in z], yerr=[r[sdcol] for r in z], fmt='o', capsize=3, color='#345b78')
        ax.axhline(ref, color='#963e39', ls='--', label='Archived 15-year reference'); ax.set_xticks(range(7), window_years, rotation=45)
        ax.set_xlabel('One-year training water year'); ax.set_ylabel(title)
    axes[0].legend(frameon=False, fontsize=7); save(fig, 'revision_training_years')
    # Independent validation curves in a common de-standardized coordinate.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    for ax, length in zip(axes[:2], [1, 15]):
        records = []
        for seed in [42, 123, 456]:
            meta = json.loads((BASE / f'tables/training_validation_{length}yr_s{seed}.json').read_text())
            h = meta['history']; scale = meta['normalization']['target_std']
            records.append(([v['updates'] for v in h], [v['train_loss'] * scale for v in h], [v['validation_loss'] * scale for v in h]))
        for pos, label, color in [(1, 'Training', '#345b78'), (2, 'Independent validation', '#963e39')]:
            vv = np.array([r[pos] for r in records]); xx = records[0][0]
            ax.plot(xx, vv.mean(0), label=label, color=color); ax.fill_between(xx, vv.mean(0) - vv.std(0, ddof=1), vv.mean(0) + vv.std(0, ddof=1), alpha=.15, color=color)
        ax.set_title(f'{length}-year training labels', fontsize=10); ax.set_xlabel('Cumulative optimizer updates'); ax.set_ylabel('Pinball loss (log-flow)')
    categories = []; vv = []
    for group, years in [('validation_1yr', '1 yr'), ('validation_equal_15yr', '15 yr'), ('validation_15yr', '15 yr')]:
        metas = [json.loads((BASE / f'tables/training_{group}_s{seed}.json').read_text()) for seed in [42, 123, 456]]
        updates = {m['optimizer_updates'] for m in metas}
        assert len(updates) == 1, (group, sorted(updates))
        categories.append(f'{years}\n{updates.pop():,} updates')
        vv.append([m['history'][-1]['validation_loss'] * m['normalization']['target_std'] for m in metas])
    axes[2].errorbar(range(3), np.mean(vv, axis=1), yerr=np.std(vv, axis=1, ddof=1), fmt='o', capsize=4, color='#963e39'); axes[2].set_xticks(range(3), categories, fontsize=8)
    axes[2].set_ylabel('Final independent-validation loss'); axes[0].legend(frameon=False, fontsize=7); save(fig, 'revision_validation_curves')
    # Multiple nominal levels; the head is re-trained rather than interpolated.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained'); levels = [.5, .8, .9, .95, .99]
    for method, ls in [('raw', '--'), ('global', '-')]:
        for length, color in [(1, '#963e39'), (15, '#345b78')]:
            z = [condition(f'multilevel_{length}yr', level=l, method=method) for l in levels]
            axes[0].errorbar(levels, [r.picp_mean for r in z], yerr=[r.picp_sd for r in z], ls=ls, marker='o', capsize=2, color=color, label=f'{length} yr, ' + ('raw' if method == 'raw' else 'CQR'))
            axes[1].plot(levels, [r.winkler_mean for r in z], ls=ls, marker='o', color=color)
    axes[0].plot([.45, 1], [.45, 1], color='#777777', lw=.8); axes[0].set_xlabel('Nominal coverage'); axes[0].set_ylabel('Achieved coverage'); axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel('Nominal coverage'); axes[1].set_ylabel('Winkler (compare within nominal level)'); save(fig, 'revision_multilevel_scarcity')
    # Time heterogeneity under the same fitted intervals.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for row, years in enumerate([1, 15]):
        for method, color in [('global', '#345b78'), ('predicted', '#b66d35')]:
            for ax, dim in zip(axes[row], ['season', 'water_year']):
                z = cf[(cf.year == years) & (cf.method == method) & (cf.dimension == dim)].groupby('group').picp.agg(['mean', 'std'])
                if dim == 'season':
                    z = z.reindex(['DJF', 'MAM', 'JJA', 'SON'])
                ax.errorbar(range(len(z)), z['mean'], yerr=z['std'], marker='o', capsize=2, label=method, color=color); ax.set_xticks(range(len(z)), z.index, rotation=45 if dim == 'water_year' else 0)
                ax.axhline(.9, color='#777777', ls='--'); ax.set_ylim(.75, 1.); ax.set_ylabel(f'{years}-year pooled PICP')
    axes[0, 0].legend(frameon=False, fontsize=8); axes[1, 0].set_xlabel('Season'); axes[1, 1].set_xlabel('Test water year (2009 is partial)'); save(fig, 'revision_time_coverage')
