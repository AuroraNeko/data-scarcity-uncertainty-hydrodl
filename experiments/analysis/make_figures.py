"""
Generate all publication-ready figures and update LaTeX tables.
Uses the corrected experimental data (fair Deep Ensembles, cross-region validation).
"""

import json, shutil, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.linewidth': 1.0,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# Colors
BLUE = '#2c7bb6'
RED = '#d7191c'
GREEN = '#1a9641'
ORANGE = '#fdae61'
DARK = '#333333'

# Data loaded from result JSONs to prevent figure/table drift
TABLES = PROJECT_ROOT / 'results' / 'tables'
def _j(name):
    return json.load(open(TABLES / name))

_s = {1: _j('scarce_1yr_results.json'), 3: _j('scarce_3yr_results.json'),
      5: _j('scarce_5yr_results.json'), 15: _j('scarce_15yr_results.json')}
years = [1, 3, 5, 15]
scarce_data = {
    'nse':      [_s[y]['test_nse'] for y in years],
    'picp_raw': [_s[y]['test_uncalibrated']['picp'] for y in years],
    'picp_cal': [_s[y]['test_calibrated']['picp'] for y in years],
    'q_cal':    [_s[y]['q_cal'] for y in years],
    'mpiw':     [_s[y]['test_calibrated']['mpiw'] for y in years],
}

# ============================================================
# FIGURE 1: Data Scarcity Degradation
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) NSE
ax = axes[0]
ax.plot(years, scarce_data['nse'], 'o-', color=BLUE, linewidth=2.5, markersize=9, label='LPU-Stream', zorder=3)
ax.fill_between(years, 0.5, scarce_data['nse'], alpha=0.08, color=BLUE)
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('Test NSE')
ax.set_title('(a) Point Prediction Accuracy', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.95)
ax.grid(True, alpha=0.25)
ax.annotate(f"$-{(scarce_data['nse'][-1]-scarce_data['nse'][0])/scarce_data['nse'][-1]*100:.0f}\\%$ (NSE)",
            xy=(1, scarce_data['nse'][0]), xytext=(2.5, 0.58),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5), fontsize=9, color=RED, ha='center')

# (b) PICP
ax = axes[1]
ax.plot(years, scarce_data['picp_raw'], 's-', color=RED, linewidth=2, markersize=9, label='Uncalibrated QR', zorder=3)
ax.plot(years, scarce_data['picp_cal'], 'o-', color=BLUE, linewidth=2.5, markersize=9, label='+ CQR Calibrated', zorder=3)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Target 90%')
ax.fill_between(years, 0.65, scarce_data['picp_raw'], alpha=0.08, color=RED)
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('PICP (Coverage)')
ax.set_title('(b) Uncertainty Coverage', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.65, 0.95)
ax.grid(True, alpha=0.25)
ax.annotate(f"$-{(scarce_data['picp_raw'][-1]-scarce_data['picp_raw'][0])*100:.0f}$ pp",
            xy=(1, scarce_data['picp_raw'][0]), xytext=(2.5, 0.68),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5), fontsize=9, color=RED, ha='center')

# (c) MPIW bars with q_cal annotations on the bars
ax = axes[2]
x_idx = np.arange(len(years))
bar_width = 0.55
bars = ax.bar(x_idx, scarce_data['mpiw'], width=bar_width, color=BLUE, alpha=0.7, edgecolor='white', linewidth=0.8, label='MPIW', zorder=2)
# Annotate each bar with MPIW value on top and q_cal value inside
for i, (mpiw_val, q_val) in enumerate(zip(scarce_data['mpiw'], scarce_data['q_cal'])):
    ax.text(i, mpiw_val + 0.04, f'MPIW={mpiw_val:.2f}', ha='center', fontsize=8, fontweight='bold', color=BLUE)
    ax.text(i, mpiw_val * 0.45, f'$q_{{\\mathrm{{cal}}}}$={q_val:.4f}', ha='center', fontsize=7.5, color='white', fontweight='bold')
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('MPIW (interval width)')
ax.set_title('(c) Interval Width & CQR Adjustment', fontweight='bold')
ax.set_xticks(x_idx)
ax.set_xticklabels(years)
ax.set_ylim(0, 1.8)
ax.grid(True, alpha=0.25, axis='y')
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig1_degradation.png')
fig.savefig(FIG_DIR / 'fig1_degradation.pdf')
plt.close()
print('Figure 1 saved')

# ============================================================
# FIGURE 2: Fair Method Comparison
# ============================================================
methods = ['MC Dropout', 'Deep\nEnsembles', 'Deep Ens.\n+ CQR', 'CQR\n(Ours)']
# Must match the reported Table 3 values checked by verify_manuscript.py:
# fair_comparison_671.json is the authoritative source for this figure.
# deep_ensembles_fair_results.json holds a different MC-dropout / CQR-single
# run and should not be used here, otherwise figure/table drift can occur.
_de = _j('fair_comparison_671.json')
picp_m    = [_de['mc_dropout']['raw']['picp'], _de['deep_ensembles']['raw']['picp'], _de['deep_ensembles']['cal']['picp'], _de['cqr_single']['cal']['picp']]
mpiw_m    = [_de['mc_dropout']['raw']['mpiw'], _de['deep_ensembles']['raw']['mpiw'], _de['deep_ensembles']['cal']['mpiw'], _de['cqr_single']['cal']['mpiw']]
winkler_m = [_de['mc_dropout']['raw']['winkler_score'], _de['deep_ensembles']['raw']['winkler_score'], _de['deep_ensembles']['cal']['winkler_score'], _de['cqr_single']['cal']['winkler_score']]
colors_m = [RED, ORANGE, GREEN, BLUE]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

for idx, (ax, data, ylabel, title, ylim, yoff) in enumerate([
    (axes[0], picp_m, 'PICP', 'Coverage', (0, 1.0), 0.02),
    (axes[1], mpiw_m, 'MPIW', 'Interval Width', (0, 1.5), 0.03),
    (axes[2], winkler_m, 'Winkler Score', 'Overall Quality (lower=better)', (0, 3.5), 0.08),
]):
    bars = ax.bar(methods, data, color=colors_m, edgecolor='white', width=0.55, linewidth=0.8, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_title(f'({"abc"[idx]}) {title}', fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2, axis='y', zorder=1)
    for b, v in zip(bars, data):
        ax.text(b.get_x() + b.get_width()/2, v + yoff, f'{v:.3f}', ha='center', fontsize=8.5, fontweight='bold')
    if idx == 0:
        ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
        ax.text(3.35, 0.915, 'Target 90%', fontsize=8, color=GREEN, ha='right')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2_method_comparison.png')
fig.savefig(FIG_DIR / 'fig2_method_comparison.pdf')
plt.close()
print('Figure 2 saved')

# ============================================================
# FIGURE 3: Calibration Analysis
# Fig 3 data loaded from results/tables/diagnosis_1yr.json (1-yr scarcity model)
_d = _j('diagnosis_1yr.json')
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Calibration curve
ax = axes[0]
width_bins = ['Narrow\n(Q1)', 'Mid\n(Q2)', 'Wide\n(Q3)', 'Widest\n(Q4)']
picp_bins = _d['calibration_curve_by_width_quartile']
bars = ax.bar(width_bins, picp_bins, color=BLUE, edgecolor='white', width=0.5, alpha=0.85, zorder=2)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Target 90%', zorder=1)
for i, v in enumerate(picp_bins):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Predicted Interval Width Quartile')
ax.set_ylabel('Observed PICP')
ax.set_title('(a) Coverage by Width Quartile', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.45, 0.95)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# (b) Width Ratio by flow regime
ax = axes[1]
flow_labels = ['Low Flow', 'Normal Flow', 'High Flow']
_wr = _d['width_ratio_by_regime']
width_ratios = [_wr['low']['width_ratio'], _wr['normal']['width_ratio'], _wr['high']['width_ratio']]
wr_colors = [BLUE, '#abd9e9', RED]
bars = ax.bar(flow_labels, width_ratios, color=wr_colors, edgecolor='white', width=0.5, alpha=0.85, zorder=2)
ax.axhline(y=1.0, color=DARK, linestyle='--', linewidth=1.2, alpha=0.6, label='Ideal (= 1.0)', zorder=1)
for b, v in zip(bars, width_ratios):
    label_color = RED if v < 0.9 else BLUE
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color=label_color)
ax.set_ylabel('Width Ratio (actual / benchmark)')
ax.set_title('(b) Interval Width by Flow Regime', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.4)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# (c) Coverage by flow regime across years
ax = axes[2]
years_plot = [1, 3, 5, 15]
_ay = _d['coverage_by_regime_across_years']
cov_low = [_ay['1']['low'], _ay['3']['low'], _ay['5']['low'], _s[15]['test_uncalibrated']['coverage_low_flow']]
cov_normal = [_ay['1']['normal'], _ay['3']['normal'], _ay['5']['normal'], _s[15]['test_uncalibrated']['coverage_normal_flow']]
cov_high = [_ay['1']['high'], _ay['3']['high'], _ay['5']['high'], _s[15]['test_uncalibrated']['coverage_high_flow']]
ax.plot(years_plot, cov_low, 'o-', color=BLUE, linewidth=2, markersize=8, label='Low Flow', zorder=3)
ax.plot(years_plot, cov_normal, 's-', color=ORANGE, linewidth=2, markersize=8, label='Normal Flow', zorder=3)
ax.plot(years_plot, cov_high, '^-', color=RED, linewidth=2, markersize=8, label='High Flow', zorder=3)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('PICP')
ax.set_title('(c) Coverage by Flow Regime', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xticks(years_plot)
ax.set_ylim(0.45, 0.95)
ax.grid(True, alpha=0.25, zorder=1)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig3_calibration.png')
fig.savefig(FIG_DIR / 'fig3_calibration.pdf')
plt.close()
print('Figure 3 saved')

# ============================================================
# FIGURE 4: Model Architecture and CQR Calibration
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(13.8, 5.0))
ax.set_xlim(0, 14.8)
ax.set_ylim(0, 5.85)
ax.axis('off')

INK = '#26313a'
MUTED = '#64717d'
LINE = '#59636d'
CONDITION = '#7b6682'
AMBER = '#b36b00'

def box_arch(ax, x, y, w, h, text, fc, ec='#7a8791', fs=10, weight='normal'):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.05,rounding_size=0.055',
        linewidth=0.95, edgecolor=ec, facecolor=fc, zorder=3
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight=weight, color=INK, linespacing=1.12, zorder=4)
    return {'x': x, 'y': y, 'w': w, 'h': h}

def anchor_arch(b, side):
    x, y, w, h = b['x'], b['y'], b['w'], b['h']
    return {
        'left': (x, y + h / 2),
        'right': (x + w, y + h / 2),
        'top': (x + w / 2, y + h),
        'bottom': (x + w / 2, y),
    }[side]

def outside_anchor(b, side, gap=0.055):
    x, y = anchor_arch(b, side)
    if side == 'left':
        return (x - gap, y)
    if side == 'right':
        return (x + gap, y)
    if side == 'top':
        return (x, y + gap)
    if side == 'bottom':
        return (x, y - gap)
    raise ValueError(side)

def arrow_arch(ax, a, b, color=LINE, lw=1.08, dashed=False, rad=0.0, style='-|>', alpha=1.0):
    arr = FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=9.5, linewidth=lw,
        color=color, linestyle=(0, (3, 2)) if dashed else 'solid',
        connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0,
        alpha=alpha, zorder=3.8
    )
    ax.add_patch(arr)

def elbow_arch(ax, pts, color=LINE, lw=1.02, dashed=False, style='-|>', alpha=1.0):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D(
            [a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
            linestyle=(0, (3, 2)) if dashed else 'solid', alpha=alpha, zorder=1
        ))
    arrow_arch(ax, pts[-2], pts[-1], color=color, lw=lw, dashed=dashed, style=style, alpha=alpha)

def stage_arch(ax, x, label, color):
    ax.text(x, 5.55, label, ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=color, zorder=4)
    ax.plot([x - 1.0, x + 1.0], [5.34, 5.34],
            color=color, lw=1.4, solid_capstyle='round')

for x, w, fc in [
    (0.15, 3.95, '#f4f9fd'),
    (4.30, 5.00, '#f7fbf8'),
    (9.55, 1.80, '#fff8f1'),
    (11.60, 3.05, '#f4fbf8'),
]:
    ax.add_patch(FancyBboxPatch(
        (x, 0.55), w, 4.72,
        boxstyle='round,pad=0.04,rounding_size=0.08',
        linewidth=0, facecolor=fc, zorder=0
    ))

stage_arch(ax, 2.10, '1 Static encoding', BLUE)
stage_arch(ax, 6.80, '2 Sequence encoding', GREEN)
stage_arch(ax, 10.45, '3 Quantile prediction', AMBER)
stage_arch(ax, 13.10, '4 Conformal calibration', GREEN)

static = box_arch(ax, 0.45, 3.92, 1.62, 0.66, 'Static attributes\n13 descriptors', '#fff2e6', fs=9.3)
basin_mlp = box_arch(ax, 2.32, 3.92, 1.55, 0.66, 'Basin MLP\n13 -> 64 -> 32', '#fff8ef', fs=9.0)
z = box_arch(ax, 4.55, 3.92, 1.48, 0.66, 'Embedding\n$z_b \\in \\mathbb{R}^{32}$', '#fff8ef', fs=9.2)

dynamic = box_arch(ax, 0.45, 2.24, 1.62, 0.66, 'Dynamic forcings\n$x_{t,b}$, 15 vars', '#eaf4fc', fs=9.0)
concat_t = box_arch(ax, 4.55, 2.24, 1.48, 0.66, 'Per-step input\n$[x_{t,b}, z_b]$', '#f0eef8', fs=9.0)
lstm = box_arch(ax, 6.42, 2.24, 1.42, 0.66, 'LSTM encoder\n128 units', '#edf7ef', fs=9.0)
h_state = box_arch(ax, 8.22, 2.24, 1.05, 0.66, 'State\n$h_T$', '#eef6f0', fs=9.0)
concat_h = box_arch(ax, 9.60, 2.24, 1.34, 0.66, 'Final concat\n$[h_T, z_b]$', '#f0eef8', fs=8.8)
head = box_arch(ax, 11.28, 2.24, 1.32, 0.66, 'Quantile head\n160 -> 64 -> 3', '#fff1f1', fs=8.6)
raw = box_arch(ax, 12.98, 2.24, 1.24, 0.66, 'Raw quantiles\n$q_{0.05}, q_{0.50}, q_{0.95}$', '#fff5f5', fs=8.0)

cal_pred = box_arch(ax, 11.28, 0.90, 1.32, 0.58, 'Calibration\npred. + $y$', '#eef9f5', fs=8.4)
scores = box_arch(ax, 12.98, 0.78, 1.24, 0.58, 'Scores\n$E_i$', '#eef9f5', fs=8.8)
qcal = box_arch(ax, 12.98, 1.60, 1.24, 0.50, '$q_{cal}$', '#eef9f5', fs=9.5)
cqr_adjust = box_arch(ax, 12.98, 3.14, 1.24, 0.50, 'CQR\nadjust', '#e9f7f1', fs=8.6)
interval = box_arch(ax, 12.56, 4.18, 2.08, 0.70,
                    'Calibrated interval\n$[q_{0.05}-q_{cal},\\ q_{0.95}+q_{cal}]$',
                    '#e9f7f1', fs=8.9)
loss = box_arch(ax, 9.60, 1.52, 1.34, 0.58, 'Training only\npinball loss', '#fff5f5', fs=8.3)

# Main forward / inference flow.
arrow_arch(ax, outside_anchor(static, 'right'), outside_anchor(basin_mlp, 'left'))
arrow_arch(ax, outside_anchor(basin_mlp, 'right'), outside_anchor(z, 'left'))
arrow_arch(ax, outside_anchor(dynamic, 'right'), outside_anchor(concat_t, 'left'))
arrow_arch(ax, outside_anchor(concat_t, 'right'), outside_anchor(lstm, 'left'))
arrow_arch(ax, outside_anchor(lstm, 'right'), outside_anchor(h_state, 'left'))
arrow_arch(ax, outside_anchor(h_state, 'right'), outside_anchor(concat_h, 'left'))
arrow_arch(ax, outside_anchor(concat_h, 'right'), outside_anchor(head, 'left'))
arrow_arch(ax, outside_anchor(head, 'right'), outside_anchor(raw, 'left'))
arrow_arch(ax, outside_anchor(raw, 'top'), outside_anchor(cqr_adjust, 'bottom'), color=GREEN)
arrow_arch(ax, outside_anchor(cqr_adjust, 'top'), outside_anchor(interval, 'bottom'), color=GREEN)

# Static embedding reuse.
arrow_arch(ax, outside_anchor(z, 'bottom'), outside_anchor(concat_t, 'top'),
           color=CONDITION, lw=0.96, alpha=0.92)
ax.text(5.31, 3.25, 'repeat over $t$', fontsize=8.3, color='#6c5674', ha='center')
elbow_arch(
    ax,
    [outside_anchor(z, 'right'), (10.27, 4.25), outside_anchor(concat_h, 'top')],
    color=CONDITION, lw=0.96, alpha=0.92,
)
ax.text(8.20, 4.38, 'reuse $z_b$ for final concat',
        fontsize=8.3, color='#6c5674', ha='center')

# Training and calibration-only flows.
arrow_arch(ax, outside_anchor(loss, 'right'), outside_anchor(head, 'bottom'),
           color=RED, dashed=True, lw=0.98)
arrow_arch(ax, outside_anchor(cal_pred, 'right'), outside_anchor(scores, 'left'),
           color=GREEN, dashed=True, lw=0.98)
arrow_arch(ax, outside_anchor(scores, 'top'), outside_anchor(qcal, 'bottom'),
           color=GREEN, dashed=True, lw=0.98)
elbow_arch(
    ax,
    [outside_anchor(qcal, 'right'), (14.42, 1.82), (14.42, 3.39), outside_anchor(cqr_adjust, 'right')],
    color=GREEN, lw=0.98, dashed=True,
)

arrow_arch(ax, (0.48, 0.83), (1.42, 0.83), color=LINE, lw=1.08)
ax.text(1.55, 0.83, 'forward / inference flow', va='center', fontsize=8.5, color=MUTED)
arrow_arch(ax, (0.48, 0.55), (1.42, 0.55), color=GREEN, lw=0.98, dashed=True)
ax.text(1.55, 0.55, 'calibration or training-only flow', va='center', fontsize=8.5, color=MUTED)

ax.text(7.45, 0.18,
        'Quantile model: 104,099 trainable parameters; CQR is post-hoc and does not retrain the network',
        ha='center', fontsize=9, color=MUTED)

fig.savefig(FIG_DIR / 'fig4_architecture.png')
fig.savefig(FIG_DIR / 'fig4_architecture.pdf')
plt.close()
print('Figure 4 saved')

# ============================================================
# FIGURE 5: Cross-Region Validation
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))

_cr = _j('cross_region_results.json')
_fair = _j('fair_comparison_671.json')
regions = ['Very Humid\n(Q1)', 'Transitional\n(Q2-Q3)', 'Dry/\nSemi-arid', 'Very Dry\n(Q4)', 'All\nCAMELS']
_rg = [_cr['very_humid'], _cr['transitional'], _cr['dry'], _cr['very_dry'],
       {'nse': _fair['cqr_single']['nse'], 'picp': _fair['cqr_single']['cal']['picp']}]
nse_r  = [d['nse'] for d in _rg]
picp_r = [d['picp'] for d in _rg]
colors_r = ['#0571b0', '#92c5de', '#f4a582', '#ca0020', '#555555']  # NSE per-regime (aridity colormap)

x = np.arange(len(regions))
w = 0.35

bars1 = ax.bar(x - w/2, nse_r, w, color=colors_r, edgecolor='#333', linewidth=0.5, label='NSE', alpha=0.85, zorder=2)
ax_twin = ax.twinx()
# PICP in GREEN so it cannot be confused with the (blue) humid-regime NSE bar.
bars2 = ax_twin.bar(x + w/2, picp_r, w, color=GREEN, edgecolor='#333', linewidth=0.5, label='PICP', alpha=0.8, zorder=2)
ax_twin.axhline(y=0.90, color=DARK, linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=9)
ax.set_ylabel('NSE', color=DARK)
ax_twin.set_ylabel('PICP', color=GREEN)
ax.set_title('Aridity-Based Validation', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y', zorder=1)
ax.set_ylim(0.6, 1.0)
ax_twin.set_ylim(0.6, 1.0)

for i in range(len(regions)):
    ax.text(i - w/2, nse_r[i] + 0.012, f'{nse_r[i]:.3f}', ha='center', fontsize=8.5, fontweight='bold')
    ax_twin.text(i + w/2, picp_r[i] + 0.012, f'{picp_r[i]:.3f}', ha='center', fontsize=8.5, color=GREEN, fontweight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig5_cross_region.png')
fig.savefig(FIG_DIR / 'fig5_cross_region.pdf')
plt.close()
print('Figure 5 saved')

# ============================================================
# FIGURE 6: Multi-Seed Stability + Ablation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# (a) Multi-seed stability chart, simplified
ax = axes[0]
# Fig 6a data loaded from results/tables/stability_1yr.json (1-yr init stability)
_st = _j('stability_1yr.json')
seeds = ['Seed 42', 'Seed 123', 'Seed 456', 'Mean']
nse_s = [r['nse'] for r in _st['runs']] + [_st['nse_mean']]
picp_s = [r['picp_cal'] for r in _st['runs']] + [_st['picp_mean']]

x = np.arange(len(seeds))
w = 0.28
bars1 = ax.bar(x - w/2, nse_s, w, color=BLUE, edgecolor='white', alpha=0.85, label='NSE', zorder=2)
ax_twin = ax.twinx()
bars2 = ax_twin.bar(x + w/2, picp_s, w, color=RED, edgecolor='white', alpha=0.85, label='PICP (CQR)', zorder=2)

ax.set_xticks(x)
ax.set_xticklabels(seeds, fontsize=9)
ax.set_ylabel('NSE', color=BLUE)
ax_twin.set_ylabel('PICP (CQR)', color=RED)
ax.set_title('(a) Multi-Seed Stability (1-yr)', fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax_twin.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# Center-aligned values above each bar
for i in range(len(seeds)):
    ax.text(i - w/2, nse_s[i] + 0.025, f'{nse_s[i]:.3f}', ha='center', fontsize=7.5, fontweight='bold', color=BLUE)
    ax_twin.text(i + w/2, picp_s[i] + 0.025, f'{picp_s[i]:.3f}', ha='center', fontsize=7.5, fontweight='bold', color=RED)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

# (b) Static embedding ablation with bracket-style delta annotations
ax = axes[1]
methods = ['1-yr\nw/ static', '1-yr\nw/o static', '5-yr\nw/ static', '5-yr\nw/o static']
nse_ab = [_s[1]['test_nse'], _j('scarce_1yr_nostatic_results.json')['test_nse'],
          _s[5]['test_nse'], _j('scarce_5yr_nostatic_results.json')['test_nse']]
colors_ab = [BLUE, '#a6dba0', '#2c7bb6', '#92c5de']
pair_labels = ['1-year', '5-year']

bars = ax.bar(range(len(methods)), nse_ab, color=colors_ab, edgecolor='white', width=0.6, alpha=0.85, zorder=2)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel('NSE')
ax.set_title('(b) Static Embedding Ablation', fontweight='bold')
ax.set_ylim(0.4, 0.95)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# Bracket-style delta annotations for each pair
for pair_idx, label in zip([0, 2], pair_labels):
    y1, y2 = nse_ab[pair_idx], nse_ab[pair_idx + 1]
    delta = y1 - y2
    # Use matplotlib annotation bracket
    ax.annotate('', xy=(pair_idx + 0.3, max(y1, y2) + 0.04), xytext=(pair_idx + 0.7, max(y1, y2) + 0.04),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5), zorder=3)
    ax.text(pair_idx + 0.5, max(y1, y2) + 0.06, f'+{delta:.3f}', ha='center', fontsize=8.5, color=GREEN, fontweight='bold')

for i, v in enumerate(nse_ab):
    va = 'bottom' if i in [0, 2] else 'top'
    y_pos = v + 0.015 if i in [0, 2] else v - 0.025
    ax.text(i, y_pos, f'{v:.3f}', ha='center', fontsize=8.5, fontweight='bold', va=va)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig6_stability_ablation.png')
fig.savefig(FIG_DIR / 'fig6_stability_ablation.pdf')
plt.close()
print('Figure 6 saved')

# ============================================================
# FIGURE 7: Robustness and Sensitivity Checks
# ============================================================
_er = _j('enhanced_robustness.json')
_diag = _j('diagnosis_1yr.json')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) calibration-window sensitivity on the full 671-basin model
ax = axes[0]
cal_years = np.array([1, 2, 3, 4, 5])
sens = _er['calibration_window_sensitivity']
picp_w = np.array([sens[str(y)]['picp'] for y in cal_years])
wink_w = np.array([sens[str(y)]['winkler'] for y in cal_years])
ax.plot(cal_years, picp_w, 'o-', color=BLUE, linewidth=2.4, markersize=7, label='PICP', zorder=3)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.3, alpha=0.65, label='Target 90%', zorder=1)
ax.set_xlabel('Calibration Window (years)')
ax.set_ylabel('PICP', color=BLUE)
ax.tick_params(axis='y', labelcolor=BLUE)
ax.set_ylim(0.83, 0.91)
ax.set_xticks(cal_years)
ax.grid(True, alpha=0.22, zorder=1)
for xval, yval in zip(cal_years, picp_w):
    ax.text(xval, yval + 0.0022, f'{yval:.3f}', ha='center', fontsize=8, color=BLUE, fontweight='bold')
ax2 = ax.twinx()
ax2.plot(cal_years, wink_w, 's-', color=RED, linewidth=2.0, markersize=6, label='Winkler', zorder=3)
ax2.set_ylabel('Winkler Score', color=RED)
ax2.tick_params(axis='y', labelcolor=RED)
ax2.set_ylim(1.32, 1.39)
ax.set_title('(a) Calibration-Window Sensitivity', fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc='lower right')

# (b) deployable predicted-regime CQR under the 1-year scarcity setting
ax = axes[1]
regimes = ['Low', 'Normal', 'High']
raw = [_diag['width_ratio_by_regime'][k]['picp'] for k in ['low', 'normal', 'high']]
global_cqr = [_diag['coverage_by_regime_cqr_global'][k] for k in ['low', 'normal', 'high']]
pred_cqr = [_diag['coverage_by_regime_cqr_predregime'][k] for k in ['low', 'normal', 'high']]
oracle_cqr = [_diag['coverage_by_regime_cqr_perregime'][k] for k in ['low', 'normal', 'high']]
x = np.arange(len(regimes))
w = 0.19
series = [
    (raw, 'Raw QR', RED),
    (global_cqr, 'Global CQR', BLUE),
    (pred_cqr, 'Pred-regime CQR', ORANGE),
    (oracle_cqr, 'Observed-regime CQR', GREEN),
]
for i, (vals, label, color) in enumerate(series):
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, vals, width=w, label=label, color=color,
                  edgecolor='white', linewidth=0.7, alpha=0.86, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f'{v:.2f}',
                ha='center', va='bottom', fontsize=7.1,
                fontweight='bold', color=color)
ax.axhline(y=0.90, color=DARK, linestyle='--', linewidth=1.2, alpha=0.55, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(regimes)
ax.set_ylabel('PICP by Observed Flow Regime')
ax.set_ylim(0.45, 1.05)
ax.set_title('(b) Conditional Calibration (1-yr)', fontweight='bold')
ax.grid(True, alpha=0.22, axis='y', zorder=1)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=True)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig7_robustness_sensitivity.png')
fig.savefig(FIG_DIR / 'fig7_robustness_sensitivity.pdf')
plt.close()
print('Figure 7 saved')

print('\nAll figures generated successfully!')
for f in sorted(FIG_DIR.glob('*.pdf')):
    print(f'  {f.name}')

for pattern in ('fig*.png', 'fig*.pdf'):
    for src in FIG_DIR.glob(pattern):
        shutil.copy2(src, PAPER_FIG_DIR / src.name)
print(f'\nFigures synchronized to {PAPER_FIG_DIR}')
