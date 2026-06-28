"""
Generate all publication-ready figures and update LaTeX tables.
Uses the corrected experimental data (fair Deep Ensembles, cross-region validation).
"""

import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality settings
plt.rcParams.update({
    'font.size': 10, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.linewidth': 1.0,
    'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ─── Colors ───
BLUE = '#2c7bb6'
RED = '#d7191c'
GREEN = '#1a9641'
ORANGE = '#fdae61'
DARK = '#333333'

# ─── Data ───
years = [1, 3, 5, 15]
scarce_data = {
    'nse': [0.665, 0.788, 0.818, 0.844],
    'picp_raw': [0.720, 0.809, 0.786, 0.889],
    'picp_cal': [0.890, 0.894, 0.880, 0.889],
    'q_cal': [0.313, 0.105, 0.093, 0.0006],
    'mpiw': [1.418, 1.059, 1.039, 0.857],
}

# ============================================================
# FIGURE 1: Data Scarcity Degradation
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) NSE
ax = axes[0]
ax.plot(years, scarce_data['nse'], 'o-', color=BLUE, linewidth=2.5, markersize=9, label='LPU-Stream', zorder=3)
ax.axhline(y=0.862, color=RED, linestyle='--', linewidth=1.5, label='XGBoost (full data)')
ax.fill_between(years, 0.5, scarce_data['nse'], alpha=0.08, color=BLUE)
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('Test NSE')
ax.set_title('(a) Point Prediction Accuracy', fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.95)
ax.grid(True, alpha=0.25)
ax.annotate("$-21\\%$ (NSE)", xy=(1, 0.665), xytext=(2.5, 0.58),
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
ax.annotate("$-19$ pp", xy=(1, 0.720), xytext=(2.5, 0.68),
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
picp_m = [0.596, 0.878, 0.926, 0.889]
mpiw_m = [0.391, 1.019, 1.150, 0.857]
winkler_m = [2.487, 1.659, 1.665, 1.403]
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
        ax.text(3.5, 0.905, 'Target 90%', fontsize=8, color=GREEN, ha='center')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig2_method_comparison.png')
fig.savefig(FIG_DIR / 'fig2_method_comparison.pdf')
plt.close()
print('Figure 2 saved')

# ============================================================
# FIGURE 3: Calibration Analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Calibration curve
ax = axes[0]
width_bins = ['Narrow\n(Q1)', 'Mid\n(Q2)', 'Wide\n(Q3)', 'Widest\n(Q4)']
picp_bins = [0.870, 0.872, 0.876, 0.874]
bars = ax.bar(width_bins, picp_bins, color=BLUE, edgecolor='white', width=0.5, alpha=0.85, zorder=2)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.7, label='Target 90%', zorder=1)
for i, v in enumerate(picp_bins):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Predicted Interval Width Quartile')
ax.set_ylabel('Observed PICP')
ax.set_title('(a) Calibration Curve (1-yr model)', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0.82, 0.93)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# (b) Width Ratio by flow regime
ax = axes[1]
flow_labels = ['Low Flow', 'Normal Flow', 'High Flow']
width_ratios = [1.128, 1.064, 0.701]
wr_colors = [BLUE, '#abd9e9', RED]
bars = ax.bar(flow_labels, width_ratios, color=wr_colors, edgecolor='white', width=0.5, alpha=0.85, zorder=2)
ax.axhline(y=1.0, color=DARK, linestyle='--', linewidth=1.2, alpha=0.6, label='Ideal (= 1.0)', zorder=1)
for b, v in zip(bars, width_ratios):
    label_color = RED if v < 0.9 else BLUE
    ax.text(b.get_x() + b.get_width()/2, v + 0.03, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color=label_color)
ax.set_ylabel('Width Ratio (actual / theoretical)')
ax.set_title('(b) Overconfidence by Flow Regime', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.4)
ax.grid(True, alpha=0.2, axis='y', zorder=1)

# (c) Coverage by flow regime across years
ax = axes[2]
years_plot = [1, 3, 5, 15]
cov_low = [0.914, 0.892, 0.885, 0.914]
cov_normal = [0.880, 0.878, 0.873, 0.879]
cov_high = [0.845, 0.856, 0.852, 0.875]
ax.plot(years_plot, cov_low, 'o-', color=BLUE, linewidth=2, markersize=8, label='Low Flow', zorder=3)
ax.plot(years_plot, cov_normal, 's-', color=ORANGE, linewidth=2, markersize=8, label='Normal Flow', zorder=3)
ax.plot(years_plot, cov_high, '^-', color=RED, linewidth=2, markersize=8, label='High Flow', zorder=3)
ax.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
ax.set_xlabel('Training Data (years)')
ax.set_ylabel('PICP')
ax.set_title('(c) Coverage by Flow Regime', fontweight='bold')
ax.legend(fontsize=9)
ax.set_xticks(years_plot)
ax.set_ylim(0.82, 0.94)
ax.grid(True, alpha=0.25, zorder=1)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig3_calibration.png')
fig.savefig(FIG_DIR / 'fig3_calibration.pdf')
plt.close()
print('Figure 3 saved')

# ============================================================
# FIGURE 4: Model Architecture (schematic)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis('off')

def draw_box(ax, cx, cy, w, h, text, color='#e0e0e0', text_color='black', fontsize=10):
    box = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='#444', linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='#555', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, shrinkA=5, shrinkB=5), zorder=1)

# Inputs
draw_box(ax, 1.5, 5.2, 2.8, 0.7, 'Dynamic Inputs\n(P, T, R, VP, DOY)', '#cce5ff')
draw_box(ax, 1.5, 3.5, 2.8, 0.7, 'Static Attributes\n(13 catchment properties)', '#ffe6cc')

# Embedding
draw_box(ax, 5.0, 3.5, 2.0, 0.6, 'MLP Embed\n(32-dim)', '#fff2cc')

# LSTM
draw_box(ax, 7.5, 5.2, 2.5, 0.7, 'LSTM\n(128 hidden)', '#d5f5e3', fontsize=11)

# Concat
draw_box(ax, 7.5, 3.5, 2.2, 0.6, 'Concat\n(h + embed)', '#e8daef')

# Output
draw_box(ax, 7.5, 1.8, 2.5, 0.7, 'Prediction Head\n(Q0.05, Q0.5, Q0.95)', '#fadbd8', fontsize=10)

# Post-hoc
draw_box(ax, 7.5, 0.6, 2.5, 0.5, 'CQR Calibration\n(Post-hoc)', '#d5f5e3', fontsize=9)

# Arrows
draw_arrow(ax, 2.9, 5.2, 6.2, 5.2)
draw_arrow(ax, 2.9, 3.5, 4.0, 3.5)
draw_arrow(ax, 6.0, 3.5, 6.4, 3.5)
ax.annotate('', xy=(7.5, 4.1), xytext=(7.5, 4.8),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5), zorder=1)
draw_arrow(ax, 7.5, 3.2, 7.5, 2.5)
draw_arrow(ax, 7.5, 1.4, 7.5, 1.1)
ax.annotate('', xy=(5.0, 5.2), xytext=(5.0, 4.2),
            arrowprops=dict(arrowstyle='->', color='#999', lw=1.0, linestyle='dashed'), zorder=1)

ax.text(1.5, 5.8, 'Time Series Data', ha='center', fontsize=9, fontstyle='italic', color='#555')
ax.text(1.5, 4.5, 'Basin Properties', ha='center', fontsize=9, fontstyle='italic', color='#555')
ax.text(9.0, 5.2, 'Hidden State', fontsize=8, fontstyle='italic', color='#555')
ax.text(9.0, 3.5, 'Conditioned\nRepresentation', fontsize=8, fontstyle='italic', color='#555')
ax.text(9.0, 0.6, 'Coverage-\nAdjusted', fontsize=8, fontstyle='italic', color='#555')
ax.text(5, 5.9, 'LPU-Stream Architecture (99K parameters)', ha='center', fontsize=13, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig4_architecture.png')
fig.savefig(FIG_DIR / 'fig4_architecture.pdf')
plt.close()
print('Figure 4 saved')

# ============================================================
# FIGURE 5: Cross-Region Validation
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8.5, 5))

regions = ['Very Humid\n(Q1)', 'Humid', 'Transitional\n(Q2-Q3)', 'Dry/\nSemi-arid', 'Very Dry\n(Q4)', 'All\nCAMELS']
nse_r = [0.843, 0.822, 0.726, 0.786, 0.837, 0.844]
picp_r = [0.890, 0.880, 0.867, 0.897, 0.930, 0.889]
colors_r = ['#0571b0', '#92c5de', '#f7f7f7', '#f4a582', '#ca0020', '#555555']

x = np.arange(len(regions))
w = 0.35

bars1 = ax.bar(x - w/2, nse_r, w, color=colors_r, edgecolor='#333', linewidth=0.5, label='NSE', alpha=0.85, zorder=2)
ax_twin = ax.twinx()
bars2 = ax_twin.bar(x + w/2, picp_r, w, color=BLUE, edgecolor='#333', linewidth=0.5, label='PICP', alpha=0.7, zorder=2)
ax_twin.axhline(y=0.90, color=GREEN, linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=9)
ax.set_ylabel('NSE', color=DARK)
ax_twin.set_ylabel('PICP', color=BLUE)
ax.set_title('Cross-Region Validation', fontweight='bold')
ax.grid(True, alpha=0.2, axis='y', zorder=1)
ax.set_ylim(0.6, 1.0)
ax_twin.set_ylim(0.6, 1.0)

for i in range(len(regions)):
    ax.text(i - w/2, nse_r[i] + 0.012, f'{nse_r[i]:.3f}', ha='center', fontsize=8.5, fontweight='bold')
    ax_twin.text(i + w/2, picp_r[i] + 0.012, f'{picp_r[i]:.3f}', ha='center', fontsize=8.5, color=BLUE, fontweight='bold')

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

# (a) Multi-seed stability chart — simplified
ax = axes[0]
seeds = ['Seed 42', 'Seed 123', 'Seed 456', 'Mean']
nse_s = [0.665, 0.729, 0.684, 0.693]
picp_s = [0.890, 0.901, 0.896, 0.896]

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

# (b) Static embedding ablation — with bracket-style delta annotations
ax = axes[1]
methods = ['1-yr\nw/ static', '1-yr\nw/o static', '5-yr\nw/ static', '5-yr\nw/o static']
nse_ab = [0.665, 0.581, 0.818, 0.740]
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

print('\nAll figures generated successfully!')
for f in sorted(FIG_DIR.glob('*.pdf')):
    print(f'  {f.name}')
