"""Vector schematic of the primary LPU-Stream implementation and CQR split.

The full-basin MC and dense-head alternatives are intentionally outside this
diagram. All arrows indicate data flow, not gradient flow.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_architecture(pdf_path, png_path):
    ink, muted = '#183746', '#526874'
    blue, teal, gold = '#176A96', '#087D75', '#A25B22'
    fig = plt.figure(figsize=(9.4, 5.85), facecolor='white')
    ax = fig.add_axes([.015, .015, .97, .97])
    ax.set(xlim=(0, 12), ylim=(0, 7.5)); ax.axis('off')

    def text(x, y, label, size=10.5, color=ink, weight='normal', ha='left'):
        return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                       ha=ha, va='center', linespacing=1.35)

    def panel(x, y, w, h, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.13',
                                   fc=face, ec='#D5E1E7', lw=.8, zorder=0))

    def box(x, y, w, h, title, detail, color=blue):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.08',
                                   fc='white', ec=color, lw=1.1, zorder=2))
        text(x+w/2, y+h*.68, title, 11, color, 'bold', 'center')
        text(x+w/2, y+h*.27, detail, 10, ink, ha='center')

    def arrow(points, color=muted, dashed=False):
        if len(points)>2:
            ax.plot(*zip(*points[:-1]), color=color, lw=1.25,
                    linestyle='--' if dashed else '-', zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                    mutation_scale=11, lw=1.25, color=color,
                                    linestyle='--' if dashed else '-', shrinkA=0,
                                    shrinkB=1, zorder=1))

    text(.12, 7.24, 'LPU-Stream', 17, weight='bold')
    text(2.36, 7.24, 'Primary regional quantile model', 12, muted)
    panel(.1, 3.55, 11.8, 3.25, '#F3F7FA')
    text(.32, 6.48, 'a   Shared predictor', 12, blue, 'bold')
    text(11.65, 6.48, '104,099 trainable parameters', 10, muted, ha='right')
    box(.35, 5.05, 1.85, .93, 'Weather inputs', r'$180\times15$ channels')
    box(2.65, 5.05, 1.85, .93, 'Input fusion', r'$15+32=47$')
    box(4.95, 5.05, 1.75, .93, 'LSTM', '128 hidden units')
    box(7.15, 5.05, 1.9, .93, 'Prediction head', r'$160\to64\to3$')
    box(9.5, 5.05, 2.15, .93, 'Ordered quantiles', r'$\hat q^{(z)}_{.05},\ \hat q^{(z)}_{.50},\ \hat q^{(z)}_{.95}$')
    for a,b in [(2.2,2.65),(4.5,4.95),(6.7,7.15),(9.05,9.5)]:
        arrow([(a,5.52),(b,5.52)])
    box(.35, 3.95, 1.85, .78, 'Basin attributes', '13 descriptors', teal)
    box(2.65, 3.95, 1.85, .78, 'Static MLP', r'$13\to64\to32$', teal)
    arrow([(2.2,4.34),(2.65,4.34)],teal)
    arrow([(3.57,4.73),(3.57,5.05)],teal)
    arrow([(4.5,4.34),(8.1,4.34),(8.1,5.05)],teal)
    text(5.73,4.59,'32-d embedding to head',9.3,teal,ha='center')
    text(5.73,4.04,'Masked pinball training loss',9.3,teal,ha='center')
    text(10.57,4.55,'Softplus offsets',10,blue,ha='center')
    text(10.57,4.04,r'$\hat q_\tau=\mu_{\rm tr}+\sigma_{\rm tr}\hat q^{(z)}_\tau$',11,ha='center')
    text(.3,3.16,r'Predict day $t$ from $x_{t-180},\ldots,x_{t-1}$; quantile fitting uses training labels only.',10.5)

    panel(.1,.32,5.65,2.46,'#EFF7F4')
    panel(6.1,.32,5.8,2.46,'#FCF5EF')
    text(.32,2.47,'b   Fit calibration quantities',12,teal,'bold')
    text(6.32,2.47,'c   Apply to test predictions',12,gold,'bold')
    text(2.93,2.02,'Calibration quantiles + observed log-flow',10.3,ha='center')
    text(9.0,2.02,'Test quantiles on the same log-flow scale',10.3,ha='center')
    box(.42,.84,4.97,.79,'Global or predicted-regime CQR',
        'One radius, or 3 radii + predicted thresholds',teal)
    box(6.63,.84,4.9,.79,'Nominal 90% interval',
        r'$[\hat q_{.05}-r,\ \hat q_{.95}+r]$',gold)
    arrow([(2.93,1.85),(2.93,1.63)],teal)
    arrow([(9.0,1.85),(9.0,1.63)],gold)
    arrow([(5.39,1.22),(6.63,1.22)],teal)
    text(6.0,1.76,'Frozen',9.2,teal,ha='center')
    text(6.0,.71,'Apply',9.2,teal,ha='center')
    text(2.93,.53,'Calibration period: Oct 1995 - Sep 2000',9.5,muted,ha='center')
    text(9.0,.53,'Test observations are used only for evaluation',9.5,muted,ha='center')
    pdf_path, png_path = Path(pdf_path), Path(png_path)
    pdf_path.parent.mkdir(parents=True,exist_ok=True);png_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=.06, metadata={'Title':'LPU-Stream: primary predictor and held-out CQR'})
    fig.savefig(pdf_path.with_suffix('.svg'), bbox_inches='tight', pad_inches=.06)
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
