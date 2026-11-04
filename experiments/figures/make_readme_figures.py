"""Build README figures and the model schematic from the current protocol.

Default outputs stay under results/. Use --copy-to-docs when updating the
committed README images. Numeric panels use the same tables as the manuscript.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import dump
OUT=ROOT/'results/figures/readme'
INK='#183746';BLUE='#176A96';TEAL='#138875';ORANGE='#C0672D';GREY='#687883';GRID='#DEE6EB'

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,
        'text.color':INK,'xtick.color':GREY,'ytick.color':GREY,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':GRID,'grid.color':GRID,'grid.linewidth':.7,
        'axes.titleweight':'bold','pdf.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=190,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)

def design():
    fig,ax=plt.subplots(figsize=(11.4,4.8));fig.subplots_adjust(left=.17,right=.97,top=.76,bottom=.19)
    fig.text(.04,.95,'Separate budgets for training and calibration',fontsize=19,weight='bold')
    fig.text(.04,.88,'Primary experiment  |  50 basins jointly trained  |  4 windows × 3 seeds = 12 regional fits',fontsize=11,color=GREY)
    date=lambda s:mdates.date2num(datetime.fromisoformat(s))
    cutoff=date('1995-10-01')
    for y,year,start in [(5,15,'1980-10-01'),(4,5,'1990-10-01'),(3,3,'1992-10-01'),(2,1,'1994-10-01')]:
        ax.barh(y,cutoff-date(start),left=date(start),height=.56,color=BLUE,zorder=3)
    for y,start,end,color in [(1,'1995-10-01','2000-10-01',TEAL),(0,'2000-10-01','2009-01-01',ORANGE)]:
        ax.barh(y,date(end)-date(start),left=date(start),height=.56,color=color,zorder=3)
    ax.set_yticks(range(6),['Test targets','Calibration targets','1-year labels','3-year labels','5-year labels','15-year labels'])
    ax.tick_params(axis='y',length=0,pad=12);ax.xaxis.set_major_locator(mdates.YearLocator(5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(date('1979-12-01'),date('2009-04-01'));ax.set_ylim(-.65,5.65);ax.grid(axis='x',zorder=0)
    ax.spines[['left','bottom']].set_visible(False);ax.axvline(cutoff,color=GREY,ls='--',lw=1)
    ax.text(cutoff-90,5.63,'Training ends 30 Sep 1995',ha='right',va='bottom',fontsize=10,color=GREY)
    fig.text(.04,.07,'180-day antecedent weather + 13 static attributes → ordered flow quantiles → held-out CQR',fontsize=11)
    save(fig,'design_timeline')

def results_panels(cells,basins):
    years=[1,3,5,15];x=np.arange(4);fig,axes=plt.subplots(1,2,figsize=(11.4,4.5),layout='constrained')
    def metric(method,field):
        return [cells[(cells.year==yr)&(cells.method==method)&(cells.dimension=='pooled')][field].to_numpy() for yr in years]
    values=[basins[(basins.year==yr)&(basins.method=='raw')].groupby('seed').nse.median().to_numpy() for yr in years]
    axes[0].errorbar(x,[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],color=BLUE,marker='o',lw=2.2,capsize=4)
    axes[0].set_title('a  Point skill improves',loc='left',pad=16);axes[0].set_ylabel('Median basin NSE');axes[0].set_ylim(.47,.70)
    for method,label,color,marker in [('raw','Raw QR',ORANGE,'o'),('global','Global CQR',TEAL,'s')]:
        values=metric(method,'picp');axes[1].errorbar(x,[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],label=label,color=color,marker=marker,lw=2.2,capsize=4)
    axes[1].axhline(.9,color=GREY,ls='--',lw=1,label='Nominal 90%');axes[1].set_ylim(.74,.94);axes[1].set_ylabel('Pooled test coverage')
    axes[1].set_title('b  Calibration improves pooled coverage',loc='left',pad=16);axes[1].legend(frameon=False,fontsize=10,loc='lower right')
    for ax in axes:ax.set_xticks(x,years);ax.set_xlabel('Supervised training years');ax.grid(axis='y')
    save(fig,'fig1_scarcity_effects')

def groups(cells):
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.6),layout='constrained');records=[]
    for ax,dimension,title in zip(axes,['predicted_flow','observed_flow'],['a  Groups assigned before the outcome','b  Groups selected by observed outcome']):
        for offset,(method,label,color) in enumerate([('raw','Raw QR',GREY),('global','Global CQR',BLUE),('predicted','Predicted-regime CQR',TEAL)]):
            values=[cells[(cells.year==1)&(cells.method==method)&(cells.dimension==dimension)&(cells.group==g)].picp.to_numpy() for g in ['low','middle','high']]
            ax.errorbar(np.arange(3)+(offset-1)*.13,[v.mean() for v in values],yerr=[v.std(ddof=1) for v in values],label=label,color=color,marker=['x','o','s'][offset],lw=1.9,capsize=3)
            records.extend({'dimension':dimension,'method':method,'group':g,'mean_picp':float(v.mean())} for g,v in zip(['low','middle','high'],values))
        ax.axhline(.9,color=GREY,ls='--',lw=1);ax.set_xticks(range(3),['Low','Middle','High']);ax.set_ylim(.65,1.01)
        ax.set_ylabel('One-year model: test coverage');ax.set_title(title,loc='left',pad=16,fontsize=11);ax.grid(axis='y')
        ax.set_xlabel('Calibration-predicted flow group' if dimension=='predicted_flow' else 'Realized test-flow group')
    axes[0].legend(frameon=False,fontsize=9,loc='lower left')
    save(fig,'fig2_conditional_coverage');return records

def architecture():
    fig,ax=plt.subplots(figsize=(8.8,4.3));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    def box(x,y,w,h,label,color='#EAF2F6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',facecolor=color,edgecolor='#B5C8D2',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11.5,color=INK)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GREY,'lw':1.3,'shrinkA':3,'shrinkB':4})
    box(.1,3,2,.9,'Weather history\nL × 15 channels')
    box(.1,.5,2,.9,'Static attributes\n13 descriptors')
    box(2.6,.5,2,.9,'Static MLP\n13 → 64 → 32')
    box(2.6,3,2,.9,'Concatenate\nat each step')
    box(5.1,3,1.8,.9,'LSTM\n128 units')
    box(7.5,3,2,.9,'Head MLP\n160 → 64 → 3')
    box(10.1,3,1.8,.9,'Quantiles\n'+r'$q_{.05}\leq q_{.50}\leq q_{.95}$')
    box(7.5,.5,2,.9,'Calibration\nflow + quantiles','#E7F2ED')
    box(10.1,.5,1.8,.9,'CQR\n90% interval','#E7F2ED')
    for a,b in [((2.1,3.45),(2.6,3.45)),((2.1,.95),(2.6,.95)),((3.6,1.4),(3.6,3)),((4.6,3.45),(5.1,3.45)),((6.9,3.45),(7.5,3.45)),((9.5,3.45),(10.1,3.45)),((11,3),(11,1.4)),((9.5,.95),(10.1,.95))]:arrow(a,b)
    ax.plot([4.6,8.5,8.5],[.95,1.95,3],color=GREY,lw=1.2)
    arrow((8.5,2.6),(8.5,3))
    ax.text(6.2,2.15,'Static embedding also enters the head',ha='center',fontsize=9,color=GREY)
    ax.text(.1,4.6,'Regional model and held-out calibration',fontsize=14,weight='bold')
    ax.text(.1,-.05,'L = 180; softplus offsets enforce quantile ordering.',fontsize=10,color=GREY)
    fig.savefig(ROOT/'results/figures/architecture_overview.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/'architecture_overview.png',dpi=190,bbox_inches='tight',pad_inches=.12);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--copy-to-docs',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);style()
    cells=pd.read_csv(ROOT/'results/tables/coverage_cells.csv');basins=pd.read_csv(ROOT/'results/tables/basin_metrics.csv')
    design();results_panels(cells,basins);group_values=groups(cells);architecture()
    paths=[ROOT/'results/tables'/name for name in ['coverage_cells.csv','basin_metrics.csv','coverage_summary.json']]
    manifest={'generator':'experiments/figures/make_readme_figures.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inputs_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'uncertainty':'Mean and sample SD over three model seeds; plotted SD bars are not confidence intervals.',
        'groups':group_values,'observed_group_rule':'Within-basin test Q0.33/Q0.67, ties in the lower group; retrospective.',
        'predicted_group_rule':'Within-basin calibration-predicted Q0.33/Q0.67; radii pooled across relative calibration strata.'}
    dump(OUT/'figure_provenance.json',manifest)
    if args.copy_to_docs:
        destination=ROOT/'docs/figures';destination.mkdir(parents=True,exist_ok=True)
        for name in ['design_timeline.png','fig1_scarcity_effects.png','fig2_conditional_coverage.png','figure_provenance.json']:
            shutil.copy2(OUT/name,destination/name)
    print('README figures and model schematic generated; documentation copied:',args.copy_to_docs,flush=True)

if __name__=='__main__':main()
