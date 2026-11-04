"""Simulation study: grouping by predicted value versus grouping by outcome.

Generates data with known conditional quantiles, then compares two interval
constructions (correct oracle quantiles and a biased, narrower model) when
coverage is measured within groups defined before the outcome is known
(predicted mean) versus groups selected after observing it (realized
outcome). This isolates how much conditional undercoverage is created by
the grouping choice itself.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import interval_metrics
from src.evaluation.protocol import dump

def main():
    (ROOT/'results/figures').mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(20261030);n=1000000;x_cal=rng.uniform(0,1,100000);x=rng.uniform(0,1,n)
    mu=.8*x;sigma=.2+.5*x;y=mu+sigma*rng.standard_normal(n)
    pred_groups=np.searchsorted(np.quantile(.8*x_cal,[1/3,2/3]),mu,side='left')
    outcome_groups=np.searchsorted(np.quantile(y,[1/3,2/3]),y,side='left')
    schemes={'Oracle conditional quantiles':(mu+sigma*ndtri(.05),mu+sigma*ndtri(.95)),
             'Biased, narrower model':(.7*mu-.1+.7*sigma*ndtri(.05),.7*mu-.1+.7*sigma*ndtri(.95))}
    rows=[]
    for name,(lo,hi) in schemes.items():
        for dimension,g in [('predicted_mean',pred_groups),('realized_outcome',outcome_groups)]:
            for group in range(3):
                sel=g==group
                rows.append({'model':name,'grouping':dimension,'group':group,
                             **interval_metrics(lo[sel],hi[sel],y[sel])})
    dump(ROOT/'results/tables/selection_simulation.json',{'seed':20261030,'n_test':n,'data_generating_process':
        'X~Uniform(0,1), Y=0.8X+(0.2+0.5X)Z, Z~N(0,1), independent calibration X for prospective thresholds',
        'constant_gaussian_analytic':{'overall_coverage':.9,'observed_upper_third_coverage':.85,
                                    'observed_top_two_percent_coverage':0.,'predictor_conditional_coverage':.9},'rows':rows})
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained');colors=['#225e78','#b16440']
    for ax,dim,title in zip(axes,['predicted_mean','realized_outcome'],['Groups available before Y','Groups selected after observing Y']):
        for j,name in enumerate(schemes):
            values=[r['picp'] for r in rows if r['model']==name and r['grouping']==dim]
            ax.plot(range(3),values,marker='o',label=name,color=colors[j])
        ax.axhline(.9,color='#777777',linestyle='--');ax.set_xticks(range(3),['Low','Middle','High']);ax.set_ylim(.35,1.)
        ax.set_title(title,fontsize=10);ax.set_ylabel('90% interval coverage')
    axes[0].legend(frameon=False,fontsize=8);fig.savefig(ROOT/'results/figures/selection_simulation.png',dpi=220);plt.close(fig)
    print('Known-quantile selection simulation complete',flush=True)

if __name__=='__main__':main()
