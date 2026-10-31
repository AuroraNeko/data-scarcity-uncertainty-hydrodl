"""Audit common-scale distributional scores without re-fitting any model."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtri

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import integrated_quantile_score,gaussian_crps,conformal_radius,scores,interval_metrics,project_nonempty
from src.evaluation.protocol import dump
BASE=ROOT/'results';ARCH=ROOT/'results/predictions/corrected_full_671'

def main():
    arch=np.load(ARCH/'test_predictions.npz');index=np.load(ARCH/'evaluation_index.npz')
    stats=json.loads((ROOT/'data/metadata/normalization_stats.json').read_text())
    mu,sd=stats['target_mean'],stats['target_std'];mask=arch['mask']>0
    y=arch['target'][mask].astype(float)*sd+mu;b=arch['basin_index'][mask];cuts=np.quantile(y,[.33,.67])
    obs=np.searchsorted(cuts,y,side='left');taus3=np.array([.05,.5,.95]);rows=[];by_basin={};audits=[];levels=[];reliability={}
    archive_cal=np.load(ARCH/'calibration_predictions.npz');calmask=archive_cal['mask']>0
    cy=archive_cal['target'][calmask].astype(float)*sd+mu
    def score(name,kind,values,grid):
        key=f'{name}_{kind}';row={'method':name,'score':kind,'grid':grid,'scale':'log1p(mm/day)',
            'pooled':float(values.mean()),'n':len(y),'low':float(values[obs==0].mean()),'high':float(values[obs==2].mean())}
        sums=np.bincount(b,weights=values,minlength=671);n=np.bincount(b,minlength=671)
        by_basin[key]={'sum':sums,'n':n}
        rng=np.random.default_rng(20261030);draws=rng.integers(0,671,(2000,671))
        boot=sums[draws].sum(1)/n[draws].sum(1);row['basin_ci_low'],row['basin_ci_high']=np.quantile(boot,[.025,.975])
        rows.append(row)
    for name in ['single_qr','ensemble','mc_dropout']:
        q=arch[name][mask].astype(float)*sd+mu
        score(name,'common_3tau_iqs',integrated_quantile_score(y,q,taus3),'0.05,0.50,0.95; trapezoid, truncated')
    for name in ['mdn','umal','mc_dropout_noise']:
        dense=np.load(ROOT/f'results/predictions/benchmark_671/dense_{name}_taus.npz')
        own=np.load(ROOT/f'results/predictions/benchmark_671/{name}_predictions.npz')
        assert np.array_equal(mask,dense['test_mask']>0) and np.array_equal(mask,own['test_mask']>0)
        assert np.array_equal(index['test_dates'],own['test_dates']) and np.array_equal(index['basin_ids'],own['basin_ids'])
        delta=np.max(np.abs(y-own['target_test'][mask]))
        assert delta<1e-6
        q=dense['test'][mask].astype(float);taus=dense['taus'];cols=[int(np.flatnonzero(np.isclose(taus,t))[0]) for t in taus3]
        cq=dense['calibration'][calmask].astype(float)
        assert np.array_equal(dense['calibration_mask']>0,calmask)
        assert np.array_equal(own['calibration_dates'],index['calibration_dates'])
        assert np.max(np.abs(own['target_calibration'][calmask]-cy))<1e-6
        reliability[name]={'quantile_coverage':{str(t):float((y<=q[:,j]).mean()) for j,t in enumerate(taus)}}
        for level in [.5,.8,.9,.95,.99]:
            alpha=1-level;lower=int(np.argmin(abs(taus-alpha/2)));upper=int(np.argmin(abs(taus-(1-alpha/2))))
            assert abs(taus[lower]-alpha/2)<1e-8 and abs(taus[upper]-(1-alpha/2))<1e-8
            radius=conformal_radius(scores(cq[:,lower],cq[:,upper],cy),alpha)
            lo,hi,nempty=project_nonempty(q[:,lower]-radius,q[:,upper]+radius)
            for mode,l,u in [('raw',q[:,lower],q[:,upper]),('cqr',lo,hi)]:
                levels.append({'method':name,'nominal':level,'mode':mode,'radius':radius,'empty_cqr_projected':nempty if mode=='cqr' else 0,
                               **interval_metrics(l,u,y,alpha)})
        score(name,'common_3tau_iqs',integrated_quantile_score(y,q[:,cols],taus3),'0.05,0.50,0.95; trapezoid, truncated')
        score(name,'dense_13tau_iqs',integrated_quantile_score(y,q,taus),','.join(str(t) for t in taus))
        audit={'method':name,'mask_and_dates_equal':True,'target_max_abs_diff':delta}
        if name=='umal':
            loc=q[:,cols[1]];scale=(q[:,cols[2]]-q[:,cols[0]])/(2*ndtri(.95))
            analytic=gaussian_crps(y,loc,scale);score(name,'analytic_gaussian_crps',analytic,'complete distribution, analytic')
            # Validate convergence using the same fixed, uniformly spaced sample subset.
            idx=np.linspace(0,len(y)-1,20000,dtype=int);validation={}
            for n_tau in [101,1001,5001]:
                grid=np.linspace(.00001,.99999,n_tau);numeric=np.zeros(len(idx))
                # Accumulate without allocating a 20k x 5001 quantile matrix.
                prev=None
                for t in grid:
                    e=y[idx]-(loc[idx]+scale[idx]*ndtri(t));pb=np.maximum(t*e,(t-1)*e)
                    if prev is not None:numeric+=(grid[1]-grid[0])*(prev+pb)
                    prev=pb
                validation[str(n_tau)]={'mean_numeric':float(numeric.mean()),'mean_analytic':float(analytic[idx].mean()),
                    'mean_absolute_error':float(np.abs(numeric-analytic[idx]).mean())}
            audit['integration_convergence']=validation
        audits.append(audit)
        dense.close();own.close();del q
        print('Scoring checked',name,flush=True)
    paired={};ref=by_basin['single_qr_common_3tau_iqs'];rng=np.random.default_rng(20261031)
    draw=rng.integers(0,671,(2000,671))
    for name in ['ensemble','mc_dropout','mdn','umal','mc_dropout_noise']:
        v=by_basin[name+'_common_3tau_iqs']
        dif=v['sum'][draw].sum(1)/v['n'][draw].sum(1)-ref['sum'][draw].sum(1)/ref['n'][draw].sum(1)
        paired[name+'_minus_single_qr']={'estimate':v['sum'].sum()/v['n'].sum()-ref['sum'].sum()/ref['n'].sum(),
                                         'basin_ci95':np.quantile(dif,[.025,.975])}
    pd.DataFrame(rows).to_csv(BASE/'tables/distribution_scores.csv',index=False)
    pd.DataFrame(levels).to_csv(BASE/'tables/dense_level_metrics.csv',index=False)
    dump(BASE/'tables/scoring_summary.json',{'rows':rows,'alignment_audit':audits,'common_grid_paired_differences':paired,
        'level_reliability':reliability,'central_interval_levels':levels,
        'correction':'Old dense scoring mixed standardized targets with log-scale quantiles. Old reference scores were standardized, not log scale.',
        'common_grid_interpretation':'Finite-grid IQS; not a complete-distribution CRPS.',
        'model_display_names':{'umal':'Gaussian location-scale quantile head','mdn':'Five-component Gaussian mixture density network',
                              'mc_dropout_noise':'MC Dropout with Gaussian observation-noise head'}})
    print('Distribution scoring complete',flush=True)

if __name__=='__main__':main()
