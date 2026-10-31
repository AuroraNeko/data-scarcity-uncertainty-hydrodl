"""Recompute all scarcity diagnostics under one explicit grouping protocol."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import (PRIMARY_CUTS,conformal_radius,scores,interval_arrays,
    interval_metrics,fit_basin_thresholds,assign_basin_groups,observed_groups,predicted_group_intervals,
    project_nonempty,nse,bootstrap_basin,clean,dump)
from src.corrections.common import load
BASE=ROOT/'results';SEEDS=[42,123,456];YEARS=[1,3,5,15];NBOOT=5000



def summarize(lo,hi,y,selected):
    if not selected.any():return {'n':0,'picp':None,'mpiw':None,'winkler':None,'p_below':None,'p_above':None}
    return interval_metrics(lo[selected],hi[selected],y[selected])



def main():
    names=pd.read_csv(ROOT/'data/raw/camels_us/camels_name.txt',sep=';',dtype={'gauge_id':str,'huc_02':str}).set_index('gauge_id')
    raw=np.load(BASE/'cache/primary_raw.npz');attr=raw['static'];rawbasins=raw['basins']
    static_cols=json.loads((BASE/'cache/primary_raw.json').read_text())['static_columns']
    aridity={b:attr[i,static_cols.index('aridity')] for i,b in enumerate(rawbasins)}
    all_rows=[];basin_rows=[];window_rows=[];tail_rows=[];cells=[];rawyear=[];physical=[];bias=[];quantile_rows=[];shifted_bias=[]
    for year in YEARS:
        for seed in SEEDS:
            cal,test=load(year,seed);q=cal['q'];tq=test['q'];y=test['y'];b=test['b'];date=test['date']
            sc=scores(q[:,0],q[:,2],cal['y']);r=conformal_radius(sc)
            pl,pu,info=predicted_group_intervals(q,cal['y'],cal['b'],tq,b)
            dl,du,_=predicted_group_intervals(q,cal['y'],cal['b'],tq,b,cuts=np.arange(.1,1.,.1))
            br={bb:conformal_radius(sc[cal['b']==bb]) for bb in np.unique(cal['b'])}
            rb=np.asarray([br[bb] for bb in b])
            aq=np.quantile(list(aridity.values()),[.25,.5,.75])
            cg=np.searchsorted(aq,[aridity[bb] for bb in cal['b']],side='left')
            tg=np.searchsorted(aq,[aridity[bb] for bb in b],side='left')
            ar=np.asarray([conformal_radius(sc[cg==g]) for g in range(4)])[tg]
            schemes={'raw':(tq[:,0],tq[:,2]),'global':(tq[:,0]-r,tq[:,2]+r),
                     'predicted':(pl,pu),'decile':(dl,du),'per_basin':(tq[:,0]-rb,tq[:,2]+rb),
                     'aridity':(tq[:,0]-ar,tq[:,2]+ar)}
            median_predictions={}
            observed_thresholds=fit_basin_thresholds(cal['y'],cal['b'])
            oc=assign_basin_groups(cal['y'],cal['b'],observed_thresholds)
            ot=assign_basin_groups(y,b,observed_thresholds)
            oradii=np.array([conformal_radius(sc[oc==g]) for g in range(3)])
            schemes['observed_reference']=(tq[:,0]-oradii[ot],tq[:,2]+oradii[ot])
            # Calibration-only median shifts, with CQR re-fitted after shifting.
            for mode in ['basin','basin_predicted']:
                cal_shift=np.zeros(len(q));test_shift=np.zeros(len(tq))
                for bb in np.unique(b):
                    for g in (range(3) if mode=='basin_predicted' else [-1]):
                        cs=(cal['b']==bb)&((info['cal_groups']==g) if g>=0 else True)
                        ts=(b==bb)&((info['test_groups']==g) if g>=0 else True)
                        fallback=cal['b']==bb
                        shift=float((cal['y'][cs]-q[cs,1]).mean()) if cs.any() else float((cal['y'][fallback]-q[fallback,1]).mean())
                        cal_shift[cs]=shift;test_shift[ts]=shift
                rr=conformal_radius(scores(q[:,0]+cal_shift,q[:,2]+cal_shift,cal['y']))
                schemes['bias_'+mode]=(tq[:,0]+test_shift-rr,tq[:,2]+test_shift+rr)
                median_predictions['bias_'+mode]=tq[:,1]+test_shift
            obs=observed_groups(y,b);pred=info['test_groups']
            for dimension,groups in [('observed_flow',obs),('predicted_flow',pred)]:
                for group in range(3):
                    sel=groups==group
                    for col,tau in enumerate([.05,.5,.95]):
                        quantile_rows.append({'year':year,'seed':seed,'dimension':dimension,'group':group,'tau':tau,
                            'n':int(sel.sum()),'fraction_below_quantile':float((y[sel]<=tq[sel,col]).mean())})
                    for bb in np.unique(b):
                        local=sel&(b==bb)
                        if not local.any():continue
                        qphys=np.maximum(np.expm1(tq[local,1]),0);yphys=np.maximum(np.expm1(y[local]),0)
                        bias.append({'year':year,'seed':seed,'basin':bb,'dimension':dimension,'group':group,
                            'n':int(local.sum()),'mean_median_forecast_bias_log':float((tq[local,1]-y[local]).mean()),
                            'median_residual_log':float(np.median(y[local]-tq[local,1])),
                            'mean_bias_physical':float((qphys-yphys).mean()),
                            'relative_mean_bias_physical':float((qphys-yphys).mean()/yphys.mean()) if yphys.mean()>0 else None})
            month=date.astype('datetime64[M]').astype(int)%12+1
            season=(month%12)//3
            wy=date.astype('datetime64[Y]').astype(int)+1970+(month>=10)
            all_rows.append({'year':year,'seed':seed,'radius':r,'n_cal':len(sc),'n_test':len(y),
                             'quantile_coverage':[(y<=tq[:,j]).mean() for j in range(3)],
                             'predicted_radii':info['radii']})
            for method,(lo,hi) in schemes.items():
                lo,hi,_=project_nonempty(lo,hi)
                point_prediction=median_predictions.get(method,tq[:,1])
                if method in ['raw','global','predicted']:
                    qlo=np.maximum(np.expm1(lo),0);qhi=np.maximum(np.expm1(hi),0);yp=np.maximum(np.expm1(y),0)
                    for group,selection in [('all',np.ones(len(y),dtype=bool)),('observed_high',obs==2),('predicted_high',pred==2)]:
                        physical.append({'year':year,'seed':seed,'method':method,'group':group,
                                         **summarize(qlo,qhi,yp,selection)})
                for dimension,groups,labels in [('pooled',np.zeros(len(y)),['all']),
                    ('observed_flow',obs,['low','middle','high']),('predicted_flow',pred,['low','middle','high']),
                    ('season',season,['DJF','MAM','JJA','SON']),('water_year',wy,[str(v) for v in np.unique(wy)])]:
                    levels=np.unique(wy) if dimension=='water_year' else range(len(labels))
                    for g,label in zip(levels,labels):
                        cells.append({'year':year,'seed':seed,'method':method,'dimension':dimension,'group':label,
                                      **summarize(lo,hi,y,groups==g)})
                ia=interval_arrays(lo,hi,y)
                for bb in np.unique(b):
                    sel=b==bb
                    basin_rows.append({'year':year,'seed':seed,'basin':bb,'huc2':names.loc[bb,'huc_02'],'method':method,
                        'nse':nse(y[sel],point_prediction[sel]),**summarize(lo,hi,y,sel),
                        'high_picp':float(ia['covered'][sel&(obs==2)].mean()) if (sel&(obs==2)).any() else np.nan})
                    if method in ['global','bias_basin','bias_basin_predicted']:
                        high=sel&(obs==2)
                        shifted_bias.append({'year':year,'seed':seed,'basin':bb,'method':method,
                            'observed_high_mean_bias_log':float((point_prediction[high]-y[high]).mean()) if high.any() else np.nan})
                    if method in ['raw','global','predicted']:
                        for period in np.unique(wy):
                            for grouping,gg in [('all',np.ones(len(y),dtype=bool)),('observed_high',obs==2),('predicted_high',pred==2)]:
                                local=sel&(wy==period)&gg
                                rawyear.append({'year':year,'seed':seed,'basin':bb,'method':method,'water_year':int(period),
                                    'group':grouping,'n':int(local.sum()),'hits':float(ia['covered'][local].sum()),
                                    'width_sum':float(ia['width'][local].sum()),'winkler_sum':float(ia['winkler'][local].sum())})
                if method in ['raw','global','predicted','decile']:
                    for fraction in [1.,.33,.2,.1,.05,.02]:
                        selected=np.ones(len(y),dtype=bool) if fraction==1 else np.zeros(len(y),dtype=bool)
                        if fraction<1:
                            for bb in np.unique(b):
                                sel=b==bb;selected[sel]=y[sel]>np.quantile(y[sel],1-fraction)
                        tail_rows.append({'year':year,'seed':seed,'method':method,'requested_upper_fraction':fraction,
                                          'actual_fraction':float(selected.mean()),**summarize(lo,hi,y,selected)})
            for length in [1,2,3,4,5]:
                sel=cal['date']>=np.datetime64(f'{2000-length}-10-01')
                rr=conformal_radius(sc[sel]);lo=tq[:,0]-rr;hi=tq[:,2]+rr
                window_rows.append({'year':year,'seed':seed,'calibration_years':length,'n_cal':int(sel.sum()),'radius':rr,
                    **interval_metrics(lo,hi,y),'observed_high_picp':summarize(lo,hi,y,obs==2)['picp'],
                    'predicted_high_picp':summarize(lo,hi,y,pred==2)['picp']})
            print('Coverage complete',year,seed,flush=True)
    for name,rows in [('coverage_cells',cells),('basin_metrics',basin_rows),('calibration_windows',window_rows),
                      ('tail_thresholds',tail_rows),('basin_year_sufficient_statistics',rawyear),
                      ('physical_interval_metrics',physical),('bias_diagnostics',bias),('group_quantile_reliability',quantile_rows)]:
        pd.DataFrame(rows).to_csv(BASE/f'tables/{name}.csv',index=False)
    pd.DataFrame(shifted_bias).to_csv(BASE/'tables/shifted_bias_diagnostics.csv',index=False)
    bf=pd.DataFrame(basin_rows);cf=pd.DataFrame(cells);yf=pd.DataFrame(rawyear)
    basins=np.sort(bf['basin'].unique());regions=names.loc[basins,'huc_02'].to_numpy()
    endpoints={};deltas={}
    for method in ['raw','global','predicted']:
        for metric in ['nse','picp','mpiw','winkler','high_picp']:
            frame=bf[(bf.method==method)&bf.year.isin([1,15])].groupby(['year','basin'])[metric].mean().unstack('year').loc[basins]
            delta=frame[15]-frame[1]
            key=f'{method}_{metric}'
            endpoints[key]=bootstrap_basin(delta.to_numpy(),basins,regions,'median' if metric=='nse' else 'mean')
            deltas[key]=delta
    # Direct paired calibration gain: estimate each seed's pooled ratio, then average seeds.
    gains={};rng=np.random.default_rng(20261031)
    for grouping in ['observed_high','predicted_high','all']:
        f=yf[(yf.year==1)&(yf.group==grouping)]
        num={};den={}
        for method in ['global','predicted']:
            ff=f[f.method==method].groupby(['seed','basin'])[['hits','n']].sum()
            num[method]=np.stack([ff.loc[s].reindex(basins)['hits'].to_numpy() for s in SEEDS])
            den[method]=np.stack([ff.loc[s].reindex(basins)['n'].to_numpy() for s in SEEDS])
        def calculate(ids):
            return np.nanmean(num['predicted'][:,ids].sum(-1)/den['predicted'][:,ids].sum(-1)-
                              num['global'][:,ids].sum(-1)/den['global'][:,ids].sum(-1))
        samples=[calculate(rng.integers(0,len(basins),len(basins))) for _ in range(NBOOT)]
        spatial=[]
        for _ in range(NBOOT):
            picks=rng.choice(np.unique(regions),len(np.unique(regions)),replace=True)
            spatial.append(calculate(np.concatenate([np.flatnonzero(regions==r) for r in picks])))
        # Synchronous water-year blocks preserve cross-basin meteorological co-occurrence.
        periods=np.sort(f.water_year.unique());ynum={};yden={}
        for method in ['global','predicted']:
            ff=f[f.method==method].groupby(['seed','water_year'])[['hits','n']].sum()
            ynum[method]=np.stack([ff.loc[s].reindex(periods)['hits'].to_numpy() for s in SEEDS])
            yden[method]=np.stack([ff.loc[s].reindex(periods)['n'].to_numpy() for s in SEEDS])
        td=[]
        for _ in range(NBOOT):
            picks=rng.integers(0,len(periods),len(periods))
            td.append(np.mean(ynum['predicted'][:,picks].sum(-1)/yden['predicted'][:,picks].sum(-1)-
                              ynum['global'][:,picks].sum(-1)/yden['global'][:,picks].sum(-1)))
        gains[grouping]={'predicted_minus_global':calculate(np.arange(len(basins))),
            'basin_ci95':np.quantile(samples,[.025,.975]),'huc2_cluster_ci95':np.quantile(spatial,[.025,.975]),
            'synchronous_water_year_ci95':np.quantile(td,[.025,.975])}
    associations={}
    x=deltas['raw_nse'].to_numpy()
    for key in ['raw_high_picp','global_high_picp','raw_winkler','global_winkler']:
        y=deltas[key].to_numpy();rho=float(spearmanr(x,y).statistic);draws=[];spatial=[]
        for _ in range(2000):
            ids=rng.integers(0,len(x),len(x));draws.append(spearmanr(x[ids],y[ids]).statistic)
            picks=rng.choice(np.unique(regions),len(np.unique(regions)),replace=True)
            ids=np.concatenate([np.flatnonzero(regions==r) for r in picks]);spatial.append(spearmanr(x[ids],y[ids]).statistic)
        associations[key]={'spearman_rho':rho,'basin_ci95':np.nanquantile(draws,[.025,.975]),
                           'huc2_cluster_ci95':np.nanquantile(spatial,[.025,.975]),'exploratory':True}
    # Seed-averaged basin-band summaries use one declared averaging order.
    band={}
    for year in YEARS:
        for method in ['raw','global','predicted','per_basin']:
            v=bf[(bf.year==year)&(bf.method==method)].groupby('basin').picp.mean()
            band[f'{year}_{method}']={'fraction_in_085_095':float(v.between(.85,.95).mean()),
                                     'min':float(v.min()),'max':float(v.max()),'sd':float(v.std())}
    summary={'protocol':{'observed_cuts':PRIMARY_CUTS,'boundary':'right-closed; high strictly > upper threshold',
        'predicted_thresholds':'within basin; fitted on calibration predictions only',
        'predicted_radius_pools':'relative predicted strata pooled across calibration basins',
        'tail_rule':'within-basin strict quantile exceedance; ties retained in lower group',
        'aggregation':'pooled basin-days per seed then mean across seeds; endpoint effects separately equal-basin weighted',
        'bootstrap_conditioning':'fixed trained models and selected historical periods',
        'partial_test_water_year':'WY2009 includes October-December 2008 only','n_bootstrap':NBOOT},
        'runs':all_rows,'endpoint_effects':endpoints,'predicted_calibration_gains_1yr':gains,
        'basin_associations':associations,'basin_tolerance_band':band}
    dump(BASE/'tables/coverage_summary.json',summary)
    pd.DataFrame({k:v for k,v in deltas.items()}).rename_axis('basin').to_csv(BASE/'tables/paired_basin_effects.csv')
    print('Coverage and clustered inference complete',flush=True)

if __name__=='__main__':main()
