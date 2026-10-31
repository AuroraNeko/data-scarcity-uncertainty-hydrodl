"""Summarize declared sensitivity fits; can be re-run as the queue progresses."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import (conformal_radius,scores,interval_metrics,project_nonempty,
                                   predicted_group_intervals,observed_groups)
from src.evaluation.protocol import dump,nse,bootstrap_basin
from src.sensitivity import specs
BASE=ROOT/'results';SEEDS=[42,123,456]

def evaluate(arrays,spec,variant='final',fullcal=False):
    prefix='fullcal' if fullcal else 'calibration';cm=arrays[f'{prefix}_mask']>0;tm=arrays['test_mask']>0
    cq=arrays[f'{prefix}_{variant}_quantiles_log'][cm].astype(float);tq=arrays[f'test_{variant}_quantiles_log'][tm].astype(float)
    cy=arrays[f'{prefix}_y_log'][cm].astype(float);y=arrays['test_y_log'][tm].astype(float)
    cb=arrays[f'{prefix}_basin'][cm];b=arrays['test_basin'][tm];taus=arrays['taus'];median=int(np.argmin(abs(taus-.5)))
    levels=[.5,.8,.9,.95,.99] if len(taus)>3 else [.9];rows=[];basin=[]
    obs=observed_groups(y,b)
    for level in levels:
        alpha=1-level;i=int(np.argmin(abs(taus-alpha/2)));j=int(np.argmin(abs(taus-(1-alpha/2))))
        assert abs(taus[i]-alpha/2)<1e-8 and abs(taus[j]-(1-alpha/2))<1e-8
        r=conformal_radius(scores(cq[:,i],cq[:,j],cy),alpha)
        lo,hi,n_empty=project_nonempty(tq[:,i]-r,tq[:,j]+r)
        predicted_lo,predicted_hi,info=predicted_group_intervals(cq[:,[i,median,j]],cy,cb,tq[:,[i,median,j]],b,alpha)
        for method,lower,upper in [('raw',tq[:,i],tq[:,j]),('global',lo,hi),('predicted',predicted_lo,predicted_hi)]:
            pooled=interval_metrics(lower,upper,y,alpha)
            rows.append({'job':spec['id'],'variant':variant,'nominal':level,'method':method,'seed':spec['seed'],
                'radius_log':r,'projected_empty':0 if method=='raw' else (n_empty if method=='global' else info['empty_intervals_projected']),
                'median_basin_nse':np.median([nse(y[b==bb],tq[b==bb,median]) for bb in np.unique(b)]),
                'observed_high_picp':float(((y>=lower)&(y<=upper))[obs==2].mean()),
                'predicted_high_picp':float(((y>=lower)&(y<=upper))[info['test_groups']==2].mean()),**pooled})
            for bb in np.unique(b):
                sel=b==bb
                basin.append({'job':spec['id'],'variant':variant,'nominal':level,'method':method,'seed':spec['seed'],'basin':bb,
                    'nse':nse(y[sel],tq[sel,median]),**interval_metrics(lower[sel],upper[sel],y[sel],alpha)})
    return rows,basin

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    declared=specs();available=[];missing=[];metrics=[];basins=[];histories={}
    for spec in declared:
        p=BASE/f'tables/training_{spec["id"]}.json'
        if not p.exists():missing.append(spec['id']);continue
        meta=json.loads(p.read_text());assert meta['completed'] and meta['spec']==spec
        available.append(spec['id']);histories[spec['id']]=meta['history']
        arrays=np.load(ROOT/meta['predictions'])
        for variant in ['final','selected'] if spec['validation'] else ['final']:
            m,b=evaluate(arrays,spec,variant);metrics.extend(m);basins.extend(b)
        if spec['id'].startswith('validation_1yr_') and 'fullcal_final_quantiles_log' in arrays.files:
            ref=dict(spec,id=f'temporal_wy1995_s{spec["seed"]}')
            m,b=evaluate(arrays,ref,'final',fullcal=True);metrics.extend(m);basins.extend(b)
        arrays.close()
    mf=pd.DataFrame(metrics);bf=pd.DataFrame(basins)
    mf.to_csv(BASE/'tables/revision_training_metrics.csv',index=False);bf.to_csv(BASE/'tables/revision_training_basin_metrics.csv',index=False)
    if args.require_complete and missing:raise RuntimeError(f'{len(missing)} training fits still missing')
    if not len(mf):dump(BASE/'tables/revision_training_summary.json',{'available':available,'missing':missing});return
    names=pd.read_csv(ROOT/'data/raw/camels_us/camels_name.txt',sep=';',dtype=str).set_index('gauge_id')
    condition=[]
    for row in metrics:
        group=row['job'].rsplit('_s',1)[0]
        if group.startswith('temporal'):family='temporal';years=1
        elif group.startswith('validation'):family='validation_equal' if 'equal' in group else 'validation';years=int(group.rsplit('_',1)[-1].replace('yr',''))
        else:family=group.split('_')[0];years=int(group.rsplit('_',1)[-1].replace('yr',''))
        condition.append({'group':group,'family':family,'years':years,**row})
    ff=pd.DataFrame(condition)
    summary_rows=[]
    cols=['median_basin_nse','picp','mpiw','winkler','observed_high_picp','predicted_high_picp']
    for key,g in ff.groupby(['group','family','years','variant','nominal','method']):
        rec=dict(zip(['group','family','years','variant','nominal','method'],key));rec['n_seeds']=len(g)
        for c in cols:rec[c+'_mean']=g[c].mean();rec[c+'_sd']=g[c].std()
        summary_rows.append(rec)
    pd.DataFrame(summary_rows).to_csv(BASE/'tables/revision_training_conditions.csv',index=False)
    comparisons={}
    for family in ['validation','validation_equal','multilevel','tcn']:
        for variant in ['final','selected'] if family.startswith('validation') else ['final']:
            for level in ([.5,.8,.9,.95,.99] if family=='multilevel' else [.9]):
                for method in ['raw','global']:
                    ids={yr:[f'{family}_{yr}yr_s{s}' for s in SEEDS] for yr in [1,15]}
                    if family=='validation_equal':ids[1]=[f'validation_1yr_s{s}' for s in SEEDS]
                    if not all(job in available for jobs in ids.values() for job in jobs):continue
                    f=bf[(bf.variant==variant)&np.isclose(bf.nominal,level)&(bf.method==method)]
                    for metric in ['nse','picp','winkler']:
                        short=f[f.job.isin(ids[1])].groupby('basin')[metric].mean();long=f[f.job.isin(ids[15])].groupby('basin')[metric].mean()
                        bs=np.sort(short.index.to_numpy());delta=(long-short).loc[bs].to_numpy();regions=names.loc[bs,'huc_02'].to_numpy()
                        comparisons[f'{family}_{variant}_{level}_{method}_{metric}']=bootstrap_basin(delta,bs,regions,'median' if metric=='nse' else 'mean')
    # Temporal composition against the fixed-epoch, common-loader WY1995 fit.
    temporal={}
    refs=[f'temporal_wy1995_s{s}' for s in SEEDS]
    if all(job in set(bf.job) for job in refs):
        f=bf[(bf.variant=='final')&(bf.method=='raw')&np.isclose(bf.nominal,.9)]
        ref=f[f.job.isin(refs)].groupby('basin')[['nse','picp','winkler']].mean()
        for wy in [1985,1990,1991,1992,1993,1994]:
            ids=[f'temporal_wy{wy}_s{s}' for s in SEEDS]
            if not all(job in available for job in ids):continue
            v=f[f.job.isin(ids)].groupby('basin')[['nse','picp','winkler']].mean();bs=np.sort(ref.index.to_numpy());regions=names.loc[bs,'huc_02'].to_numpy()
            temporal[str(wy)]={m:bootstrap_basin((v[m]-ref[m]).loc[bs].to_numpy(),bs,regions,'median' if m=='nse' else 'mean') for m in ['nse','picp','winkler']}
    temporal_long={}
    original=pd.read_csv(BASE/'tables/basin_metrics.csv',dtype={'basin':str})
    longref=original[(original.year==15)&(original.method=='raw')].groupby('basin')[['nse','picp','winkler']].mean()
    f=bf[(bf.variant=='final')&(bf.method=='raw')&np.isclose(bf.nominal,.9)]
    for wy in [1985,1990,1991,1992,1993,1994,1995]:
        ids=[f'temporal_wy{wy}_s{s}' for s in SEEDS]
        if not all(job in set(f.job) for job in ids):continue
        v=f[f.job.isin(ids)].groupby('basin')[['nse','picp','winkler']].mean();bs=np.sort(longref.index.to_numpy());regions=names.loc[bs,'huc_02'].to_numpy()
        temporal_long[str(wy)]={m:bootstrap_basin((longref[m]-v[m]).loc[bs].to_numpy(),bs,regions,'median' if m=='nse' else 'mean') for m in ['nse','picp','winkler']}
    memory={}
    for years in [1,15]:
        ids180=[f'memory180_15yr_s{s}' for s in SEEDS] if years==15 else [f'temporal_wy1995_s{s}' for s in SEEDS]
        ids365=[f'memory365_{years}yr_s{s}' for s in SEEDS]
        if not all(job in set(bf.job) for job in ids180+ids365):continue
        for method in ['raw','global']:
            f=bf[(bf.variant=='final')&(bf.method==method)&np.isclose(bf.nominal,.9)]
            a=f[f.job.isin(ids180)].groupby('basin')[['nse','picp','mpiw','winkler']].mean()
            b=f[f.job.isin(ids365)].groupby('basin')[['nse','picp','mpiw','winkler']].mean()
            bs=np.sort(a.index.to_numpy());regions=names.loc[bs,'huc_02'].to_numpy()
            memory[f'{years}yr_{method}']={m:bootstrap_basin((b[m]-a[m]).loc[bs].to_numpy(),bs,regions,'median' if m=='nse' else 'mean') for m in ['nse','picp','mpiw','winkler']}
    dump(BASE/'tables/revision_training_summary.json',{'declared_jobs':len(declared),'available':available,'missing':missing,
        'conditions':summary_rows,'endpoint_comparisons':comparisons,'alternative_year_minus_wy1995':temporal,
        'archived_15yr_minus_alternative_year':temporal_long,'matched_target_memory_effects_365_minus_180':memory,'histories':histories,
        'notes':['All inference is fixed before looking at test results.','Validation-selected models use 1995-1996 only; their CQR pool is 1996-2000.',
                 'Temporal experiments retain a fixed 1995-2000 CQR pool; earlier training years also change recency.',
                 'The WY1995 temporal reference reuses fixed-final-epoch validation-model weights, not validation-selected weights.',
                 'Multilevel extension changes the quantile head; it is an endpoint sensitivity, not a reuse of the three-quantile distribution.']})
    print('Training analysis:',len(available),'of',len(declared),'fits complete',flush=True)

if __name__=='__main__':main()
