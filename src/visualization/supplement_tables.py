"""Generate the retained supplementary tables from compact results and arrays."""
import json
import numpy as np
import pandas as pd
from src.corrections.common import load
from src.evaluation.protocol import conformal_radius,scores,interval_metrics,project_nonempty
from src.data.dataset import STATIC_COLS

def supplementary_tables(root,emit):
    tables=root/'results/tables'
    read=lambda name:json.loads((tables/name).read_text(encoding='utf-8'))
    num=lambda v:f'{v:.3f}'
    pm=lambda d:f'{d["mean"]:.3f} $\\pm$ {d["sd"]:.3f}'
    # Architecture and optimization controls, including all model-seed SDs.
    architecture=read('scarcity_standard_lstm/architecture_sensitivity_summary.json')['architectures']
    for zh in [False,True]:
        rows=[]
        for key,label in [('lpu_stream','LPU-Stream'),('standard_lstm','标准 LSTM' if zh else 'Standard LSTM')]:
            a=architecture[key]
            for year in [1,15]:
                d=a['by_year'][str(year)]
                rows.append([label,year,f'{a["n_parameters"]:,}',pm(d['median_nse']),*[num(d[k]['mean']) for k in ['raw_picp','cqr_picp','raw_mpiw_log','raw_winkler_log']]])
        emit('architecture_rows'+('_zh' if zh else '')+'.tex',rows)
    budgets=read('scarcity_equal_updates/equal_update_summary.json')['training_budgets']
    for zh in [False,True]:
        rows=[]
        for key,label in [('fixed_epochs','固定 30 个 epoch' if zh else 'Fixed 30 epochs'),('fixed_updates','固定 1,080 次更新' if zh else 'Fixed 1,080 updates')]:
            for year in [1,15]:
                d=budgets[key]['by_year'][str(year)];n=budgets[key]['optimizer_budget'][str(year)]['optimizer_steps']
                rows.append([label,year,f'{n:,}',pm(d['median_nse']),*[num(d[k]['mean']) for k in ['raw_picp','cqr_picp','raw_mpiw_log','raw_winkler_log']]])
        emit('equal_update_rows'+('_zh' if zh else '')+'.tex',rows)
    rep=read('scarcity_replications/scarcity_replication_summary.json')
    for zh in [False,True]:
        rows=[]
        for i,(key,data) in enumerate([*rep['by_subset'].items(),('combined',rep['combined_effects_15_minus_1'])]):
            label=('综合结果' if zh else 'Combined') if key=='combined' else (('主样本' if zh else 'Primary sample') if i==0 else (f'分层样本 {i}' if zh else f'Stratified sample {i}'))
            d=data if key=='combined' else data['effects_15_minus_1'];n=d['nse'];p=d['raw_picp']
            rows.append([label,f'{n["estimate"]:.3f} [{n["ci95"][0]:.3f}, {n["ci95"][1]:.3f}]',f'{100*n["fraction_basins_positive"]:.0f}\\%',
                f'{100*p["estimate"]:.1f} [{100*p["ci95"][0]:.1f}, {100*p["ci95"][1]:.1f}]',f'{100*p["fraction_basins_positive"]:.0f}\\%'])
        emit('replication_rows'+('_zh' if zh else '')+'.tex',rows)
    attrs=read('basin_representativeness.json')['attributes']
    chinese=['高程 (m)',r'坡度 (m km$^{-1}$)',r'流域面积 (km$^2$)',r'平均降水 (mm d$^{-1}$)',r'平均潜在蒸散 (mm d$^{-1}$)',
        '干旱度 (PET/P)','降雪比例','降水季节性','土壤深度 (m)','土壤孔隙度','森林比例','LAI 季节差异','地质孔隙度']
    for zh in [False,True]:
        rows=[]
        for index,key in enumerate(STATIC_COLS):
            d=attrs[key]
            rows.append([chinese[index] if zh else d['attribute'],f'{d["remaining621_mean"]:.3f} ({d["remaining621_std"]:.3f})',
                f'{d["scarcity50_mean"]:.3f} ({d["scarcity50_std"]:.3f})',num(d['ks_p_raw']),f'{d["cohens_d_subset_minus_remaining"]:.2f}'])
        emit('representativeness_rows'+('_zh' if zh else '')+'.tex',rows)
    climate=pd.read_csv(tables/'corrected_full_671/climate_quartiles.csv')
    zh_climate=['Q1 最湿润','Q2 湿润过渡','Q3 干旱过渡','Q4 最干旱','全部 CAMELS-US']
    for zh in [False,True]:
        rows=[]
        for i,d in climate.iterrows():
            rows.append([zh_climate[i] if zh else d.group,int(d.n_basins),f'{d.aridity_min:.2f}--{d.aridity_max:.2f}',
                *[num(d[k]) for k in ['median_basin_nse','picp','mpiw']]])
        emit('climate_rows'+('_zh' if zh else '')+'.tex',rows)
    unseen=read('ungauged_scarcity/ungauged_summary.json')['per_length'];rows=[]
    for year in [1,3,5,15]:
        d=unseen[f'{year}yr'];rows.append([year,*[f'{d[k][0]:.3f} $\\pm$ {d[k][1]:.3f}' for k in ['nse_median','raw_picp','transfer_picp','local_picp','raw_winkler','local_winkler']]])
    emit('ungauged_rows.tex',rows)
    # Rebuild descriptive hydroclimatic strata and aridity calibration pools.
    with np.load(root/'results/cache/primary_raw.npz') as raw:
        basins=raw['basins'];a=raw['static'];aridity=dict(zip(basins,a[:,STATIC_COLS.index('aridity')]))
        snow=dict(zip(basins,a[:,STATIC_COLS.index('frac_snow')]))
    c,_=load(15,42);flash={}
    for b in basins:
        sel=c['b']==b;order=np.argsort(c['date'][sel],kind='stable');dates=c['date'][sel][order]
        flow=np.maximum(np.expm1(c['y'][sel][order]),0);pair=np.diff(dates).astype('timedelta64[D]').astype(int)==1
        flash[b]=float(np.abs(np.diff(flow)[pair]).sum()/np.clip(flow[:-1][pair].sum(),1e-9,None))
    group_of=dict(zip(basins,np.asarray(pd.qcut([aridity[b] for b in basins],4,labels=False))))
    strata={'all':set(basins)}
    for name,values in [('arid',aridity),('snow',snow),('flashy',flash)]:
        labels=np.asarray(pd.qcut([values[b] for b in basins],4,labels=False));strata[name]=set(basins[labels==3])
    regime=[];quartiles=[]
    for year in [1,3,5,15]:
        for seed in [42,123,456]:
            c,t=load(year,seed);sc=scores(c['q'][:,0],c['q'][:,2],c['y']);r=conformal_radius(sc)
            cg=np.array([group_of[b] for b in c['b']]);tg=np.array([group_of[b] for b in t['b']]);qr=np.array([conformal_radius(sc[cg==g]) for g in range(4)])
            for method,rr in [('raw',0.),('global',r),('grouped',qr[tg])]:
                lo,hi,_=project_nonempty(t['q'][:,0]-rr,t['q'][:,2]+rr)
                if method!='grouped':
                    for name,picked in strata.items():
                        sel=np.isin(t['b'],list(picked));regime.append({'year':year,'seed':seed,'stratum':name,'method':method,**interval_metrics(lo[sel],hi[sel],t['y'][sel])})
                if method!='raw':
                    for g in range(4):
                        sel=tg==g;quartiles.append({'year':year,'seed':seed,'quartile':g,'method':method,**interval_metrics(lo[sel],hi[sel],t['y'][sel])})
    rf=pd.DataFrame(regime);qf=pd.DataFrame(quartiles);rf.to_csv(tables/'descriptive_group_metrics.csv',index=False);qf.to_csv(tables/'aridity_group_metrics.csv',index=False)
    rows=[]
    for key,label in [('all','All 50 basins'),('arid','Arid quartile'),('snow','Snow quartile'),('flashy','Flashy quartile')]:
        row=[label]
        for year in [1,15]:
            f=rf[(rf.year==year)&(rf.stratum==key)];raw=f[f.method=='raw'];cal=f[f.method=='global']
            row.extend([f'{raw.picp.mean():.3f} $\\pm$ {raw.picp.std():.3f}',f'{cal.picp.mean():.3f} $\\pm$ {cal.picp.std():.3f}',num(cal.p_above.mean()),num(cal.winkler.mean())])
        rows.append(row)
    emit('regime_rows.tex',rows)
    rows=[];matrix=[]
    for g in range(4):
        values=[aridity[b] for b in basins if group_of[b]==g];row=[f'Q{g+1}',f'{min(values):.2f}--{max(values):.2f}',len(values)];v=[]
        for method in ['global','grouped']:
            for year in [1,3,5,15]:
                value=qf[(qf.quartile==g)&(qf.method==method)&(qf.year==year)].picp.mean();row.append(num(value));v.append(value)
        rows.append(row);matrix.append(v)
    values=np.asarray(matrix);rows.append(['Spread (max--min)','','',*[num(v) for v in values.max(0)-values.min(0)]])
    emit('aridity_rows.tex',rows)
