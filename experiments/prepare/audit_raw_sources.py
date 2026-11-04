"""Independently verify raw forcings, flow units, masks and target transforms.

The raw values are parsed positionally rather than through the preprocessing
reader. File-index helpers only locate upstream files. No model is refitted.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.data_preprocessing import _build_forcing_index,_build_streamflow_index
from src.evaluation.protocol import dump

def main():
    out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True)
    forcing_rows=[];target_rows=[];sources=['daymet','maurer','nldas'];cols=['prcp','srad','tmax','tmin','vp']
    indices={s:_build_forcing_index(s) for s in sources};flow_index=_build_streamflow_index()
    stats=json.loads((ROOT/'data/metadata/normalization_stats.json').read_text(encoding='utf-8'))
    files=sorted((ROOT/'data/processed/camels_us').glob('*.csv'))
    if not files:raise FileNotFoundError('Preprocess CAMELS-US before auditing raw sources')
    for i,p in enumerate(files,1):
        use=['date',*[f'{c}_{s}' for s in sources for c in cols],'streamflow_cfs','streamflow_mm','flow_mask','target','target_norm','area_gages2']
        d=pd.read_csv(p,usecols=use,parse_dates=['date']).set_index('date')
        for source in sources:
            raw_path=indices[source][p.stem]
            with raw_path.open(encoding='utf-8') as handle:header=[next(handle) for _ in range(4)][-1].split()
            assert len(header)>=7 and header[-3].lower().startswith('tmax') and header[-2].lower().startswith('tmin')
            values=np.loadtxt(raw_path,skiprows=4)
            assert values.shape[1]==11 and np.all((values[:,1]>=1)&(values[:,1]<=12))
            dates=pd.to_datetime(dict(year=values[:,0].astype(int),month=values[:,1].astype(int),day=values[:,2].astype(int)))
            raw=pd.DataFrame(values[:,[5,6,8,9,10]],columns=cols,index=dates).reindex(d.index)
            a=raw.to_numpy();saved=d[[f'{c}_{source}' for c in cols]].to_numpy();diff=np.abs(a-saved)
            row={'basin':p.stem,'source':source,'n_days':len(raw),'raw_nonfinite':int((~np.isfinite(a)).sum()),
                 'raw_sentinel_values':int(np.isin(a,[-999.,-9999.]).sum()),'max_abs_processed_difference':float(np.nanmax(diff)),
                 'n_difference_over_1e_minus6':int((diff>1e-6).sum()),'unequal_tmax_tmin_days':int((raw.tmax!=raw.tmin).sum())}
            assert row['raw_nonfinite']==row['raw_sentinel_values']==row['n_difference_over_1e_minus6']==0,row
            forcing_rows.append(row)
        original=pd.read_csv(flow_index[p.stem],sep=r'\s+',header=None,names=['basin','year','month','day','q','qc'])
        dates=pd.to_datetime(dict(year=original.year,month=original.month,day=original.day))
        q=pd.Series(original.q.to_numpy(dtype=float),index=dates).reindex(d.index).to_numpy(copy=True);q[q<0]=np.nan
        mask=d.flow_mask.to_numpy()>0;assert np.array_equal(np.isfinite(q),mask),(p.stem,'flow mask')
        physical=q[mask]*.0283168*86400/(float(d.area_gages2.iloc[0])*1e6)*1000
        log=np.log1p(physical);affine=(log-stats['target_mean'])/stats['target_std']
        row={'basin':p.stem,'n_observed':int(mask.sum()),'max_cfs_difference':float(np.max(abs(q[mask]-d.streamflow_cfs.to_numpy()[mask]))),
             'max_mmday_difference':float(np.max(abs(physical-d.streamflow_mm.to_numpy()[mask]))),
             'max_log1p_difference':float(np.max(abs(log-d.target.to_numpy()[mask]))),
             'max_affine_difference':float(np.max(abs(affine-d.target_norm.to_numpy()[mask])))}
        assert max(v for k,v in row.items() if k.startswith('max_'))<1e-8,row
        target_rows.append(row)
        if i%100==0:print('Independent raw-source audit',i,len(files),flush=True)
    f=pd.DataFrame(forcing_rows);t=pd.DataFrame(target_rows)
    f.to_csv(out/'raw_forcing_audit.csv',index=False);t.to_csv(out/'target_transform_audit.csv',index=False)
    totals=f.groupby('source').agg(n_basins=('basin','size'),n_days=('n_days','sum'),raw_nonfinite=('raw_nonfinite','sum'),
        raw_sentinel_values=('raw_sentinel_values','sum'),max_abs_processed_difference=('max_abs_processed_difference','max'),
        n_difference_over_1e_minus6=('n_difference_over_1e_minus6','sum'),unequal_tmax_tmin_days=('unequal_tmax_tmin_days','sum'))
    dump(out/'raw_forcing_audit_summary.json',totals.to_dict(orient='index'))
    dump(out/'target_transform_audit_summary.json',{'n_basins':len(t),'n_observed':int(t.n_observed.sum()),
        'max_absolute_differences':t.filter(like='max_').max().to_dict(),'raw_observation_masks_match':True,
        'cfs_to_m3s_factor':.0283168,'target_scale':'log1p(mm/day)','affine_source':'data/metadata/normalization_stats.json'})
    print('All raw forcing, physical-unit and transform checks passed',len(files),'basins',flush=True)

if __name__=='__main__':main()
