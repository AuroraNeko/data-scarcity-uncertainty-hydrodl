"""Separate no-local-observation transfer from locally informed calibration."""
from pathlib import Path
import gc,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import conformal_radius,scores,interval_metrics,project_nonempty
from src.evaluation.protocol import dump

def main():
    rows=[]
    for year in [1,15]:
        for seed in [42,123,456]:
            d=np.load(ROOT/f'results/predictions/ungauged_scarcity/scarcity_{year}yr_unseen_modelseed{seed}.npz')
            cm=d['val_mask']>0;tm=d['test_mask']>0;cq=d['val_prediction_log'][cm].astype(float);cy=d['val_target_log'][cm].astype(float)
            dates=d['val_target_dates'][cm];q=d['test_prediction_log'][tm].astype(float);y=d['test_target_log'][tm].astype(float)
            seen=json.loads((ROOT/f'results/tables/corrected_scarcity/scarcity_{year}yr_basinseed42_modelseed{seed}.json').read_text())['q_cal_log']
            lo,hi,empty=project_nonempty(q[:,0]-seen,q[:,2]+seen)
            rows.append({'training_years':year,'seed':seed,'local_calibration_years':0,'mode':'seen_radius_transfer',
                         'n_local_calibration':0,'radius':seen,'empty_projected':empty,**interval_metrics(lo,hi,y)})
            sc=scores(cq[:,0],cq[:,2],cy)
            for n_year in [1,2,3,4,5]:
                fit=dates>=np.datetime64(f'{2000-n_year}-10-01');r=conformal_radius(sc[fit]);lo,hi,empty=project_nonempty(q[:,0]-r,q[:,2]+r)
                rows.append({'training_years':year,'seed':seed,'local_calibration_years':n_year,'mode':'pooled_unseen_observations',
                             'n_local_calibration':int(fit.sum()),'radius':r,'empty_projected':empty,**interval_metrics(lo,hi,y)})
            d.close();del cq,cy,q,y,sc;gc.collect();print('Unseen calibration budget',year,seed,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results/tables/unseen_calibration_windows.csv',index=False)
    records=[]
    for (year,n_year,mode),g in frame.groupby(['training_years','local_calibration_years','mode']):
        records.append({'training_years':year,'local_calibration_years':n_year,'mode':mode,'n_local_calibration':int(g.n_local_calibration.iloc[0]),
                        **{c+'_mean':float(g[c].mean()) for c in ['picp','mpiw','winkler']},'picp_sd':float(g.picp.std())})
    dump(ROOT/'results/tables/unseen_calibration_summary.json',{'rows':records,'n_unseen_basins':621,
        'definition':'Local observations are pooled across all 621 model-unseen basins; local means the evaluation basins supply calibration observations, not per-basin radii.',
        'interpretation':'Only the transferred seen-basin radius uses no local flow observations. Local recalibration is a different information setting and is not genuinely ungauged deployment.'})
    print('Unseen calibration extension complete',flush=True)

if __name__=='__main__':main()
