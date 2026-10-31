"""Forward calibration-year blocks and per-basin score drift diagnostics."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import conformal_radius,scores,interval_metrics,project_nonempty
from src.corrections.common import load
from src.evaluation.protocol import dump

def main():
    rows=[];basins=[]
    for year in [1,3,5,15]:
        for seed in [42,123,456]:
            cal,test=load(year,seed);sc=scores(cal['q'][:,0],cal['q'][:,2],cal['y'])
            for fit_years in [1,2,3,4,5]:
                cutoff=np.datetime64(f'{1995+fit_years}-10-01');fit=cal['date']<cutoff
                radius=conformal_radius(sc[fit]);local={b:conformal_radius(sc[fit&(cal['b']==b)]) for b in np.unique(cal['b'])}
                evaluations={'fit_period':{k:v[fit] for k,v in cal.items()},'later_test':test}
                if fit_years<5:
                    held=(cal['date']>=cutoff)&(cal['date']<np.datetime64(f'{1996+fit_years}-10-01'))
                    evaluations['next_calibration_water_year']={k:v[held] for k,v in cal.items()}
                for period,data in evaluations.items():
                    for method in ['global','per_basin']:
                        rr=radius if method=='global' else np.asarray([local[b] for b in data['b']])
                        lo,hi,n_empty=project_nonempty(data['q'][:,0]-rr,data['q'][:,2]+rr)
                        rows.append({'training_years':year,'seed':seed,'fit_calibration_years':fit_years,'evaluation':period,'method':method,
                                     'projected_empty':n_empty,**interval_metrics(lo,hi,data['y'])})
                        if fit_years==5:
                            for b in np.unique(data['b']):
                                sel=data['b']==b
                                basins.append({'training_years':year,'seed':seed,'basin':b,'evaluation':period,'method':method,
                                    'radius':radius if method=='global' else local[b],**interval_metrics(lo[sel],hi[sel],data['y'][sel])})
            print('Forward calibration blocks',year,seed,flush=True)
    frame=pd.DataFrame(rows);bf=pd.DataFrame(basins)
    frame.to_csv(ROOT/'results/tables/forward_calibration_blocks.csv',index=False)
    bf.to_csv(ROOT/'results/tables/per_basin_calibration_drift.csv',index=False)
    summary=frame.groupby(['training_years','fit_calibration_years','evaluation','method'])[['picp','p_below','p_above','winkler']].mean().reset_index()
    dump(ROOT/'results/tables/calibration_block_summary.json',{'rows':summary.to_dict(orient='records'),
        'interpretation':'The exact finite-sample rank controls the fitted score order statistic; smaller pools do not by themselves imply systematic below-nominal future coverage. Forward held-out blocks expose realized temporal changes and dependence without reusing their outcomes for calibration.'})
    print('Forward calibration diagnostics complete',flush=True)

if __name__=='__main__':main()
