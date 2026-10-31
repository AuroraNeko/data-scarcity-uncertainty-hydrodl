"""Causal sequential, adaptive and asymmetric variants under the shared scorer."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import conformal_radius,scores,interval_metrics,project_nonempty
from src.corrections.common import load
from src.evaluation.protocol import dump
BASE=ROOT/'results'

def main():
    rows=[];daily=[];asymrows=[]
    for year in [1,3,5,15]:
        for seed in [42,123,456]:
            c,t=load(year,seed);cal_scores=scores(c['q'][:,0],c['q'][:,2],c['y']);global_r=conformal_radius(cal_scores)
            combined_q=np.concatenate([c['q'],t['q']]);combined_y=np.concatenate([c['y'],t['y']]);combined_dates=np.concatenate([c['date'],t['date']])
            order=np.argsort(combined_dates,kind='stable');aq,ay,ad=combined_q[order],combined_y[order],combined_dates[order]
            sc=scores(aq[:,0],aq[:,2],ay);days=np.unique(t['date']);start=np.searchsorted(ad,days,side='left');end=np.searchsorted(ad,days,side='right')
            test_day=np.searchsorted(days,t['date']);radii={'raw':np.zeros(len(t['y'])),'global':np.full(len(t['y']),global_r)}
            for scheme,window in [('roll90',90),('roll180',180),('roll365',365),('expand',None)]:
                lower=np.zeros(len(days),dtype=int) if window is None else np.searchsorted(ad,days-np.timedelta64(window,'D'),side='left')
                rr=np.asarray([conformal_radius(sc[lo:hi]) for lo,hi in zip(lower,start)])
                radii[scheme]=rr[test_day]
            lower365=np.searchsorted(ad,days-np.timedelta64(365,'D'),side='left');alpha_info={}
            for gamma in [.002,.005,.01,.02]:
                alpha=.1;rr=[];alphas=[]
                for lo,hi,last in zip(lower365,start,end):
                    radius=conformal_radius(sc[lo:hi],alpha);rr.append(radius)
                    ll,uu,_=project_nonempty(aq[hi:last,0]-radius,aq[hi:last,2]+radius)
                    miss=float(((ay[hi:last]<ll)|(ay[hi:last]>uu)).mean())
                    alpha=float(np.clip(alpha+gamma*(.1-miss),1e-4,.5));alphas.append(alpha)
                key=f'aci_g{gamma}';radii[key]=np.asarray(rr)[test_day]
                alpha_info[key]={'alpha_min':min(alphas),'alpha_max':max(alphas),'alpha_final':alphas[-1]}
            for scheme,r in radii.items():
                lo,hi,nempty=project_nonempty(t['q'][:,0]-r,t['q'][:,2]+r)
                rows.append({'year':year,'seed':seed,'scheme':scheme,'mean_radius':float(r.mean()),'empty_projected':nempty,
                             **alpha_info.get(scheme,{}),**interval_metrics(lo,hi,t['y'])})
                if year in [1,15] and scheme in ['raw','global','roll180','expand']:
                    hits=((t['y']>=lo)&(t['y']<=hi)).astype(float)
                    num=np.bincount(test_day,weights=hits,minlength=len(days));den=np.bincount(test_day,minlength=len(days))
                    daily.extend([{'year':year,'seed':seed,'scheme':scheme,'date':str(d),'hits':h,'n':n} for d,h,n in zip(days,num,den)])
            rlow=conformal_radius(c['q'][:,0]-c['y'],.05);rup=conformal_radius(c['y']-c['q'][:,2],.05)
            lo,hi,nempty=project_nonempty(t['q'][:,0]-rlow,t['q'][:,2]+rup)
            asymrows.append({'year':year,'seed':seed,'scheme':'asym','r_low':rlow,'r_up':rup,'empty_projected':nempty,**interval_metrics(lo,hi,t['y'])})
            print('Causal calibration variants',year,seed,flush=True)
    frame=pd.DataFrame(rows);af=pd.DataFrame(asymrows);df=pd.DataFrame(daily)
    frame.to_csv(BASE/'tables/sequential_calibration_variants.csv',index=False);af.to_csv(BASE/'tables/asymmetric_calibration.csv',index=False);df.to_csv(BASE/'tables/sequential_daily_coverage.csv',index=False)
    dump(BASE/'tables/calibration_variants_summary.json',{'schemes':frame.groupby(['year','scheme'])[['picp','mpiw','winkler','empty_projected']].mean().reset_index().to_dict(orient='records'),
        'asymmetric':af.groupby('year')[['r_low','r_up','picp','mpiw','winkler','p_below','p_above','empty_projected']].mean().reset_index().to_dict(orient='records'),
        'time_order':'Each radius uses only observations strictly before the forecast target date; ACI updates after all current-day outcomes are revealed.',
        'information_budget':'Sequential rules additionally require observed flows arriving during testing; static schemes do not.',
        'aci':'Clipped daily pooled feedback, alpha in [0.0001,0.5]; results are empirical, not a new guarantee.'})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for ax,year in zip(axes,[15,1]):
        for scheme,label,color in [('raw','Raw QR','#999999'),('global','Global CQR','#345b78'),('roll180','Rolling 180 days','#b66d35'),('expand','Expanding','#6d6a96')]:
            curves=[]
            for seed in [42,123,456]:
                f=df[(df.year==year)&(df.seed==seed)&(df.scheme==scheme)].copy();f.index=pd.to_datetime(f.date)
                curves.append(f.hits.rolling(365,center=True,min_periods=365).sum()/f.n.rolling(365,center=True,min_periods=365).sum())
            average=pd.concat(curves,axis=1).mean(axis=1);ax.plot(average.index,average,label=label,color=color)
        ax.axhline(.9,color='#777777',ls=':');ax.set_title(f'{year}-year training labels');ax.set_ylabel('365-day pooled PICP');ax.tick_params(axis='x',labelrotation=30)
    axes[0].legend(frameon=False,fontsize=8);fig.savefig(BASE/'figures/revision_rolling_calibration.pdf',bbox_inches='tight');fig.savefig(BASE/'figures/revision_rolling_calibration.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print('Causal calibration variants complete',flush=True)

if __name__=='__main__':main()
