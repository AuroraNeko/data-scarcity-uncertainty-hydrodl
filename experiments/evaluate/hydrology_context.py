"""Describe missingness and historical-period differences; no model fitting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,wasserstein_distance

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import dump
BASE=ROOT/'results'
PERIODS={'train_1yr':('1994-10-01','1995-09-30'),'train_3yr':('1992-10-01','1995-09-30'),
 'train_5yr':('1990-10-01','1995-09-30'),'train_15yr':('1980-10-01','1995-09-30'),
 'calibration_5yr':('1995-10-01','2000-09-30'),'independent_validation':('1995-10-01','1996-09-30'),
 'calibration_4yr':('1996-10-01','2000-09-30'),'test':('2000-10-01','2008-12-31')}

def longest_true_run(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(delta==1);ends=np.flatnonzero(delta==-1)
    return int((ends-starts).max()) if len(starts) else 0

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=np.load(BASE/'cache/primary_raw.npz')['basins'];missing=[];context=[];annual=[];month_rows=[];files=sorted((ROOT/'data/processed/camels_us').glob('*.csv'))
    for number,p in enumerate(files,1):
        b=p.stem;df=pd.read_csv(p,usecols=['date','target','flow_mask','prcp_daymet'])
        dates=df['date'].to_numpy(dtype='datetime64[D]');y=df.target.to_numpy();mask=(df.flow_mask.to_numpy()>0)&np.isfinite(y)
        flow=np.maximum(np.expm1(np.nan_to_num(y)),0);precip=df.prcp_daymet.to_numpy()
        months=dates.astype('datetime64[M]').astype(int)%12+1;wy=dates.astype('datetime64[Y]').astype(int)+1970+(months>=10)
        cal=(dates>=np.datetime64('1995-10-01'))&(dates<=np.datetime64('2000-09-30'))&mask
        q10,q95,q99=np.quantile(flow[cal],[.1,.95,.99]);test=(dates>=np.datetime64('2000-10-01'))&(dates<=np.datetime64('2008-12-31'))&mask
        for label,(start,end) in PERIODS.items():
            sel=(dates>=np.datetime64(start))&(dates<=np.datetime64(end));valid=sel&mask
            missing.append({'basin':b,'primary':b in primary,'period':label,'n_grid':int(sel.sum()),'n_observed':int(valid.sum()),
                'n_missing':int((sel&~mask).sum()),'missing_fraction':float((~mask[sel]).mean()),'longest_missing_run':longest_true_run(~mask[sel])})
            v=flow[valid]
            context.append({'basin':b,'primary':b in primary,'period':label,'n':len(v),'mean_flow':float(v.mean()),
                'q05':float(np.quantile(v,.05)),'q50':float(np.quantile(v,.5)),'q95':float(np.quantile(v,.95)),
                'q99':float(np.quantile(v,.99)),'max':float(v.max()),'zero_flow_fraction':float((v==0).mean()),
                'below_calibration_q10_fraction':float((v<=q10).mean()),'above_calibration_q95_fraction':float((v>q95).mean()),
                'above_calibration_q99_fraction':float((v>q99).mean()),'mean_daily_precip':float(precip[sel].mean()),
                'wet_day_fraction':float((precip[sel]>1).mean()),
                'calibration_test_ks_statistic':float(ks_2samp(flow[cal],flow[test]).statistic) if label=='test' else None,
                'calibration_test_wasserstein_log':float(wasserstein_distance(y[cal],y[test])) if label=='test' else None})
        for period in np.unique(wy):
            sel=wy==period;valid=sel&mask;v=flow[valid]
            annual.append({'basin':b,'primary':b in primary,'water_year':int(period),'n_grid':int(sel.sum()),'n_missing':int((sel&~mask).sum()),
                'n_observed':int(valid.sum()),'mean_flow':float(v.mean()) if len(v) else None,
                'q95_flow':float(np.quantile(v,.95)) if len(v) else None,'max_flow':float(v.max()) if len(v) else None,
                'high_flow_frequency_cal_q95':float((v>q95).mean()) if len(v) else None,'mean_precip':float(precip[sel].mean())})
        for m in range(1,13):
            sel=months==m
            month_rows.append({'basin':b,'month':m,'n_grid':int(sel.sum()),'n_missing':int((sel&~mask).sum())})
        if number%100==0:print('Historical context',number,len(files),flush=True)
    mf=pd.DataFrame(missing);cf=pd.DataFrame(context);af=pd.DataFrame(annual)
    mf.to_csv(BASE/'tables/missingness_by_basin_period.csv',index=False)
    cf.to_csv(BASE/'tables/hydrological_periods_by_basin.csv',index=False)
    af.to_csv(BASE/'tables/hydrological_water_years.csv',index=False)
    pd.DataFrame(month_rows).to_csv(BASE/'tables/missingness_by_basin_month.csv',index=False)
    summaries={}
    for label,flag in [('primary_50',True),('all_671',None)]:
        mm=mf[mf.primary] if flag else mf;cc=cf[cf.primary] if flag else cf
        miss=mm.groupby('period')[['n_grid','n_observed','n_missing']].sum()
        hyd=cc.groupby('period')[['mean_flow','q95','q99','zero_flow_fraction','below_calibration_q10_fraction',
                                  'above_calibration_q95_fraction','above_calibration_q99_fraction','mean_daily_precip']].median()
        summaries[label]={'missingness':miss.to_dict(orient='index'),'median_basin_hydrology':hyd.to_dict(orient='index'),
                         'test_period_ks_median':cc[cc.period=='test'].calibration_test_ks_statistic.median()}
    dump(BASE/'tables/hydrology_context_summary.json',{'groups':summaries,'notes':[
        'Thresholds q10/q95/q99 are fitted per basin on 1995-2000 calibration observations and kept fixed across periods.',
        'Low-flow frequency includes ties; zero-flow basins can exceed 10% at q10=0.',
        'High-flow events strictly exceed reference thresholds; these are flow-duration diagnostics, not return-period floods.',
        'KS statistics are descriptive; no iid two-sample p values are interpreted.',
        'WY1980 and WY2009 are partial in the processed/test archive. Missing target flows are unknown, so missingness is not claimed random.']})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    heat=af[af.primary].pivot(index='basin',columns='water_year',values='n_missing').reindex(sorted(primary))
    im=axes[0].imshow(heat,aspect='auto',interpolation='nearest',cmap='magma_r');axes[0].set_xlabel('Water year');axes[0].set_ylabel('Primary basins (sorted ID)')
    ticks=np.arange(0,heat.shape[1],5);axes[0].set_xticks(ticks,heat.columns[ticks]);fig.colorbar(im,ax=axes[0],label='Missing target days')
    annual50=af[af.primary&af.water_year.between(1981,2008)]
    agg=annual50.groupby('water_year').high_flow_frequency_cal_q95.agg(['median',lambda v:v.quantile(.25),lambda v:v.quantile(.75)])
    axes[1].plot(agg.index,agg['median'],color='#225e78');axes[1].fill_between(agg.index,agg.iloc[:,1],agg.iloc[:,2],alpha=.2,color='#225e78')
    axes[1].axvspan(1995.5,2000.5,color='#d79b43',alpha=.15,label='Calibration');axes[1].axvspan(2000.5,2008.5,color='#225e78',alpha=.08,label='Full test water years')
    axes[1].set_ylabel('Fraction above basin calibration Q95');axes[1].set_xlabel('Water year');axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(BASE/'figures/historical_context.png',dpi=220);plt.close(fig)
    print('Historical context complete',flush=True)

if __name__=='__main__':main()
