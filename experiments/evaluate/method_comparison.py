"""Main 671-basin method comparison.

Reads the archived calibration/test predictions of every interval method
(quantile baselines, MDN, Gaussian head, MC-dropout variants, memberwise
split-CQR), fits one global conformal radius per method on the calibration
window, and scores raw and calibrated intervals on the shared test window.
Uncertainty is quantified with basin and HUC2-region bootstrap resampling;
timing comes from the archived dense-grid reliability table.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import conformal_radius,scores,interval_arrays,interval_metrics,project_nonempty
from src.evaluation.protocol import dump,nse
BASE=ROOT/'results';ARCH=ROOT/'results/predictions/corrected_full_671'

def main():
    test=np.load(ARCH/'test_predictions.npz');cal=np.load(ARCH/'calibration_predictions.npz');idx=np.load(ARCH/'evaluation_index.npz')
    norm=json.loads((ROOT/'data/metadata/normalization_stats.json').read_text());mu,sd=norm['target_mean'],norm['target_std']
    tm=test['mask']>0;cm=cal['mask']>0;y=test['target'][tm].astype(float)*sd+mu;cy=cal['target'][cm].astype(float)*sd+mu
    b=test['basin_index'][tm];names=idx['basin_ids'];basin_names=pd.read_csv(ROOT/'data/raw/camels_us/camels_name.txt',sep=';',dtype=str).set_index('gauge_id')
    regions=basin_names.loc[names,'huc_02'].to_numpy();rng=np.random.default_rng(20261030)
    draw=rng.integers(0,671,(2000,671));unique=np.unique(regions);rids=[np.flatnonzero(regions==r) for r in unique]
    spatial=[np.concatenate([rids[j] for j in rng.integers(0,len(unique),len(unique))]) for _ in range(2000)]
    timing=json.loads((ROOT/'results/tables/benchmark_671/dense_reliability.json').read_text())['timing']
    methods=[];sufficient={};paired={}
    entries=[('mc_dropout','MC Dropout, archived spread',50),('single_qr','Single QR',1),('ensemble','Mean-quantile ensemble',5),
             ('mc_dropout_noise','MC Dropout + Gaussian noise',50),('mc_dropout_reselect','MC Dropout, tail-selected',50),
             ('mdn','Gaussian MDN (K=5)',1),('umal','Gaussian quantile head',1),('memberwise_cqr','Memberwise split-CQR average',5)]
    for key,label,passes in entries:
        if key in ['mc_dropout','single_qr','ensemble','memberwise_cqr']:
            source='ensemble' if key=='memberwise_cqr' else key
            q=test[source][tm].astype(float)*sd+mu;cq=cal[source][cm].astype(float)*sd+mu
            if key=='memberwise_cqr':
                member=json.loads((ROOT/'results/tables/corrected_full_671/memberwise_comparison.json').read_text())
                radius=member['mean_member_radius']
            else:radius=conformal_radius(scores(cq[:,0],cq[:,2],cy))
        else:
            d=np.load(ROOT/f'results/predictions/benchmark_671/{key}_predictions.npz')
            assert np.array_equal(d['test_mask']>0,tm) and np.array_equal(d['calibration_mask']>0,cm)
            assert np.array_equal(d['test_dates'],idx['test_dates']) and np.array_equal(d['basin_ids'],names)
            assert np.max(np.abs(d['target_test'][tm]-y))<1e-6
            q=d['test_quantiles_log'][tm].astype(float);cq=d['calibration_quantiles_log'][cm].astype(float)
            radius=conformal_radius(scores(cq[:,0],cq[:,2],cy))
        raw=interval_metrics(q[:,0],q[:,2],y);lo,hi,n_empty=project_nonempty(q[:,0]-radius,q[:,2]+radius)
        calibrated=interval_metrics(lo,hi,y);a=interval_arrays(lo,hi,y)
        n=np.bincount(b,minlength=671);stat={k:np.bincount(b,weights=a[k],minlength=671) for k in ['covered','width','winkler']};stat['n']=n
        sufficient[key]=stat;ci={}
        for metric,field in [('picp','covered'),('mpiw','width'),('winkler','winkler')]:
            boot=stat[field][draw].sum(1)/n[draw].sum(1)
            sp=np.asarray([stat[field][p].sum()/n[p].sum() for p in spatial])
            ci[metric]={'basin_ci95':np.quantile(boot,[.025,.975]),'huc2_cluster_ci95':np.quantile(sp,[.025,.975])}
        nses=[nse(y[b==j],q[b==j,1]) for j in range(671)]
        time_key='mc_dropout' if key=='mc_dropout_reselect' else ('ensemble' if key=='memberwise_cqr' else key)
        methods.append({'id':key,'display_name':label,'passes':passes,'radius_log':radius,'raw':raw,'cqr':calibrated,
            'empty_cqr_intervals_projected_to_midpoint':n_empty,
            'median_basin_nse':float(np.nanmedian(nses)),'ci':ci,
            'seconds_per_million':timing[time_key]['seconds_per_million'],
            'timing_note':'same architecture proxy from archived MC timing' if key=='mc_dropout_reselect' else 'archived common-hardware benchmark',
            'training_replicates':'five-member ensemble' if key in ['ensemble','memberwise_cqr'] else 'one trained fit; seed 42',
            'calibration_reused_for_checkpoint_selection':key=='mc_dropout'})
        print('Compared',key,calibrated['winkler'],flush=True)
    for key in sufficient:
        if key=='single_qr':continue
        a=sufficient[key];ref=sufficient['single_qr'];difference={}
        for metric,field in [('picp','covered'),('mpiw','width'),('winkler','winkler')]:
            boot=a[field][draw].sum(1)/a['n'][draw].sum(1)-ref[field][draw].sum(1)/ref['n'][draw].sum(1)
            sp=np.asarray([a[field][p].sum()/a['n'][p].sum()-ref[field][p].sum()/ref['n'][p].sum() for p in spatial])
            difference[metric]={'estimate':float(a[field].sum()/a['n'].sum()-ref[field].sum()/ref['n'].sum()),
                               'basin_ci95':np.quantile(boot,[.025,.975]),'huc2_cluster_ci95':np.quantile(sp,[.025,.975])}
        paired[key+'_minus_single_qr']=difference
    dump(BASE/'tables/method_comparison.json',{'methods':methods,'paired_differences':paired,'n_test':int(tm.sum()),
        'n_calibration':int(cm.sum()),'n_huc2_clusters':len(unique),'bootstrap_conditioning':'fixed trained fits and fixed historical periods',
        'seed_variability_note':'Across-basin bootstrap intervals are not variability across independent training seeds.',
        'memberwise_note':'Static average of separately split-calibrated member intervals; not Jensen et al. EnCQR; no inherited general coverage guarantee.'})
    pd.DataFrame([{'method':m['display_name'],'raw_picp':m['raw']['picp'],'cqr_picp':m['cqr']['picp'],
        'cqr_mpiw_log':m['cqr']['mpiw'],'cqr_winkler_log':m['cqr']['winkler'],'median_nse':m['median_basin_nse'],'passes':m['passes'],
        'seconds_per_million':m['seconds_per_million']} for m in methods]).to_csv(BASE/'tables/method_comparison.csv',index=False)
    print('Expanded method comparison complete',flush=True)

if __name__=='__main__':main()
