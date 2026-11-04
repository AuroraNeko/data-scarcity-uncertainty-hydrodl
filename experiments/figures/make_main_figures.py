"""Generate manuscript numbers, table rows and scientific figures from frozen tables."""
from pathlib import Path
import argparse,json,re,shutil,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.evaluation.protocol import dump
from src.corrections.common import load
from src.evaluation.protocol import conformal_radius,scores,predicted_group_intervals
BASE=ROOT/'results';DRAFT=BASE/'drafts';GEN=DRAFT/'generated';FIG=DRAFT/'figures'
COLORS=['#345b78','#6b93a5','#cf9a52','#963e39'];METHOD_COLORS=['#868686','#345b78','#b66d35']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def num(v,d=3):return f'{v:.{d}f}'
def ci(v,d=3,mult=1):return '['+', '.join(num(x*mult,d) for x in v)+']'
def pm(values,d=3):return num(np.mean(values),d)+r' $\pm$ '+num(np.std(values,ddof=1),d)
def rows(name,values):
    (GEN/name).write_text('\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in values)+'\n',encoding='utf-8')
def save(fig,name):
    fig.savefig(BASE/f'figures/{name}.pdf',bbox_inches='tight');fig.savefig(BASE/f'figures/{name}.png',dpi=220,bbox_inches='tight')
    shutil.copy2(BASE/f'figures/{name}.pdf',FIG/f'{name}.pdf');plt.close(fig)
def select_mean(frame,**selection):
    f=frame
    for key,value in selection.items():f=f[f[key]==value]
    return f

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    assert not(args.preview and args.require_complete)
    training_path=BASE/'tables/revision_training_summary.json'
    if not args.preview:
        if not training_path.exists():raise RuntimeError('Completed training analysis is required for publication outputs')
        declared=json.loads(training_path.read_text(encoding='utf-8'))
        if declared.get('missing') or len(declared.get('available',[]))!=48:
            raise RuntimeError('All 48 sensitivity fits are required; --preview is only for author layout work')
    GEN.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);(BASE/'figures').mkdir(parents=True,exist_ok=True)
    from src.visualization.supplement_tables import supplementary_tables
    supplementary_tables(ROOT,rows)
    cf=pd.read_csv(BASE/'tables/coverage_cells.csv');bf=pd.read_csv(BASE/'tables/basin_metrics.csv',dtype={'basin':str,'huc2':str})
    tail=pd.read_csv(BASE/'tables/tail_thresholds.csv');win=pd.read_csv(BASE/'tables/calibration_windows.csv')
    cov=json.loads((BASE/'tables/coverage_summary.json').read_text());comp=json.loads((BASE/'tables/method_comparison.json').read_text())
    from src.visualization.result_macros import control_macros
    macros,bindings=control_macros(BASE/'tables')
    def pooled(year,method):return select_mean(cf,year=year,method=method,dimension='pooled')
    def group(year,method,dimension,label):return select_mean(cf,year=year,method=method,dimension=dimension,group=label)
    def nse_values(year,method='raw'):
        return select_mean(bf,year=year,method=method).groupby('seed').nse.median().to_numpy()
    primary=[]
    for year in [1,3,5,15]:
        raw=pooled(year,'raw');cal=pooled(year,'global')
        meta=json.loads((ROOT/f'results/tables/corrected_scarcity/scarcity_{year}yr_basinseed42_modelseed42.json').read_text())
        primary.append([year,f'{meta["train_observed_targets"]:,}',pm(nse_values(year)),pm(raw.picp),pm(cal.picp),num(raw.mpiw.mean()),num(raw.winkler.mean())])
    rows('scarcity_rows.tex',primary)
    for prefix,year in [('OneYear',1),('FifteenYear',15)]:
        raw=pooled(year,'raw');cal=pooled(year,'global')
        for suffix,value in [('MedianNSE',np.mean(nse_values(year))),('RawPICP',raw.picp.mean()),('CQRPICP',cal.picp.mean()),
                             ('RawWidth',raw.mpiw.mean()),('RawWinkler',raw.winkler.mean())]:macros[prefix+suffix]=num(value)
    for metric,name in [('mpiw','OneYearRawWidthIncreasePct'),('winkler','OneYearRawWinklerIncreasePct')]:
        macros[name]=num(100*(pooled(1,'raw')[metric].mean()/pooled(15,'raw')[metric].mean()-1),0)+r'\%'
    ep=cov['endpoint_effects'];macros['ScarcityMedianNSEDelta']=num(ep['raw_nse']['estimate']);macros['ScarcityMedianNSEDeltaCI']=ci(ep['raw_nse']['basin_ci95'])
    macros['RevisionNSESpatialCI']=ci(ep['raw_nse']['huc2_cluster_ci95'])
    for key,prefix in [('raw_picp','RawPICPEndpoint'),('global_picp','CQRPICPEndpoint'),('raw_high_picp','RawHighPICPEndpoint'),('global_high_picp','CQRHighPICPEndpoint')]:
        macros[prefix+'PP']=num(ep[key]['estimate']*100,1);macros[prefix+'PPCI']=ci(ep[key]['basin_ci95'],1,100)
    macros['RevisionRawPICPSpatialCI']=ci(ep['raw_picp']['huc2_cluster_ci95'],1,100)
    macros['ScarcityFractionBasinsImproved']=num(100*ep['raw_nse']['fraction_positive'],0)+r'\%'
    macros['RawPICPPositiveBasins']=num(100*ep['raw_picp']['fraction_positive'],0)+r'\%'
    for name,method,label in [('OneYearRawHighPICP','raw','high'),('OneYearCQRHighPICP','global','high'),
        ('OneYearCQRLowPICP','global','low'),('OneYearPredCQRHighPICP','predicted','high'),('OneYearOracleCQRHighPICP','observed_reference','high')]:
        macros[name]=num(group(1,method,'observed_flow',label).picp.mean())
    macros['FifteenYearRawHighPICP']=num(group(15,'raw','observed_flow','high').picp.mean())
    macros['OneYearPredCQROverallPICP']=num(pooled(1,'predicted').picp.mean())
    gain=cov['predicted_calibration_gains_1yr']['predicted_high']
    macros['RevisionPredHighGlobal']=num(group(1,'global','predicted_flow','high').picp.mean())
    macros['RevisionPredHighStratified']=num(group(1,'predicted','predicted_flow','high').picp.mean())
    macros['RevisionPredHighGainPP']=num(gain['predicted_minus_global']*100,1)
    for suffix,key in [('CI','basin_ci95'),('SpatialCI','huc2_cluster_ci95'),('TemporalCI','synchronous_water_year_ci95')]:macros['RevisionPredHighGain'+suffix]=ci(gain[key],1,100)
    for year,method,name in [(1,'global','OneYearGlobalBasinBandPct'),(1,'predicted','OneYearPredBasinBandPct'),(15,'global','FifteenYearGlobalBasinBandPct')]:
        macros[name]=num(100*cov['basin_tolerance_band'][f'{year}_{method}']['fraction_in_085_095'],0)+r'\%'
    conditional=[]
    for dimension,title in [('observed_flow','Observed'),('predicted_flow','Predicted')]:
        for label in ['low','middle','high']:
            conditional.append([title,label.capitalize(),*[num(group(1,m,dimension,label).picp.mean()) for m in ['raw','global','predicted']]])
    rows('revision_conditional_rows.tex',conditional)
    for m,label in [('global','Global'),('predicted','Predicted')]:
        f=select_mean(tail,year=1,method=m,requested_upper_fraction=.02)
        macros['RevisionTail'+label+'Two']=num(f.picp.mean())
        macros['OneYear'+('Global' if m=='global' else 'Pred')+'TopTwoPICP']=num(f.picp.mean())
        macros['OneYear'+('Global' if m=='global' else 'Pred')+'TopThirdPICP']=num(select_mean(tail,year=1,method=m,requested_upper_fraction=.33).picp.mean())
    # Strict tail gain with seed-wise pooled ratios and shared basin resampling.
    hits=[];counts=[]
    for seed in [42,123,456]:
        c,t=load(1,seed);q=t['q'];r=conformal_radius(scores(c['q'][:,0],c['q'][:,2],c['y']))
        lo,hi,_=predicted_group_intervals(c['q'],c['y'],c['b'],q,t['b'])
        h=[];n=[]
        for b in np.sort(np.unique(t['b'])):
            sel=t['b']==b;sel=sel&(t['y']>np.quantile(t['y'][sel],.98))
            pg=((t['y']>=lo)&(t['y']<=hi))[sel].sum();gg=((t['y']>=q[:,0]-r)&(t['y']<=q[:,2]+r))[sel].sum()
            h.append(pg-gg);n.append(sel.sum())
        hits.append(h);counts.append(n)
    hits=np.asarray(hits);counts=np.asarray(counts);rng=np.random.default_rng(20261030);draw=rng.integers(0,50,(5000,50))
    dist=(hits[:,draw].sum(-1)/counts[:,draw].sum(-1)).mean(0)
    macros['OneYearPredTopTwoGainPP']=num(100*(hits.sum(1)/counts.sum(1)).mean(),1);macros['OneYearPredTopTwoGainPPCI']=ci(np.quantile(dist,[.025,.975]),1,100)
    methods={m['id']:m for m in comp['methods']};method_rows=[]
    labels={'mc_dropout':'MC, archived spread','single_qr':'Single QR','ensemble':'Mean-quantile ensemble',
        'mc_dropout_noise':'MC + Gaussian noise','mc_dropout_reselect':'MC, tail-selected','mdn':'Gaussian MDN ($K=5$)',
        'umal':'Gaussian quantile head','memberwise_cqr':'Memberwise split-CQR average'}
    for m in comp['methods']:
        method_rows.append([labels[m['id']],num(m['median_basin_nse']),num(m['raw']['picp']),
            num(m['cqr']['picp'])+' '+ci(m['ci']['picp']['basin_ci95']),num(m['cqr']['mpiw']),num(m['cqr']['winkler']),m['passes']])
    rows('revision_method_rows.tex',method_rows)
    for key,prefix in [('single_qr','FullSingle'),('ensemble','FullEnsemble'),('mc_dropout','FullMC')]:
        m=methods[key]
        for suffix,value in [('RawPICP',m['raw']['picp']),('CQRPICP',m['cqr']['picp']),('CQRWinkler',m['cqr']['winkler'])]:macros[prefix+suffix]=num(value)
        macros[prefix+'RelativeTime']=num(m['seconds_per_million']/methods['single_qr']['seconds_per_million'],1)
    macros['FullMCCQRPICPCI']=ci(methods['mc_dropout']['ci']['picp']['basin_ci95'])
    for key,name in [('mc_dropout_noise','RevisionNoiseWinkler'),('mc_dropout_reselect','RevisionTailSelectedWinkler'),
                     ('mdn','RevisionMDNWinkler'),('umal','RevisionGaussianWinkler'),('memberwise_cqr','RevisionMemberwiseWinkler')]:macros[name]=num(methods[key]['cqr']['winkler'])
    macros['RevisionMemberwisePICP']=num(methods['memberwise_cqr']['cqr']['picp'])
    s17=[]
    for key in ['mc_dropout','mc_dropout_noise','mc_dropout_reselect','single_qr','mdn','umal','ensemble']:
        m=methods[key];s17.append([labels[key],m['passes'],*[num(m['raw'][c]) for c in ['picp','mpiw','winkler']],
                                 *[num(m['cqr'][c]) for c in ['picp','mpiw','winkler']],num(m['median_basin_nse'])])
    rows('methods_rows.tex',s17)
    rows('encqr_rows.tex',[[labels[k],num(methods[k]['cqr']['picp']),num(methods[k]['cqr']['mpiw']),num(methods[k]['cqr']['winkler'])] for k in ['single_qr','ensemble','memberwise_cqr']])
    # Calibration windows: all training lengths and all stated window budgets.
    rows('revision_calibration_window_rows.tex',[[year,years,f'{int(select_mean(win,year=year,calibration_years=years).n_cal.iloc[0]):,}',pm(select_mean(win,year=year,calibration_years=years).picp),
        num(select_mean(win,year=year,calibration_years=years).mpiw.mean()),num(select_mean(win,year=year,calibration_years=years).winkler.mean())]
        for year in [1,3,5,15] for years in [1,2,3,4,5]])
    # Physical-scale and retrospective bias tables share the main grouping rule.
    pf=pd.read_csv(BASE/'tables/physical_interval_metrics.csv');bd=pd.read_csv(BASE/'tables/bias_diagnostics.csv',dtype={'basin':str})
    phys=[];bias_rows=[]
    for year in [1,3,5,15]:
        phys.append([year,*[pm(select_mean(pf,year=year,method=m,group=g)[c]) for m,g,c in
             [('raw','all','mpiw'),('global','all','mpiw'),('raw','all','winkler'),('global','all','winkler'),('raw','observed_high','winkler'),('global','observed_high','winkler')]]])
        row=[year]
        for g in [0,1,2]:row.append(num(select_mean(bd,year=year,dimension='observed_flow',group=g).groupby('seed').mean_median_forecast_bias_log.median().mean()))
        f=select_mean(bd,year=year,dimension='observed_flow',group=2)
        row.extend([num(100*f.groupby('seed').relative_mean_bias_physical.median().mean(),1),num(100*f.groupby('seed').mean_median_forecast_bias_log.apply(lambda v:(v<0).mean()).mean(),0)])
        bias_rows.append(row)
    rows('physical_scale_rows.tex',phys);rows('bias_rows.tex',bias_rows)
    per=[]
    for year in [1,3,5,15]:
        row=[year]
        for method in ['global','per_basin']:
            f=pooled(year,method);band=cov['basin_tolerance_band'][f'{year}_{method}']['fraction_in_085_095']
            row.extend([pm(f.picp),num(band,2),num(f.winkler.mean())])
        per.append(row)
    rows('per_basin_rows.tex',per)
    dec=[]
    for label,dim,col,gr in [('Pooled PICP','pooled','picp',None),('Observed high PICP','observed_flow','picp','high'),
                             ('Predicted high PICP','predicted_flow','picp','high'),('Top 2\\% PICP','tail','picp',None),('Winkler','pooled','winkler',None)]:
        row=[label]
        for year in [1,15]:
            for method in ['global','predicted','decile']:
                f=select_mean(tail,year=year,method=method,requested_upper_fraction=.02) if dim=='tail' else (pooled(year,method) if dim=='pooled' else group(year,method,dim,gr))
                row.append(pm(f[col]) if col=='picp' else num(f[col].mean()))
        dec.append(row)
    rows('decile_rows.tex',dec)
    sb=pd.read_csv(BASE/'tables/shifted_bias_diagnostics.csv',dtype={'basin':str});shiftrows=[]
    for label,dim,col,gr in [('Pooled PICP','pooled','picp',None),('Observed high PICP','observed_flow','picp','high'),
                            ('Winkler','pooled','winkler',None),('Upper-group median bias','bias','bias',None)]:
        row=[label]
        for year in [1,15]:
            for method in ['global','bias_basin','bias_basin_predicted']:
                if dim=='bias':value=select_mean(sb,year=year,method=method).groupby('seed').observed_high_mean_bias_log.median().mean();row.append(num(value))
                else:
                    f=pooled(year,method) if dim=='pooled' else group(year,method,dim,gr)
                    row.append(pm(f[col]) if col=='picp' else num(f[col].mean()))
        shiftrows.append(row)
    rows('biasfix_rows.tex',shiftrows)
    quantile=[]
    for year in [1,3,5,15]:
        run=[r for r in cov['runs'] if r['year']==year]
        raw=pooled(year,'raw');cal=pooled(year,'global')
        quantile.append([year,*[pm([r['quantile_coverage'][j] for r in run]) for j in range(3)],
            num(raw.p_below.mean())+' / '+num(raw.p_above.mean()),num(cal.p_below.mean())+' / '+num(cal.p_above.mean())])
    rows('quantile_rows.tex',quantile)
    variant_path=BASE/'tables/sequential_calibration_variants.csv'
    if variant_path.exists():
        seq=pd.read_csv(variant_path);asym=pd.read_csv(BASE/'tables/asymmetric_calibration.csv')
        for metric,output in [('picp','calibration_variant_picp_rows.tex'),('winkler','calibration_variant_winkler_rows.tex')]:
            variants=[]
            for label,frame,scheme in [('Global CQR',seq,'global'),('Rolling 90 days',seq,'roll90'),('Rolling 180 days',seq,'roll180'),
                ('Rolling 365 days',seq,'roll365'),('Expanding',seq,'expand'),('ACI ($\\gamma=0.01$)',seq,'aci_g0.01'),('Asymmetric per side',asym,'asym')]:
                variants.append([label,*[pm(select_mean(frame,year=year,scheme=scheme)[metric]) for year in [1,3,5,15]]])
            variants.append(['Aridity-grouped',*[pm(pooled(year,'aridity')[metric]) for year in [1,3,5,15]]])
            rows(output,variants)
    # Raw finite-grid distribution scores and analytical Gaussian CRPS.
    sf=pd.read_csv(BASE/'tables/distribution_scores.csv');score_rows=[]
    for key in ['single_qr','ensemble','mc_dropout','mdn','umal','mc_dropout_noise']:
        def score_value(kind):
            v=select_mean(sf,method=key,score=kind)
            return num(v.pooled.iloc[0],5) if len(v) else '--'
        score_rows.append([labels[key],score_value('common_3tau_iqs'),score_value('dense_13tau_iqs'),score_value('analytic_gaussian_crps')])
    rows('revision_distribution_rows.tex',score_rows)
    dense=pd.read_csv(BASE/'tables/dense_level_metrics.csv');lev=[]
    for level in [.5,.8,.9,.95,.99]:
        row=[int(level*100)]
        for method in ['mdn','umal','mc_dropout_noise']:
            for mode in ['raw','cqr']:row.append(num(select_mean(dense,method=method,nominal=level,mode=mode).picp.iloc[0]))
        lev.append(row)
    rows('level_rows.tex',lev)
    # Coverage uncertainty table.
    uncertainty=[]
    for key,label,mult,d in [('raw_nse','NSE: 15 minus 1 yr',1,3),('raw_picp','Raw PICP difference (pp)',100,1),
                            ('global_picp','CQR PICP difference (pp)',100,1),('raw_winkler','Raw Winkler difference',1,3),('global_winkler','CQR Winkler difference',1,3)]:
        v=ep[key];uncertainty.append([label,num(v['estimate']*mult,d),ci(v['basin_ci95'],d,mult),ci(v['huc2_cluster_ci95'],d,mult),'--'])
    uncertainty.append(['Predicted-high calibration gain (pp)',num(gain['predicted_minus_global']*100,1),ci(gain['basin_ci95'],1,100),ci(gain['huc2_cluster_ci95'],1,100),ci(gain['synchronous_water_year_ci95'],1,100)])
    rows('revision_uncertainty_rows.tex',uncertainty)
    difference_rows=[]
    for key,values in comp['paired_differences'].items():
        method=key.removesuffix('_minus_single_qr')
        for metric,metric_label,mult,digits in [('picp','PICP (pp)',100,1),('winkler','Winkler',1,3)]:
            v=values[metric]
            difference_rows.append([labels[method],metric_label,num(v['estimate']*mult,digits),ci(v['basin_ci95'],digits,mult),ci(v['huc2_cluster_ci95'],digits,mult)])
    rows('revision_method_difference_rows.tex',difference_rows)
    rows('revision_association_rows.tex',[[k.replace('_',r'\_'),num(v['spearman_rho']),ci(v['basin_ci95']),ci(v['huc2_cluster_ci95'])] for k,v in cov['basin_associations'].items()])
    hc=json.loads((BASE/'tables/hydrology_context_summary.json').read_text())['groups'];miss=[]
    for period,label in [('train_1yr','Training: 1 yr'),('train_3yr','Training: 3 yr'),('train_5yr','Training: 5 yr'),('train_15yr','Training: 15 yr'),
                         ('independent_validation','Independent validation'),('calibration_4yr','Calibration: 4 yr'),('calibration_5yr','Calibration: 5 yr'),('test','Test')]:
        row=[label]
        for setname in ['primary_50','all_671']:
            v=hc[setname]['missingness'][period];row.extend([f'{v["n_observed"]:,}',f'{v["n_missing"]:,}'])
        miss.append(row)
    rows('revision_missing_rows.tex',miss)
    hyd=[]
    for period,label in [('calibration_5yr','Calibration 1995--2000'),('test','Test 2000--2008')]:
        v=hc['primary_50']['median_basin_hydrology'][period]
        hyd.append([label,*[num(v[c]) for c in ['mean_flow','q95','q99','below_calibration_q10_fraction','above_calibration_q95_fraction','mean_daily_precip']]])
    rows('revision_hydrology_rows.tex',hyd)
    unseen=json.loads((BASE/'tables/unseen_calibration_summary.json').read_text())['rows']
    rows('revision_unseen_window_rows.tex',[[r['training_years'],r['local_calibration_years'],f'{r["n_local_calibration"]:,}',
        num(r['picp_mean'])+r' $\pm$ '+num(r['picp_sd']),num(r['mpiw_mean']),num(r['winkler_mean'])] for r in unseen])
    # Core figures, generated from exactly the same frames as the manuscript tables.
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained');xx=np.arange(4)
    for ax in axes.flat:ax.set_xticks(xx,[1,3,5,15]);ax.set_xlabel('Supervised training years')
    nv=[nse_values(y) for y in [1,3,5,15]];axes[0,0].errorbar(xx,[np.mean(v) for v in nv],yerr=[np.std(v,ddof=1) for v in nv],marker='o',color=COLORS[0],capsize=3)
    axes[0,0].set_ylabel('Median basin NSE')
    for ax,metric in [(axes[0,1],'picp'),(axes[1,0],'mpiw'),(axes[1,1],'winkler')]:
        for method,color,label in [('raw',METHOD_COLORS[0],'Raw QR'),('global',METHOD_COLORS[1],'Global CQR')]:
            vv=[pooled(y,method)[metric].to_numpy() for y in [1,3,5,15]]
            ax.errorbar(xx,[np.mean(v) for v in vv],yerr=[np.std(v,ddof=1) for v in vv],marker='o',label=label,color=color,capsize=3)
        ax.set_ylabel({'picp':'Pooled coverage','mpiw':'MPIW (log-flow)','winkler':'Winkler (log-flow)'}[metric])
    axes[0,1].axhline(.9,color='black',ls='--',lw=.8);axes[1,0].legend(frameon=False)
    for ax,letter in zip(axes.flat,'abcd'):ax.text(-.13,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    save(fig,'revision_scarcity')
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for row,(dimension,title) in enumerate([('observed_flow','Observed outcomes'),('predicted_flow','Predicted flow')]):
        for g,color in zip(['low','middle','high'],[COLORS[1],COLORS[2],COLORS[3]]):
            vv=[group(y,'global',dimension,g).picp.to_numpy() for y in [1,3,5,15]]
            axes[row,0].errorbar(xx,[v.mean() for v in vv],yerr=[v.std(ddof=1) for v in vv],marker='o',color=color,label=g.capitalize(),capsize=2)
        axes[row,0].set_xticks(xx,[1,3,5,15]);axes[row,0].set_xlabel('Supervised training years');axes[row,0].set_title(title+' groups, global CQR',fontsize=10)
        for offset,(method,color,label) in enumerate(zip(['raw','global','predicted'],METHOD_COLORS,['Raw QR','Global CQR','Predicted-regime CQR'])):
            vv=[group(1,method,dimension,g).picp.to_numpy() for g in ['low','middle','high']]
            axes[row,1].errorbar(np.arange(3)+(offset-1)*.14,[v.mean() for v in vv],yerr=[v.std(ddof=1) for v in vv],marker='o',color=color,label=label,capsize=2)
        axes[row,1].set_xticks(range(3),['Low','Middle','High']);axes[row,1].set_title(title+' groups, 1-year model',fontsize=10)
        for ax in axes[row]:ax.axhline(.9,color='black',ls='--',lw=.8);ax.set_ylim(.65,1.);ax.set_ylabel('Coverage')
    axes[0,0].legend(frameon=False,fontsize=8);axes[1,1].legend(frameon=False,fontsize=7);save(fig,'revision_conditional')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained');fractions=[.33,.2,.1,.05,.02]
    for year,color in zip([1,3,5,15],COLORS):
        vv=[select_mean(tail,year=year,method='global',requested_upper_fraction=f).picp.to_numpy() for f in fractions]
        axes[0].errorbar(np.asarray(fractions)*100,[v.mean() for v in vv],yerr=[v.std(ddof=1) for v in vv],marker='o',color=color,label=f'{year} yr',capsize=2)
    for method,color,label in zip(['raw','global','predicted'],METHOD_COLORS,['Raw QR','Global CQR','Predicted-regime CQR']):
        vv=[select_mean(tail,year=1,method=method,requested_upper_fraction=f).picp.to_numpy() for f in fractions]
        axes[1].errorbar(np.asarray(fractions)*100,[v.mean() for v in vv],yerr=[v.std(ddof=1) for v in vv],marker='o',color=color,label=label,capsize=2)
    for ax in axes:ax.set_xscale('log');ax.set_xlim(36,1.8);ax.set_xticks([33,20,10,5,2],[33,20,10,5,2]);ax.set_xlabel('Requested upper share (%)');ax.set_ylabel('Observed-tail coverage');ax.axhline(.9,color='black',ls='--',lw=.8);ax.legend(frameon=False,fontsize=8)
    for ax,letter in zip(axes,'ab'):ax.text(-.16,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    save(fig,'revision_tail')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    for ax,year in zip(axes,[1,15]):
        groups=[select_mean(bf,year=year,method=m).groupby('basin').picp.mean().to_numpy() for m in ['raw','global','predicted']]
        ax.axhspan(.85,.95,color='#dce6eb',alpha=.8);ax.axhline(.9,color='#777777',ls='--')
        ax.boxplot(groups,positions=[0,1,2],widths=.38,showfliers=False,medianprops={'color':'black'})
        rng=np.random.default_rng(20261030)
        for j,(values,color) in enumerate(zip(groups,METHOD_COLORS)):
            ax.scatter(j+rng.uniform(-.15,.15,len(values)),values,color=color,s=12,alpha=.65)
        ax.set_xticks([0,1,2],['Raw QR','Global CQR','Predicted CQR']);ax.set_ylim(.3,1.);ax.set_ylabel('Seed-averaged basin coverage');ax.set_title(f'{year}-year record')
        bands=[f'{100*np.mean((v>=.85)&(v<=.95)):.0f}%' for v in groups]
        ax.text(.5,-.24,'Within 0.85--0.95: '+', '.join(bands),transform=ax.transAxes,ha='center',fontsize=9)
    save(fig,'revision_basin_calibration')
    fig,axes=plt.subplots(1,3,figsize=(9.5,5.2),layout='constrained',gridspec_kw={'width_ratios':[1,1,1]});yy=np.arange(len(comp['methods']))
    short_labels={'mc_dropout':'Archived MC','single_qr':'Single QR','ensemble':'Ensemble','mc_dropout_noise':'MC + noise',
                  'mc_dropout_reselect':'Tail-selected MC','mdn':'MDN','umal':'Gaussian head','memberwise_cqr':'Memberwise CQR'}
    names_plot=[short_labels[m['id']] for m in comp['methods']]
    for ax,metric in zip(axes[:2],['picp','winkler']):
        values=np.asarray([m['cqr'][metric] for m in comp['methods']]);bounds=np.asarray([m['ci'][metric]['basin_ci95'] for m in comp['methods']])
        ax.errorbar(values,yy,xerr=np.vstack([values-bounds[:,0],bounds[:,1]-values]),fmt='o',capsize=3,color=COLORS[0]);ax.set_yticks(yy,names_plot if ax==axes[0] else []);ax.invert_yaxis()
    axes[0].axvline(.9,color='#777777',ls='--');axes[0].set_xlabel('Calibrated PICP');axes[1].set_xlabel('Calibrated Winkler (log-flow)')
    axes[2].scatter([m['seconds_per_million'] for m in comp['methods']],yy,color=COLORS[2]);axes[2].set_xscale('log');axes[2].set_yticks(yy,[]);axes[2].invert_yaxis();axes[2].set_xlabel('Seconds / million samples')
    save(fig,'revision_methods')
    sscore=json.loads((BASE/'tables/scoring_summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for method,color in zip(['mdn','umal','mc_dropout_noise'],[COLORS[0],COLORS[2],COLORS[3]]):
        qc=sscore['level_reliability'][method]['quantile_coverage'];tx=np.array([float(t) for t in qc]);axes[0].plot(tx,[qc[t] for t in qc],label=labels[method],color=color)
        for mode,ls in [('raw','--'),('cqr','-')]:
            f=select_mean(dense,method=method,mode=mode).sort_values('nominal');axes[1].plot(f.nominal,f.picp,ls,marker='o',color=color)
    for ax in axes:ax.plot([0,1],[0,1],color='#888888',lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1)
    axes[0].set_xlabel('Quantile level');axes[0].set_ylabel('Fraction below quantile');axes[0].legend(frameon=False,fontsize=7)
    axes[1].set_xlabel('Nominal central coverage');axes[1].set_ylabel('Achieved interval coverage');save(fig,'revision_dense_reliability')
    # Publish training-specific outputs only when all declared fits are available.
    training_path=BASE/'tables/revision_training_summary.json'
    if training_path.exists():
        tr=json.loads(training_path.read_text())
        if not tr['missing'] or args.preview:
            from src.visualization.training_outputs import training_outputs
            if args.preview:tr['preview']=True
            training_outputs(tr,macros,rows,save)
        elif args.require_complete:raise RuntimeError(f'{len(tr["missing"])} training fits missing')
    elif args.require_complete:raise RuntimeError('Training analysis is missing')
    macros['RevisionPackageID']=re.search(r'^version:\s*"?([^"\s]+)',(ROOT/'CITATION.cff').read_text(),re.M).group(1)
    (GEN/'result_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    dump(BASE/'tables/publication_macro_values.json',macros)
    dump(BASE/'tables/publication_control_macro_bindings.json',bindings)
    print('Publication tables and core figures generated',flush=True)

if __name__=='__main__':main()
