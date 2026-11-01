"""Shared evaluation definitions for the paper's analyses.

Targets and bounds passed to these functions must already share one scale.
Observed-flow groups are retrospective diagnostics; reference-prediction groups
can be assigned prospectively when their thresholds were fitted before testing.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.special import ndtr

PRIMARY_CUTS=(0.33,0.67)

def conformal_radius(scores,alpha=0.1):
    scores=np.asarray(scores,dtype=np.float64)
    scores=scores[np.isfinite(scores)]
    if not len(scores):raise ValueError('An empty calibration pool needs an explicit parent-pool fallback')
    if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    if k>len(scores):return float('inf')
    return float(np.partition(scores,k-1)[k-1])

def scores(lower,upper,y):
    return np.maximum(np.asarray(lower)-y,np.asarray(y)-upper)

def project_nonempty(lower,upper):
    """Replace an empty CQR interval by its midpoint singleton.

    Negative calibration radii can make a valid set-valued conformal interval
    empty. The singleton is a superset of that empty set, so this projection
    cannot lower coverage, and permits conventional nonnegative-width scoring.
    """
    lo=np.asarray(lower,dtype=float).copy();hi=np.asarray(upper,dtype=float).copy()
    crossed=lo>hi
    mid=(lo[crossed]+hi[crossed])/2
    lo[crossed]=mid;hi[crossed]=mid
    return lo,hi,int(crossed.sum())

def interval_arrays(lower,upper,y,alpha=0.1):
    lower,upper,y=np.broadcast_arrays(np.asarray(lower,dtype=float),np.asarray(upper,dtype=float),np.asarray(y,dtype=float))
    if np.any(lower>upper):raise ValueError('Crossed bounds cannot be scored as ordered intervals')
    width=upper-lower
    return {'covered':((y>=lower)&(y<=upper)).astype(float),'width':width,
            'winkler':width+2/alpha*(np.maximum(lower-y,0)+np.maximum(y-upper,0)),
            'below':(y<lower).astype(float),'above':(y>upper).astype(float)}

def interval_metrics(lower,upper,y,alpha=0.1):
    a=interval_arrays(lower,upper,y,alpha)
    return dict(picp=float(a['covered'].mean()),mpiw=float(a['width'].mean()),
                winkler=float(a['winkler'].mean()),p_below=float(a['below'].mean()),
                p_above=float(a['above'].mean()),n=len(np.asarray(y)))

def fit_basin_thresholds(values,basins,cuts=PRIMARY_CUTS):
    values=np.asarray(values);basins=np.asarray(basins)
    return {str(b):np.quantile(values[basins==b],cuts) for b in np.unique(basins)}

def assign_basin_groups(values,basins,thresholds):
    """Right-closed groups: low <= cut1; middle (cut1,cut2]; high > cut2.

    Ties stay in the lower group. A zero upper threshold consequently does not
    label zero-flow observations as high flow. Empty diagnostic groups remain empty.
    """
    values=np.asarray(values);basins=np.asarray(basins);out=np.zeros(len(values),dtype=np.int16)
    for b in np.unique(basins):
        sel=basins==b
        out[sel]=np.searchsorted(thresholds[str(b)],values[sel],side='left')
    return out

def observed_groups(values,basins,cuts=PRIMARY_CUTS):
    return assign_basin_groups(values,basins,fit_basin_thresholds(values,basins,cuts))

def predicted_group_intervals(cal_q,cal_y,cal_basin,test_q,test_basin,alpha=0.1,cuts=PRIMARY_CUTS):
    thresholds=fit_basin_thresholds(cal_q[:,1],cal_basin,cuts)
    cg=assign_basin_groups(cal_q[:,1],cal_basin,thresholds)
    tg=assign_basin_groups(test_q[:,1],test_basin,thresholds)
    cal_scores=scores(cal_q[:,0],cal_q[:,2],cal_y)
    global_r=conformal_radius(cal_scores,alpha)
    radii=[];fallback=[]
    for g in range(len(cuts)+1):
        sel=cg==g
        radii.append(conformal_radius(cal_scores[sel],alpha) if sel.any() else global_r)
        fallback.append(not sel.any())
    rt=np.asarray(radii)[tg]
    lo,hi,n_empty=project_nonempty(test_q[:,0]-rt,test_q[:,2]+rt)
    return lo,hi,{'thresholds':thresholds,'radii':radii,'empty_intervals_projected':n_empty,
        'fallback_to_global':fallback,'cal_groups':cg,'test_groups':tg}

def pinball(y,q,tau):
    error=np.asarray(y)-np.asarray(q)
    return np.maximum(tau*error,(tau-1)*error)

def integrated_quantile_score(y,q,taus):
    """Twice trapezoidal pinball integral on the stated finite quantile grid."""
    y=np.asarray(y,dtype=float);q=np.asarray(q,dtype=float);taus=np.asarray(taus,dtype=float)
    if q.shape!=(len(y),len(taus)):raise ValueError('Quantiles must have shape (n_samples,n_taus)')
    if len(taus)<2 or np.any(np.diff(taus)<=0) or taus[0]<=0 or taus[-1]>=1:
        raise ValueError('A strictly increasing grid inside (0,1) is required')
    weights=np.empty_like(taus);weights[0]=taus[1]-taus[0];weights[-1]=taus[-1]-taus[-2]
    weights[1:-1]=taus[2:]-taus[:-2]
    result=np.zeros(len(y),dtype=float)
    for j,t in enumerate(taus):result+=weights[j]*pinball(y,q[:,j],t)
    return result

def gaussian_crps(y,mu,sigma):
    y,mu,sigma=np.broadcast_arrays(np.asarray(y,dtype=float),np.asarray(mu,dtype=float),np.asarray(sigma,dtype=float))
    if np.any(sigma<=0):raise ValueError('Gaussian scale must be positive')
    z=(y-mu)/sigma
    return sigma*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))


def nse(y, q):
    """Physical-scale Nash-Sutcliffe efficiency from log1p-scale inputs."""
    y = np.maximum(np.expm1(y), 0); q = np.maximum(np.expm1(q), 0)
    den = np.square(y - y.mean()).sum()
    return float(1 - np.square(y - q).sum() / den) if den > 0 else np.nan


def bootstrap_basin(values, basins, regions, stat="mean", seed=20261030, n_boot=5000):
    """Equal-basin bootstrap and HUC2 cluster bootstrap for a basin-level statistic."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    aggregate = np.nanmedian if stat == "median" else np.nanmean
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    dist = aggregate(values[draws], axis=1)
    unique = np.unique(regions)
    spatial = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(regions == r) for r in sampled])
        spatial.append(aggregate(values[ids]))
    return {"estimate": float(aggregate(values)), "basin_ci95": np.quantile(dist, [.025, .975]),
            "huc2_cluster_ci95": np.quantile(spatial, [.025, .975]), "n_basins": len(basins),
            "n_huc2_clusters": len(unique), "fraction_positive": float((values > 0).mean())}


def clean(value):
    """Convert numpy scalars to JSON-native types; nonfinite floats become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def dump(path, value):
    """Write a JSON file with numpy-safe serialization."""
    from pathlib import Path
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
