"""Materialize archived 671-basin split mirrors using the original Dataset.

CPU construction avoids coupling pandas preparation to a CUDA training process.
--verify-existing compares sampled cached windows with independently loaded data.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.dataset import CamelsDataset,get_basin_list
from src.benchmark.gpu_loader import GpuSplitData
CACHE=ROOT/'results/cache/benchmark_gpu'
SPLITS={'train':('1980-10-01','1995-09-30',None),
        'calibration':('1994-10-01','2000-09-30','1995-10-01'),
        'test':('1999-10-02','2008-12-31','2000-10-01'),
        'selection':('1991-10-02','1995-09-30','1992-10-01')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-existing',action='store_true');args=parser.parse_args()
    basins=get_basin_list();stats=json.loads((ROOT/'data/metadata/normalization_stats.json').read_text())
    rows=[]
    for split,(start,end,target_start) in SPLITS.items():
        # Derive antecedent dates rather than relying on leap-year arithmetic.
        if target_start is not None:start=str(np.datetime64(target_start)-np.timedelta64(365,'D'))
        ids=basins[:3] if args.verify_existing else basins
        ds=CamelsDataset(ids,start,end,seq_len=365,normalization_stats=stats,
                         target_start_date=target_start,target_end_date=end if target_start else None)
        if not args.verify_existing:
            mirror=GpuSplitData(ds,torch.device('cpu'),365);mirror.save(CACHE/f'{split}.npz')
            print('Prepared',split,mirror.n,flush=True);continue
        d=np.load(CACHE/f'{split}.npz');cached_ids=json.loads((CACHE/f'{split}.basins.json').read_text())
        assert list(ids)==cached_ids[:len(ids)]
        maxima={k:0. for k in ['dynamic','static','target','mask']}
        for k in np.linspace(0,len(ds)-1,36,dtype=int):
            expected=ds[k][:4];b=int(d['win_basin'][k]);a=int(d['win_start'][k]);t=int(d['win_target'][k])
            actual=[d['flat_dyn'][a:a+365],d['static'][b],d['flat_tgt'][t:t+1],d['flat_mask'][t:t+1]]
            for key,v,w in zip(maxima,actual,expected):
                maxima[key]=max(maxima[key],float(np.max(abs(v-w.numpy()))));assert maxima[key]<1e-6,(split,key,maxima)
            bidx,offset=ds.indices[k];assert ds.basin_data[bidx]['dates'][offset+365]==d['target_dates'][k]
        rows.append({'split':split,'sample_windows':36,'max_abs_differences':maxima,'dates_equal':True})
        print('Verified',split,maxima,flush=True)
    if args.verify_existing:
        out=ROOT/'results/qa/full_basin_cache_verification.json';out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({'passed':True,'checks':rows},indent=2),encoding='utf-8')

if __name__=='__main__':main()
