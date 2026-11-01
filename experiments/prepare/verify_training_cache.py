"""Independent comparison against the original dataset/normalization path."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.dataset import CamelsDataset,compute_normalization_stats,DYNAMIC_VARS,STATIC_COLS
from src.data.windows import Windows
from src.evaluation.protocol import dump

def main():
    torch.set_num_threads(2);raw=np.load(ROOT/'results/cache/primary_raw.npz');basins=raw['basins'].tolist();checks=[]
    for start,end,seq in [('1994-10-01','1995-09-30',180),('1980-10-01','1995-09-30',180),('1984-10-01','1985-09-30',180),
                          ('1980-12-31','1995-09-30',180),('1980-12-31','1995-09-30',365),('1994-10-01','1995-09-30',365)]:
        w=Windows(raw,start,end,torch.device('cpu'),seq=seq);stats=compute_normalization_stats(basins,start,end)
        assert abs(stats['target_mean']-w.norm['target_mean'])<1e-12
        assert abs(stats['target_std']-w.norm['target_std'])<1e-12
        for j,c in enumerate(DYNAMIC_VARS):
            assert abs(stats[c+'_mean']-w.norm['dynamic_mean'][j])<1e-9
            assert abs(stats[c+'_std']-w.norm['dynamic_std'][j])<1e-8
        first=np.datetime64(start)-np.timedelta64(seq,'D')
        ds=CamelsDataset(basins[:3],str(first),end,seq_len=seq,normalization_stats=stats,target_start_date=start,target_end_date=end)
        worst={'dynamic':0.,'static':0.,'target':0.,'mask':0.}
        indices=np.linspace(0,len(ds)-1,25,dtype=int)
        for k in indices:
            bidx,offset=ds.indices[k];date=ds.basin_data[bidx]['dates'][offset+seq];day=np.flatnonzero(raw['dates']==date)[0]
            got=w.batch(torch.tensor([bidx*w.nd+day]));expected=ds[k][:4]
            for label,a,e in zip(worst,got,expected):
                diff=float(np.max(np.abs(a[0].numpy()-e.numpy())));worst[label]=max(worst[label],diff)
                assert diff<1e-6,(label,diff,start,end)
        checks.append({'start':start,'end':end,'seq_len':seq,'sample_windows_checked':len(indices),'max_abs_differences':worst,
                       'normalization_agrees_with_original':True})
        print('Independent cache verification passed',start,end,worst,flush=True)
    dump(ROOT/'results/qa/training_cache_verification.json',{'passed':True,'checks':checks})

if __name__=='__main__':main()
