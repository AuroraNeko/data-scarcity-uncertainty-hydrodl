"""Materialize the primary basins once; no fitting and no CUDA initialization."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dataset import DYNAMIC_VARS, STATIC_COLS

def main():
    out=ROOT/'results/cache/primary_raw.npz'
    if out.exists():
        print('Existing raw cache:',out,flush=True)
        return
    meta=json.loads((ROOT/'results/tables/corrected_scarcity/scarcity_1yr_basinseed42_modelseed42.json').read_text())
    basins=meta['selected_basins']
    dynamic=[];target=[];mask=[];static=[];hashes={};dates=None
    for b in basins:
        p=ROOT/f'data/processed/camels_us/{b}.csv'
        df=pd.read_csv(p,usecols=['date',*DYNAMIC_VARS,'target','flow_mask',*STATIC_COLS])
        dt=df['date'].to_numpy(dtype='datetime64[D]')
        if dates is None:dates=dt
        assert np.array_equal(dates,dt)
        dynamic.append(df[DYNAMIC_VARS].to_numpy(dtype=np.float64))
        target.append(df['target'].to_numpy(dtype=np.float64))
        mask.append(df['flow_mask'].to_numpy(dtype=np.uint8))
        static.append(df[STATIC_COLS].iloc[0].to_numpy(dtype=np.float64))
        hashes[b]=hashlib.sha256(p.read_bytes()).hexdigest()
    np.savez(out,dynamic=np.stack(dynamic),target=np.stack(target),mask=np.stack(mask),
             static=np.stack(static),dates=dates,basins=np.asarray(basins))
    (out.with_suffix('.json')).write_text(json.dumps({'dynamic_columns':DYNAMIC_VARS,
        'static_columns':STATIC_COLS,'source_sha256':hashes,'target_scale':'log1p(mm/day)'},indent=2),encoding='utf-8')
    print('Prepared',out,round(out.stat().st_size/1e6,1),'MB',flush=True)

if __name__=='__main__':main()
