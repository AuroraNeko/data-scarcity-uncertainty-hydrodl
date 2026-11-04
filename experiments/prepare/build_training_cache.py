"""Materialize the primary basins once; no fitting and no CUDA initialization."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dataset import DYNAMIC_VARS, STATIC_COLS, primary_basin_ids

def main():
    out=ROOT/'results/cache/primary_raw.npz'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        metadata=out.with_suffix('.json')
        if not metadata.exists():raise RuntimeError('Raw cache has no source manifest; rebuild it before fitting')
        saved=json.loads(metadata.read_text(encoding='utf-8'))
        expected=primary_basin_ids()
        if set(saved['source_sha256'])!=set(expected):raise RuntimeError('Raw cache basin set changed; rebuild it before fitting')
        for b,digest in saved['source_sha256'].items():
            p=ROOT/f'data/processed/camels_us/{b}.csv'
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Raw cache source changed: {b}; rebuild it before fitting')
        with np.load(out) as raw:
            assert raw['basins'].tolist()==expected,'Raw cache basin order differs'
            assert raw['dynamic'].shape[:2]==raw['target'].shape==raw['mask'].shape
        print('Existing raw cache source hashes verified:',out,flush=True)
        return
    # The primary basin set is derived from metadata alone, so this stage
    # does not depend on any training-stage output.
    basins=primary_basin_ids()
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
