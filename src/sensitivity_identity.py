"""Fail closed when a completed fit or optimizer state has different inputs."""
import hashlib
import json
import numpy as np
from src.data.dataset import DYNAMIC_VARS, STATIC_COLS
from src.data.windows import fit_normalization
from src.utils import sha256_file


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def training_input_identity(base,spec):
    path=base/'cache/primary_raw.npz';manifest=path.with_suffix('.json')
    if not manifest.is_file():
        raise RuntimeError('Training cache source manifest is missing; rebuild the cache')
    source=json.loads(manifest.read_text(encoding='utf-8'))
    if source['dynamic_columns']!=DYNAMIC_VARS or source['static_columns']!=STATIC_COLS:
        raise RuntimeError('Training cache columns differ from the current input definitions')
    with np.load(path) as raw:
        norm=fit_normalization(raw,spec['train_start'],spec['train_end'])
    inputs={'schema':1,'cache_sha256':sha256_file(path),'cache_manifest_sha256':sha256_file(manifest),
            'dynamic_columns':DYNAMIC_VARS,'static_columns':STATIC_COLS,'spec':spec,'normalization':norm}
    return {**inputs,'sha256':canonical_sha(inputs)}


def require_input_identity(record,current,label):
    if record.get('input_identity')!=current:
        raise RuntimeError(f'{label}: training input identity missing or changed. Preserve existing artifacts; '
                           'use a new run location, or explicitly audit legacy completed fits before binding them.')


def completed_run_valid(root,base,spec,current=None):
    result=base/f'tables/training_{spec["id"]}.json'
    if not result.exists():return False
    record=json.loads(result.read_text(encoding='utf-8'))
    if not record.get('completed') or record.get('spec')!=spec:
        raise RuntimeError(f'Existing result does not match protocol: {spec["id"]}')
    current=current or training_input_identity(base,spec)
    require_input_identity(record,current,spec['id'])
    for key in ['checkpoint','predictions']:
        path=(root/record[key]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path)!=record[key+'_sha256']:
            raise RuntimeError(f'{spec["id"]}: {key} missing or altered; preserve and audit the existing run')
    if record['normalization']!=current['normalization']:
        raise RuntimeError(f'{spec["id"]}: recorded normalization differs from current inputs')
    return True
