"""Run declared sensitivity fits serially on the one available GPU."""
from pathlib import Path
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.sensitivity import specs, queue_position
from src.utils import sha256_file

BASE=ROOT/'results'
def write_status(status):
    temp=BASE/'training_queue_status.json.tmp'
    temp.write_text(json.dumps(status,indent=2),encoding='utf-8');temp.replace(BASE/'training_queue_status.json')

jobs=specs()
jobs.sort(key=lambda s:(queue_position(s),s['train_start'],s['seed']))
started=time.time();done=[]
for spec in jobs:
    result=BASE/f'tables/training_{spec["id"]}.json'
    if result.exists():
        existing=json.loads(result.read_text())
        if existing.get('completed') and existing['spec']==spec:
            valid=all((ROOT/existing[key]).exists() and sha256_file(ROOT/existing[key])==existing[key+'_sha256']
                      for key in ['checkpoint','predictions'])
            if valid:
                done.append(spec['id']);print('SKIP validated',spec['id'],flush=True);continue
            print('STALE',spec['id'],'stored artifacts missing or altered; retraining',flush=True)
    path=BASE/f'logs/{spec["id"]}.log'
    write_status({'running':spec['id'],'completed_this_invocation':done,'total_declared_jobs':len(jobs),'elapsed_seconds':time.time()-started})
    print('RUN',spec['id'],flush=True)
    with path.open('a',encoding='utf-8') as log:
        rc=subprocess.call([sys.executable,'-u',str(ROOT/'experiments/train_sensitivity/train_variants.py'),'--id',spec['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if rc:
        write_status({'running':spec['id'],'failed':spec['id'],'returncode':rc,'completed_this_invocation':done,'elapsed_seconds':time.time()-started})
        print('FAILED',spec['id'],'see',path,flush=True);sys.exit(rc)
    done.append(spec['id'])
    print('COMPLETE',spec['id'],flush=True)
write_status({'completed':True,'jobs':done,'elapsed_seconds':time.time()-started})
print('ALL TRAINING COMPLETE',flush=True)
