"""Run declared sensitivity fits serially on the one available GPU."""
from pathlib import Path
import json
import subprocess
import sys
import time
from train_revision import specs,hash_file

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results'
priority={'validation_':0,'temporal_':1,'multilevel_':2,'tcn_':3,'memory':4}
jobs=specs()
jobs.sort(key=lambda s:(next(v for k,v in priority.items() if s['id'].startswith(k)),s['train_start'],s['seed']))
started=time.time();done=[]
for spec in jobs:
    result=BASE/f'tables/training_{spec["id"]}.json'
    if result.exists():
        existing=json.loads(result.read_text())
        if existing.get('completed') and existing['spec']==spec:
            for key in ['checkpoint','predictions']:
                p=ROOT/existing[key]
                assert p.exists() and hash_file(p)==existing[key+'_sha256']
            done.append(spec['id']);print('SKIP validated',spec['id'],flush=True);continue
    path=BASE/f'logs/{spec["id"]}.log'
    status={'running':spec['id'],'completed_this_invocation':done,'total_declared_jobs':len(jobs),'elapsed_seconds':time.time()-started}
    (BASE/'training_queue_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('RUN',spec['id'],flush=True)
    with path.open('a',encoding='utf-8') as log:
        rc=subprocess.call([sys.executable,'-u',str(BASE/'code/train_revision.py'),'--id',spec['id']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if rc:
        status.update(failed=spec['id'],returncode=rc)
        (BASE/'training_queue_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        print('FAILED',spec['id'],'see',path,flush=True);sys.exit(rc)
    done.append(spec['id'])
    print('COMPLETE',spec['id'],flush=True)
(BASE/'training_queue_status.json').write_text(json.dumps({'completed':True,'jobs':done,'elapsed_seconds':time.time()-started},indent=2),encoding='utf-8')
print('ALL TRAINING COMPLETE',flush=True)
