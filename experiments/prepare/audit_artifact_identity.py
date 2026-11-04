"""Resolve and verify every recorded checkpoint/prediction SHA-256."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.utils import repository_path,sha256_file
from src.evaluation.protocol import dump

def audit(root):
    rows=[];failures=[]
    for path in sorted((root/'results/tables').rglob('*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict):continue
        for key in ['checkpoint','predictions']:
            if not isinstance(data.get(key),str):continue
            expected=data.get(key+'_sha256')
            try:
                artifact=repository_path(root,data[key])
                actual=sha256_file(artifact)
                passed=isinstance(expected,str) and expected==actual
                error=None if passed else 'SHA-256 differs or is missing'
            except (OSError,ValueError) as exc:
                actual=None;passed=False;error=str(exc)
            row={'metadata':path.relative_to(root).as_posix(),'field':key,'artifact':data[key],
                 'expected_sha256':expected,'actual_sha256':actual,'passed':passed,'error':error}
            rows.append(row)
            if not passed:failures.append(row)
    return {'passed':bool(rows) and not failures,'checked_references':len(rows),'failed_references':failures,'references':rows}

def main():
    result=audit(ROOT);dump(ROOT/'results/qa/artifact_identity.json',result)
    print('Artifact references',result['checked_references'],'failed',len(result['failed_references']),flush=True)
    if not result['passed']:raise RuntimeError('Artifact identity audit failed; see results/qa/artifact_identity.json')

if __name__=='__main__':main()
