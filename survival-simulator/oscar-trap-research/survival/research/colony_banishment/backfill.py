"""Audit and reproduce metrics-only colony runs, retaining explicit provenance.

No original result file is edited. Each missing source row gets its own new
simulation and unique replay. Existing durable per-run results make this resumable.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/colony_banishment'
sys.path.insert(0,str(HERE))


def original_rows():
    inventory=json.loads((OUT/'experiment-inventory.json').read_text())
    hashes={hashlib.sha256(p.read_bytes()).hexdigest():version for version,p in
            [(v,HERE/('policy.py' if v==5 else f'policy_v{v}.py')) for v in range(1,6)]}
    rows=[]
    for file in inventory['files']:
        for index,row in enumerate(json.loads((OUT/file['file']).read_text())['runs']):
            reference=f"{file['file']}#{index}"
            version=hashes[row['policy_sha256']]
            replay=row.get('replay')
            exists=bool(replay and (ROOT.parent/replay).is_file())
            digest=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            rows.append(dict(reference=reference,version=version,original=row,original_result_sha256=digest,
                original_replay=replay if exists else None))
    return rows


def comparison(original,reproduction):
    fields=['score','duration','alive','metrics','events','releases','trace']
    if 'guide_episodes' in original:fields.append('guide_episodes')
    equal={k:original[k]==reproduction.get(k) for k in fields}
    return dict(equal_fields=equal,all_compared_fields_match=all(equal.values()),
        score_delta=round(reproduction['score']-original['score'],6),
        duration_delta=round(reproduction['duration']-original['duration'],6))


def existing_reproductions():
    found={}
    for path in sorted((OUT/'run-results').glob('*.json')):
        row=json.loads(path.read_text());source=row.get('reproduction_of')
        if source and row.get('replay') and (ROOT.parent/row['replay']).is_file():
            found[source]=row
    return found


def audit():
    saved=existing_reproductions();rows=[]
    for row in original_rows():
        original=row.pop('original');saved_row=saved.get(row['reference'])
        if row['original_replay']:
            row.update(status='original_recorded',replay=row['original_replay'])
        elif saved_row:
            row.update(status='new_reproduction_recorded',replay=saved_row['replay'],result_file=saved_row['result_file'],
                comparison=comparison(original,saved_row),reproduction_run_id=saved_row['run_id'])
        else:row.update(status='metrics_only',replay=None)
        rows.append(row)
    data=dict(original_reported_runs=len(rows),original_recorded=sum(r['status']=='original_recorded' for r in rows),
        new_reproductions=sum(r['status']=='new_reproduction_recorded' for r in rows),
        still_metrics_only=sum(r['status']=='metrics_only' for r in rows),
        provenance='New reproductions cover old configurations; they do not recover original missing frames. Original metrics are preserved.',rows=rows)
    path=OUT/'replay-coverage-audit.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
    return data


def worker(row):
    module=importlib.import_module('run' if row['version']==5 else f"run_v{row['version']}")
    original=row['original']
    native=(row['reference'].startswith('heldout-v5-fixtures.json#') and original['scenario']=='obstacles'
            and original['seed']==24 and original['mode']=='banish')
    request=dict(label='reproduce-'+row['reference'],reproduction_of=row['reference'],
        original_result_sha256=row['original_result_sha256'],every=20 if native else 10,native_render=native)
    result=module.run_case((original['scenario'],original['seed'],original['mode'],original['horizon'],original['settings'],request))
    return dict(reference=row['reference'],replay=result['replay'],result_file=result['result_file'],
        comparison=comparison(original,result))


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int);p.add_argument('--version',type=int);a=p.parse_args()
    status=audit();print(json.dumps({k:v for k,v in status.items() if k!='rows'}),flush=True)
    if a.audit_only:return
    missing={r['reference'] for r in status['rows'] if r['status']=='metrics_only'}
    jobs=[r for r in original_rows() if r['reference'] in missing and (not a.version or r['version']==a.version)]
    # Deliver final v5 findings first, followed by the historical iterations.
    jobs.sort(key=lambda r:(-r['version'],r['reference']))
    if a.limit:jobs=jobs[:a.limit]
    failures=[];completed=0;start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(worker,row):row['reference'] for row in jobs}
        for future in as_completed(pending):
            try:
                result=future.result();completed+=1
                print(json.dumps(dict(completed=completed,total=len(jobs),elapsed_seconds=round(time.perf_counter()-start,1),**result)),flush=True)
            except Exception as exc:
                failure=dict(reference=pending[future],error=repr(exc));failures.append(failure);print(json.dumps(failure),flush=True)
            audit()
    (OUT/'backfill-last-errors.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit().items() if k!='rows'}),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
