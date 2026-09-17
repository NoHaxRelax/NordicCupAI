"""Inventory old results, reproduce missing replay cases, preserve every receipt.
This never recovers original missing frames. Reproductions run saved parameters
and archived controllers; versions/known reconstruction limits are explicit.
"""
import argparse,gzip,hashlib,json,os,subprocess,sys,traceback,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1];OUT=ROOT/'results/dedicated_sprinter'
AUDIT=OUT/'replay-backfill-audit.json'
EXISTING={
 'replays.json':['orchard-founder.json.gz','orchard-specialist.json.gz'],
 'replays-final-v5.json':['final-founder-orchard.json.gz','final-founder-no-food.json.gz','final-specialist-orchard.json.gz','final-newborn-cutoff.json.gz','final-real-mutant.json.gz','final-heldout-capture.json.gz'],
 'protection-replay-v5.json':['final-protected-workers.json.gz'],
}

def write_audit(data):
    temporary=OUT/('.replay-audit-'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(AUDIT)


def inventory():
    cases=[];excluded=[];original_files={}
    for p in sorted(OUT.glob('*.json')):
        data=json.loads(p.read_text())
        if 'runs' not in data:continue
        original_files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for index,r in enumerate(data['runs']):
            if 'horizon' not in r:
                excluded.append(dict(source=p.name,row=index,reason='Non-time-advancing mutation distribution search: no game ticks, no simulation time. Not represented as gameplay footage.'))
                continue
            module=Path(data.get('archived_controller',{}).get('file','controller.py')).stem
            case=dict(id=f'{p.stem}:{index:03d}',source=p.name,row=index,controller=module,
                kind='protection' if 'worker_captures' in r else 'generated' if 'samples' in r else 'single',
                original_metrics=r,source_sha256=original_files[p.name],status='missing recording')
            if p.name=='food-v1.json':
                case['controller']='controller_v2'
                case['limitation']='Historical food-v1 file is an exact metrics duplicate of food.json and stores the v2 source hash; its later v1 archive annotation was wrong. Reproduced with v2. Earlier v1 reproduction remains a separate completed run.'
            if p.name in EXISTING:
                replay=OUT/EXISTING[p.name][index]
                d=json.load(gzip.open(replay,'rt'))
                assert d['format']=='survival-replay'
                case.update(status='original recording exists',replay=str(replay.relative_to(ROOT)))
            if p.name=='fullgame-caste-1-v3.json':
                case['limitation']='Archived v3 controller plus reconstructed early uncapped assignment wrapper. Original wrapper file was not archived; reproduced behavior is not claimed byte-identical.'
            cases.append(case)
    # This earlier sweep completed cases but failed when serializing its aggregate.
    # Its line-delimited printed outcomes preserve the configurations and version.
    log=OUT/'validation-final.log'
    for index,line in enumerate(log.read_text().splitlines()):
        if not line.startswith('{'):continue
        r=json.loads(line);r['horizon']=140
        cases.append(dict(id=f'validation-final-log:{index:03d}',source=log.name,row=index,
            controller='controller_v4',kind='single',original_metrics=r,
            source_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),status='missing recording',
            limitation='New reproduction from a completed-case log. Original full metrics aggregate failed serialization.'))
    payload=dict(scope='Every saved completed time-advancing sprinter run record, plus the log-only v4 validation sweep.',
        note='Original missing frames cannot be recovered. Similar repeated configurations remain distinct records. Other console logs mirror the saved aggregates and are not double-counted. Unlogged diagnostic reruns cannot be individually reconstructed.',
        original_files=original_files,excluded_non_game_records=excluded,cases=cases)
    write_audit(payload)
    print('Inventoried',len(cases),'game run records;',sum(c['status']=='original recording exists' for c in cases),'already recorded;',len(excluded),'non-game distribution records excluded.')

def completed():
    found={}
    expected={c['id']:c['controller'] for c in json.loads(AUDIT.read_text())['cases']}
    for p in (OUT/'runs').glob('*.json'):
        try:d=json.loads(p.read_text())
        except json.JSONDecodeError:continue # A receipt from an earlier non-atomic writer may still be finishing.
        r=d['result'];repro=r.get('reproduction_of') or {}
        if repro.get('original_id') and repro.get('controller')==expected.get(repro['original_id']):found[repro['original_id']]=(p,r)
    return found

def execute(case):
    os.environ['DEDICATED_CONTROLLER']=case['controller']
    # Entry subprocess has not imported the engine/controller until this point.
    from experiment import run
    from protection import protect
    from fullgame import game
    original=case['original_metrics']
    provenance=dict(original_id=case['id'],source=case['source'],row=case['row'],
        original_metrics_sha256=case['source_sha256'],controller=case['controller'],
        original_outcome={k:original[k] for k in ('duration','score','target_fraction','death') if k in original},
        limitations=case.get('limitation','Saved configuration and controller version rerun locally; new footage, not the original missing recording.'))
    # Original renderer only for two central success/failure demonstrations.
    native=case['id'] in ('controls-final-v5:008','validation-final-v5:020')
    label=f"Sprinter {case['source']} row {case['row']} ({case['controller']})"
    if case['kind']=='single':
        result=run(original['scenario'],original['policy'],original['horizon'],label,
            controller_module=case['controller'],reproduction=provenance,native=native)
    elif case['kind']=='protection':
        result=protect(original['scenario'],original['mode'],original['horizon'],label,
            controller_module=case['controller'],reproduction=provenance,native=False)
    else:
        policy_class=None
        if case['source']=='fullgame-caste-1-v3.json':
            from colony_v3_reconstructed import CastePolicy
            policy_class=CastePolicy
        result=game(original['seed'],original['mode'],original['horizon'],
            reproduction=provenance,native=False,policy_class=policy_class)
    print(json.dumps(dict(id=case['id'],replay=result['replay'],duration=result.get('duration',result['horizon']),old_duration=original.get('duration'),target_fraction=result.get('target_fraction'))),flush=True)

def worker(part,parts,skip_generated=False):
    cases=json.loads(AUDIT.read_text())['cases'];done=completed()
    pending=[c for c in cases if c['status']!='original recording exists']
    for index,case in enumerate(pending):
        if index%parts!=part or case['id'] in done or (skip_generated and case['kind']=='generated'):continue
        command=[sys.executable,str(Path(__file__).resolve()),'--case',case['id']]
        result=subprocess.run(command,cwd=ROOT.parent)
        if result.returncode:
            failures=OUT/f'backfill-failures-{part}.jsonl'
            with failures.open('a') as f:f.write(json.dumps(dict(id=case['id'],returncode=result.returncode))+'\n')


def refresh():
    data=json.loads(AUDIT.read_text());done=completed()
    for case in data['cases']:
        if case['id'] in done:
            receipt,r=done[case['id']]
            case.update(status='new reproduction saved',replay=r['replay'],receipt=str(receipt.relative_to(ROOT)),
                reproduced_outcome={k:r[k] for k in ('duration','score','target_fraction','death') if k in r})
    counts={status:sum(c['status']==status for c in data['cases']) for status in sorted({c['status'] for c in data['cases']})}
    data['counts']=counts
    for name,digest in data['original_files'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
    write_audit(data);print(json.dumps(counts))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inventory',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--case');ap.add_argument('--skip-generated',action='store_true');ap.add_argument('--refresh',action='store_true');a=ap.parse_args()
    if a.inventory:inventory()
    elif a.case:
        if a.case in completed():print('Already reproduced:',a.case,flush=True)
        else:execute(next(c for c in json.loads(AUDIT.read_text())['cases'] if c['id']==a.case))
    elif a.worker is not None:worker(a.worker,a.workers,a.skip_generated)
    elif a.refresh:refresh()
