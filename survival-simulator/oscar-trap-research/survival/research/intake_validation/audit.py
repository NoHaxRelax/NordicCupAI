"""Independently audit all completed intake research receipts and replay DTOs."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
FOLDERS=['intake_validation','intake_gate_sol','intake_guides_sol','intake_geometry_sol']

def audit():
    hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').rglob('*.py')}
    for wrapper in (ROOT/'research').rglob('observed_gap_guide_policy_v2.py'):
        dependency=wrapper.with_name('observed_gap_base_v2_snapshot.py')
        if dependency.exists():hashes.add(hashlib.sha256(wrapper.read_bytes()+dependency.read_bytes()).hexdigest())
    for wrapper in (ROOT/'research').rglob('observed_gap_guide_policy.py'):
        dependency=wrapper.with_name('observed_gap_base_snapshot.py')
        if dependency.exists():hashes.add(hashlib.sha256(wrapper.read_bytes()+dependency.read_bytes()).hexdigest())
    for version in (3,4):
        for wrapper in (ROOT/'research').rglob(f'observed_gap_guide_policy_v{version}.py'):
            dependency=wrapper.with_name('observed_gap_base_v2_snapshot.py')
            if dependency.exists():hashes.add(hashlib.sha256(wrapper.read_bytes()+dependency.read_bytes()).hexdigest())
    for wrapper in (ROOT/'research').rglob('observed_shallow_gap_policy.py'):
        dependency=wrapper.with_name('observed_gap_base_v2_snapshot.py')
        if dependency.exists():hashes.add(hashlib.sha256(wrapper.read_bytes()+dependency.read_bytes()).hexdigest())
    seen=set(); errors=[]; warnings=[]
    for folder in FOLDERS:
        for path in (ROOT/'results'/folder).rglob('*.json'):
            row=json.loads(path.read_text())
            if not isinstance(row,dict) or 'replay' not in row or row['replay'] in seen:continue
            seen.add(row['replay'])
            try:
                replay=json.loads(gzip.decompress((ROOT/row['replay']).read_bytes()))
                frames=replay['frames']
                assert replay['format']=='survival-replay'
                assert replay['meta']['policy_sha256'] in hashes, 'missing archived policy source'
                assert frames[0]['t']==0
                assert frames[-1]['t']==row['seconds'], 'duration differs'
                assert all(a['t']<b['t'] for a,b in zip(frames,frames[1:])), 'frame chronology'
                arrived=row.get('arrivals',row.get('spawned',row.get('predators')))
                if isinstance(arrived,int):assert len(frames[-1]['predators'])==arrived, 'population differs'
                recorded_deaths={e['entity_id'] for e in replay['events'] if e['type']=='death'}
                expected_deaths={e['id'] for e in row['deaths']}
                if recorded_deaths != expected_deaths:
                    missing=expected_deaths-recorded_deaths
                    assert not recorded_deaths-expected_deaths, 'unexpected replay deaths'
                    for aid in missing:
                        death=next(e for e in row['deaths'] if e['id']==aid)
                        delivery=next((e for e in row.get('deliveries',[]) if e.get('guide_id')==aid),None)
                        assert delivery and abs(death['time']-delivery['spawned']-.1)<.001, 'unexplained missing death'
                        assert not any(a['id']==aid for f in frames for a in f['agents']), 'recorded live agent missing death'
                    warnings.append({'receipt':str(path.relative_to(ROOT)), 'warning':'Staged guides born and killed within one step were absent from replay entity/death events; receipt records them. No frames reconstructed.', 'guide_ids':sorted(missing)})
                for f in frames:
                    for a in f['agents']:
                        for o in a.get('observations',[])+a.get('action_observations',[]):
                            if o.get('type')=='Predator':
                                assert not {'id','energy','resting','x','y','target'} & o.keys(), 'privileged predator DTO'
                if row.get('joint_success'):
                    assert row['seconds']==row['requested_seconds']
                    assert row['arrivals']==row['acquired']==row['tail_joint_min']==row['predators']
                    assert row['joint_losses']==0 and any(a['id']==0 for a in frames[-1]['agents'])
                if row.get('physical_success'):
                    assert row['seconds']==row['requested_seconds']
                    assert row['arrivals']==row['physical_acquired']==row['tail_physical_min']==row['predators']
                    assert row['physical_losses']==0
            except Exception as e:errors.append({'receipt':str(path.relative_to(ROOT)),'error':str(e)})
    result={'completed_replays':len(seen),'errors':errors,'warnings':warnings}
    (ROOT/'results/intake_validation/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    raise SystemExit(bool(audit()['errors']))
