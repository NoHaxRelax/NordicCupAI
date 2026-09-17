"""Audit saved outcomes against real replays and preserve source provenance."""
import gzip
import hashlib
import json
from pathlib import Path
from run import ROOT, OUT


def audit():
    sources = {hashlib.sha256(p.read_bytes()).hexdigest() for pattern in
               ['run*.py','observed_policy*.py'] for p in Path(__file__).parent.glob(pattern)}
    count = 0
    seen_replays=set()
    for path in OUT.glob('*.json'):
        row = json.loads(path.read_text())
        if not isinstance(row,dict) or 'replay' not in row:
            continue
        if row['replay'] in seen_replays:continue
        seen_replays.add(row['replay'])
        replay = json.loads(gzip.decompress((ROOT/row['replay']).read_bytes()))
        assert replay['format'] == 'survival-replay'
        assert replay['meta']['policy_sha256'] in sources, path
        frames = replay['frames']
        assert frames[0]['t'] == 0
        assert frames[-1]['t'] == row['seconds'], path
        assert all(a['t'] < b['t'] for a,b in zip(frames,frames[1:])), path
        observed='predators' in row
        if observed:
            assert len(frames[-1]['predators']) == row.get('arrivals',row['predators']), path
            if row['all_held']:
                assert row['tail_min']==row['predators']
                assert row['holders_alive']>0
            # Native DTO recordings must not smuggle evaluator fields to policy.
            for frame in frames:
                for agent in frame['agents']:
                    for obs in agent.get('action_observations',[])+agent.get('observations',[]):
                        if obs.get('type')=='Predator':
                            assert not {'id','energy','resting','x','y'} & obs.keys(), path
        else:
            assert len(frames[-1]['predators']) == len(row['deliveries']), path
            assert (any(a['id']==0 for a in frames[-1]['agents'])) == row['holder_alive'], path
        assert len([e for e in replay['events'] if e['type']=='death']) == len(row['deaths']), path
        if row.get('final_success'):
            assert row['holder_alive'] and len(row['deliveries']) == row['waves']
            assert row['seconds'] == row['requested_seconds']
            assert all(d['tail_hold']==1 for d in row['deliveries'])
        count += 1
    assert count > 0
    print(f'Validated {count} outcomes, replay timelines, populations, deaths and source hashes.')


if __name__ == '__main__':
    audit()
