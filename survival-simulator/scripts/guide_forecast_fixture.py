"""Iterate on a captured public guide input without rebuilding a whole policy.

Optional native verification replays original actions up to the captured tick,
then substitutes only this guide's new action. World state verifies replay;
it is never an input to action selection. Run that check on the original host.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify(folder, fixture, changed):
    from fastsim import SimulationCore
    from src.utils.DTOs import ActionRequest

    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['seed']!=fixture['seed']:
        raise RuntimeError('Fixture and native recording seeds differ')
    sim=SimulationCore(seed=fixture['seed'])
    sim.step([])
    for file in sorted((folder/'chunks').glob('*.json.gz')):
        with gzip.open(file,'rt') as handle:
            rows=json.load(handle)
        for row in rows:
            actions=[(a['agent_id'],ActionRequest(**a)) for a in row['actions']]
            if abs(row['time']-fixture['time'])<1e-6:
                actual={a.agent_id:a for a in sim.env.agents}
                expected={a['agent_id']:a for a in row['world']['agents']}
                if set(actual)!=set(expected):
                    raise RuntimeError('Native replay agent IDs differ')
                error=max(math.hypot(actual[k].x-v['x'],actual[k].y-v['y'])
                          for k,v in expected.items())
                if error>1e-7 or abs(sim.env.time-row['time'])>1e-7:
                    raise RuntimeError(f'Native replay differs before counterfactual: {error}')
                # Refuse to apply a public fixture to a different observation.
                ordinary=next(s for s in row['input'] if s['agent_id']==fixture['agent'])
                if ordinary!=fixture['state']:
                    raise RuntimeError('Fixture DTO differs from the recorded tick')
                original=next(a for aid,a in actions if aid==fixture['agent'])
                replacement=original.model_copy(update=changed)
                sim.step([(aid,replacement if aid==fixture['agent'] else a) for aid,a in actions])
                return dict(maximum_replay_position_error=error,
                    alive_after_tick=fixture['agent'] in sim.env.agents_dict,
                    note='One native tick only; no future-survival claim.')
            if row['time']>fixture['time']:
                raise RuntimeError('Fixture time not present in recording')
            sim.step(actions)
            sim.pop_events()
    raise RuntimeError('Fixture time not present in recording')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture',type=Path)
    p.add_argument('--search',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--replay',type=Path,help='Optional original native replay folder')
    p.add_argument('--fastsim',type=Path,help='Native engine root; required with --replay')
    args=p.parse_args()
    if args.replay and not args.fastsim:
        p.error('--replay requires --fastsim')
    if args.fastsim:
        sys.path.insert(0,str(args.fastsim.resolve()))
        import fastsim  # Load the selected engine before our policy imports.
    sys.path.insert(0,str(ROOT))
    from models.entrapment import guide_steering

    spec=importlib.util.spec_from_file_location('guide_search_probe',args.search)
    search_sha256=hashlib.sha256(args.search.read_bytes()).hexdigest()
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    fixture=json.loads(args.fixture.read_text())
    original=fixture.get('recorded_action',fixture.get('nominal_action'))
    if original is None:
        raise ValueError('Fixture has no recorded action')
    guide_steering.search=module.search
    changed=guide_steering.prioritize(original,fixture['bait'],fixture['edges'],
        fixture['state'],fixture['memory'],target=fixture['target_observation'])
    result=dict(seed=fixture['seed'],agent=fixture['agent'],time=fixture['time'],
        original=original,changed={**original,**changed},
        changed_forecast=fixture['memory']['debug']['forecast'],
        search_sha256=search_sha256,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        note='Action chosen from captured ordinary DTO and observed-map search inputs only.')
    if args.replay:
        result['native']=verify(args.replay,fixture,changed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
