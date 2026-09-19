"""Replay recorded native actions and change only a sprint victim's fatal move.

This checks a single counterfactual native tick, not future survival. Policy
inputs are the victim's ordinary DTO and recorded estimated traffic geometry.
Native positions only verify that replay reached the original state exactly.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--fastsim', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.fastsim))
    from fastsim import SimulationCore
    sys.path.insert(0,str(ROOT))
    from models.entrapment.bystander_avoidance import avoid_predators
    from src.utils.DTOs import ActionRequest

    summary = json.loads((args.folder/'summary.json').read_text())
    cases = [c for c in summary['native_evaluation']['sprint_available_predator_death_cases']
             if not c['intentional_delivery'] and c['role']!='guide']
    fatal_times = {round(c['time'],6) for c in cases}
    rows = []
    for path in sorted((args.folder/'chunks').glob('*.json.gz')):
        # Keep action history, not thousands of full spectator world snapshots.
        with gzip.open(path,'rt') as handle:
            chunk = json.load(handle)
        rows.extend(row if round(row['time'],6) in fatal_times else
                    dict(time=row['time'],actions=row['actions']) for row in chunk)
        del chunk
    results = []
    for case in cases:
        aid = case['agent']
        tick = min(range(len(rows)),key=lambda i:abs(rows[i]['time']-case['time']))
        row = rows[tick]
        pose = row['policy']['estimated_agents'][str(aid)]

        def local(point):
            x,y = point[0]-pose['position'][0],point[1]-pose['position'][1]
            h = pose['heading']
            return x*math.cos(h)+y*math.sin(h),-x*math.sin(h)+y*math.cos(h)

        bait = local(row['policy']['site']['goal']) if row['policy']['bait'] is not None else None
        paths = [[local(p) for p in c['points']] for c in row['policy']['guide_corridors']
                 if c['group']==pose['group'] and c['guide']!=aid
                 and min(math.dist(p,pose['position']) for p in c['points'])<=320.]
        original = next(a for a in row['actions'] if a['agent_id']==aid)
        ordinary = next(s for s in row['input'] if s['agent_id']==aid)
        changed, _ = avoid_predators(ActionRequest(**original),ordinary,bait,(),paths)
        outcome = dict(agent=aid,time=row['time'],original=original,changed=changed.model_dump())
        for label,replace_action in [('original',False),('changed',True)]:
            sim = SimulationCore(seed=summary['seed'])
            sim.step([])
            for prior in rows[:tick]:
                sim.step([(a['agent_id'],ActionRequest(**a)) for a in prior['actions']])
                sim.pop_events()
            actual = {a.agent_id:a for a in sim.env.agents}
            expected = {a['agent_id']:a for a in row['world']['agents']}
            if set(actual)!=set(expected): raise RuntimeError('Replay living-agent IDs differ')
            error = max((math.hypot(actual[k].x-v['x'],actual[k].y-v['y']) for k,v in expected.items()),default=0.)
            if error > 1e-7 or abs(sim.env.time-row['time'])>1e-7:
                raise RuntimeError(f'Replay differs before counterfactual: {error}')
            actions = [(a['agent_id'],changed if replace_action and a['agent_id']==aid else ActionRequest(**a))
                       for a in row['actions']]
            sim.step(actions)
            events = sim.pop_events()
            outcome[label+'_alive_after_tick'] = aid in sim.env.agents_dict
            outcome[label+'_deaths'] = [list(e) for e in events if e[0] in ('predator','starvation')]
            outcome['maximum_replay_position_error'] = error
        results.append(outcome)
        print(json.dumps(outcome),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(seed=summary['seed'],source_replay=str(args.folder),
        changed_source_sha256=hashlib.sha256((ROOT/'models/entrapment/bystander_avoidance.py').read_bytes()).hexdigest(),
        note='Single native tick. Local DTO and estimated corridors only; no future-survival claim.',cases=results),indent=2)+'\n')


if __name__=='__main__': main()
