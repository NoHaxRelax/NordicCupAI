"""Rebuild ordinary guide memory, then replace only one fatal forecast.

Run on the original recording's machine/environment. Recorded world state is
used exclusively to verify native replay, never as a policy input.
"""
import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--policy-root',type=Path,required=True)
    parser.add_argument('--fastsim',type=Path,required=True)
    parser.add_argument('--search',type=Path,required=True)
    parser.add_argument('--agent',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.fastsim.resolve()))
    from fastsim import SimulationCore
    sys.path.insert(0,str(args.policy_root.resolve()))
    from models.core import EntrapmentPolicy, local
    from models.entrapment import guide_steering
    from src.utils.DTOs import ActionRequest

    manifest=json.loads((args.folder/'manifest.json').read_text())
    summary=json.loads((args.folder/'summary.json').read_text())
    case=next(c for c in summary['native_evaluation']['sprint_available_predator_death_cases']
              if c['agent']==args.agent and not c['intentional_delivery'])
    options={k:manifest[k] for k in inspect.signature(EntrapmentPolicy).parameters if k in manifest}
    options['survival_settings']=manifest.get('survival_overrides',{})
    policy=EntrapmentPolicy(**options)
    sim=SimulationCore(seed=manifest['seed'])
    states=sim.step([])['observations']
    spec=importlib.util.spec_from_file_location('guide_search_probe',args.search)
    replacement=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=replacement
    spec.loader.exec_module(replacement)
    for file in sorted((args.folder/'chunks').glob('*.json.gz')):
        with gzip.open(file,'rt') as handle:
            rows=json.load(handle)
        for row in rows:
            prior_forecasts={t.key:t.memory.get('_forecast_observation') for t in policy.tracks.values()}
            policy(states,sim.env.time)  # Observations/time only: rebuild maps and guide memories.
            actions=[(a['agent_id'],ActionRequest(**a)) for a in row['actions']]
            if abs(row['time']-case['time'])<1e-6:
                actual={a.agent_id:a for a in sim.env.agents}
                expected={a['agent_id']:a for a in row['world']['agents']}
                if set(actual)!=set(expected): raise RuntimeError('Native replay agent IDs differ')
                error=max(math.hypot(actual[k].x-v['x'],actual[k].y-v['y']) for k,v in expected.items())
                if error>1e-7: raise RuntimeError(f'Native replay position differs by {error}')
                track=next(t for t in policy.tracks.values() if t.guide_id==args.agent)
                pose=policy.estimator.poses[args.agent]
                state=next(s for s in states if s['agent_id']==args.agent)
                original=next(a for aid,a in actions if aid==args.agent)
                # The ordinary policy already called the old search this tick.
                # A replacement must see its previous sample, not that newly
                # written sample, when estimating the unobserved last move.
                previous=prior_forecasts.get(track.key)
                if previous is None: track.memory.pop('_forecast_observation',None)
                else: track.memory['_forecast_observation']=previous
                guide_steering.search=replacement.search
                changed=guide_steering.prioritize(original.model_dump(),local(pose,policy.site['goal']),
                    [(local(pose,a),local(pose,b)) for a,b in track.edges],state,track.memory,
                    target=track.observers.get(args.agent))
                action=original.model_copy(update=changed)
                sim.step([(aid,action if aid==args.agent else a) for aid,a in actions])
                result=dict(seed=manifest['seed'],agent=args.agent,time=row['time'],
                    maximum_replay_position_error=error,original=original.model_dump(),changed=action.model_dump(),
                    alive_after_tick=args.agent in sim.env.agents_dict,
                    original_forecast=case['guide_plan']['forecast'],changed_forecast=track.memory['debug']['forecast'],
                    search_sha256=hashlib.sha256(args.search.read_bytes()).hexdigest(),
                    note='Single native counterfactual tick; policy rebuilt from ordinary DTOs. No future-survival claim.')
                args.out.parent.mkdir(parents=True,exist_ok=True)
                args.out.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result),flush=True)
                return
            states=sim.step(actions)['observations']
            sim.pop_events()
    raise RuntimeError('Fatal tick not present in recording')


if __name__=='__main__':main()
