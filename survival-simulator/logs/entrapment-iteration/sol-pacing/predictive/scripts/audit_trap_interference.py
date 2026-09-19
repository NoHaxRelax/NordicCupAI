"""Read-only hearing-exposure audit of recorded native games (never policy input)."""
import argparse
import gzip
import json
import math
from pathlib import Path


def audit(folder):
    steps = exposed = trap_exposed = 0
    ids = set()
    examples = []
    for path in sorted((folder/'chunks').glob('*.json.gz')):
        with gzip.open(path, 'rt') as handle:
            rows = json.load(handle)
        for row in rows:
            roles = row['policy']['roles']
            agents = row['world']['agents']
            predators = row['world']['predators']
            bait = next((a for a in agents if a['agent_id'] == row['policy']['bait']), None)
            held = [p for p in predators if bait and math.hypot(p['x']-bait['x'],p['y']-bait['y']) <= 40.]
            for agent in agents:
                aid = agent['agent_id']
                if roles.get(str(aid)) in ('guide','bait','replacement_bait','retired_bait'):
                    continue
                steps += 1
                nearby = any(math.hypot(p['x']-agent['x'],p['y']-agent['y']) <= 60. for p in predators)
                near_trap = any(math.hypot(p['x']-agent['x'],p['y']-agent['y']) <= 60. for p in held)
                exposed += nearby
                trap_exposed += near_trap
                if nearby:
                    ids.add(aid)
                    if len(examples) < 10:
                        examples.append(dict(time=row['time'],agent=aid,role=roles.get(str(aid)),near_trap=near_trap))
    return dict(folder=str(folder),bystander_agent_ticks=steps,hearing_exposed_agent_ticks=exposed,
                trapped_predator_hearing_exposed_agent_ticks=trap_exposed,
                distinct_exposed_agents=sorted(ids),examples=examples,
                note='Evaluation from recorded world positions; hearing only, not vision. No matched baseline comparison.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
