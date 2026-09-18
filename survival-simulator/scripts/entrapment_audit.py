"""Post-run spectator audit and optional replay of policy using public DTOs only."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.observed_trap_sites import observed_rectangles, our_sites


def audit(folder, recheck_policy=False):
    summary = json.loads((folder/'summary.json').read_text())
    static = json.loads((folder/'static.json').read_text())
    group = SimpleNamespace(anchored=True, world_size=(static['width'], static['height']),
                            edges=[SimpleNamespace(start=np.array(a), end=np.array(b)) for a, b in static['edges']])
    # Full geometry is used HERE, after the game, solely as an evaluator.
    true_sites = our_sites(observed_rectangles(group, tolerance=.0001))
    policy = None
    if recheck_policy:
        from models.entrapment_policy import EntrapmentPolicy
        policy = EntrapmentPolicy()
    expected = checked = rear_ticks = maximum_rear = 0
    first = last_occupied = None
    gap = longest = current = maximum_error = 0.
    arrivals, chunks, differences = set(), [], []
    for filename in sorted((folder/'chunks').glob('*.json.gz')):
        chunks.append(dict(file=filename.name, sha256=hashlib.sha256(filename.read_bytes()).hexdigest()))
        with gzip.open(filename, 'rt') as handle: rows = json.load(handle)
        for row in rows:
            if row['tick'] != expected or abs(row['time']-expected*.1) >= 1e-6:
                raise ValueError(f'Missing or misordered tick at {expected}')
            expected += 1
            if policy is not None and row['actions']:
                # Do not pass world, static geometry, summary, or evaluator
                # decisions into this separate controller instance.
                actual = [a.model_dump() for _, a in policy(row['input'], row['time'])]
                recorded = row['actions']
                equal = len(actual) == len(recorded)
                if equal:
                    for a, b in zip(actual, recorded):
                        equal &= a['agent_id'] == b['agent_id'] and a['spawn_agent'] == b['spawn_agent']
                        equal &= all(abs(a[k]-b[k]) < 1e-8 for k in ('move_distance', 'move_direction', 'turn_angle'))
                if not equal:
                    differences.append(dict(tick=row['tick'], expected=recorded, actual=actual))
                    # One divergence is enough to reject reproduction. Continue
                    # the physical audit, but do not assert later equivalence.
                    policy = None
                else: checked += 1
            pol, world = row['policy'], row['world']
            poses, site, occupied = pol['estimated_agents'], pol['site'], []
            if site and true_sites:
                refs = [a for a in world['agents'] if str(a['agent_id']) in poses
                        and poses[str(a['agent_id'])]['group'] == pol['site_group']]
                refs.sort(key=lambda a: (pol['roles'].get(str(a['agent_id'])) != 'bait',
                                         poses[str(a['agent_id'])]['uncertainty']))
                if refs:
                    a = refs[0]; pose = poses[str(a['agent_id'])]
                    angle = a['direction']-pose['heading']
                    c, s = math.cos(angle), math.sin(angle)
                    x, y = np.array(site['goal'])-pose['position']
                    projected = (a['x']+c*x-s*y, a['y']+s*x+c*y)
                    true_site = min(true_sites, key=lambda q: math.dist(projected, q['goal']))
                    error = math.dist(projected, true_site['goal'])
                    maximum_error = max(maximum_error, error)
                    if error < 5.:
                        occupied = [a for a in world['agents'] if pol['roles'].get(str(a['agent_id']))
                                    in ('bait', 'replacement_bait', 'retired_bait')
                                    and math.dist((a['x'], a['y']), true_site['goal']) <= 3.]
                        rear = sum(math.dist((pr['x'], pr['y']), true_site['other_mouth']) <= 40.
                                   and sum((pr[k]-true_site['mouth'][i])*true_site['inward'][i]
                                           for i, k in enumerate(('x', 'y'))) >= true_site['overlap']-5.
                                   for pr in world['predators'])
                        maximum_rear = max(maximum_rear, rear)
                        rear_ticks += bool(rear)
            if occupied:
                arrivals.update(a['agent_id'] for a in occupied)
                last_occupied = row['time']
                current = 0.
                if first is None: first = row['time']
            elif first is not None:
                gap += .1; current += .1; longest = max(longest, current)
        if len(chunks) % 20 == 0:
            print(json.dumps(dict(audited_ticks=expected, rechecked_policy_ticks=checked,
                                  divergences=len(differences))), flush=True)
    if expected != summary['frames']: raise ValueError('Frame count differs from summary')
    manifest = json.loads((folder/'manifest.json').read_text())
    changed = [f for f, h in manifest['sources'].items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest() != h]
    result = dict(frames=expected, contiguous_ticks=True, source_changes_since_run=changed,
                  policy_recheck_requested=recheck_policy, policy_ticks_rechecked=checked,
                  policy_action_divergences=differences,
                  true_map_eligible_sites=len(true_sites), physical_first_bait=first,
                  physical_last_occupied_time=last_occupied,
                  physical_bait_gap_seconds=round(gap, 3), physical_longest_gap_seconds=round(longest, 3),
                  physical_bait_agents=len(arrivals), maximum_projected_site_error=maximum_error,
                  rear_occupied_ticks=rear_ticks, maximum_predators_at_rear=maximum_rear,
                  method='Post-run full-map evaluator matches candidate; actual bait must be within 3 units of true depth-5 goal. Rear occupation: within 40 of opposite mouth and axial depth >= overlap-5. Never feeds the live policy.',
                  chunks=chunks)
    (folder/'audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ('chunks', 'policy_action_divergences')}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--recheck-policy', action='store_true')
    args = parser.parse_args()
    audit(args.folder, args.recheck_policy)
