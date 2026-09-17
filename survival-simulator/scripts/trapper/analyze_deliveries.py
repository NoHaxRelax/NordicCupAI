"""Aggregate delivery outcomes over run JSONs: python scripts/trapper/analyze_deliveries.py results/trapper/trapper-seed*-LABEL-*.json

Deliveries are grouped by how they began (chased agent vs attracted) and whether the guide was
replaced by a transfer; the failure tail classes look at the last trace rows before the end."""
import collections, json, sys

outcomes = collections.Counter(); phases = collections.Counter(); by_turn = collections.defaultdict(lambda: [0, 0])
by_group = collections.defaultdict(collections.Counter)
starts = 0; held = 0.0; n = 0; transfers = 0; handoffs = 0; guards = 0; scores = []
why = collections.Counter(); durations = []
for f in sys.argv[1:]:
    r = json.load(open(f)); n += 1
    held += r.get('held_fraction') or 0
    scores.append(r.get('score'))
    m = r.get('metrics', {})
    transfers += m.get('trap_guide_transfers', 0); handoffs += m.get('trap_handoffs', 0); guards += m.get('trap_guard_interceptions', 0)
    started = {}; transferred = set()
    for e in r['events']:
        k = e['kind']
        if k == 'delivery_started':
            starts += 1; started[e['pid']] = e; transferred.discard(e['pid'])
        elif k == 'guide_transferred':
            transferred.add(e['pid'])
        elif k in ('delivered', 'delivery_failed'):
            s = started.get(e['pid'], {})
            out = 'delivered' if k == 'delivered' else str(e.get('reason'))
            outcomes[out] += 1
            grp = ('chased' if s.get('chased') else 'attract') + ('+transfer' if e['pid'] in transferred else '')
            by_group[grp][out] += 1
            by_turn[min(180, (s.get('turn', 0) // 30) * 30)][0 if k == 'delivered' else 1] += 1
            durations.append(round(e['t'] - s.get('t', e['t']), 1))
            if k == 'delivery_failed':
                phases[e.get('max_phase')] += 1
        if k in ('delivery_failed', 'guide_captured') and e.get('trace'):
            tail = e['trace'][-4:]
            guide = e.get('agent')
            targets = [t[3] for t in tail]
            los = [t[4] for t in tail]
            if k == 'guide_captured':
                why['captured'] += 1
            elif any(t is not None and t != guide for t in targets):
                why['stolen by another agent'] += 1
            elif all(t is None for t in targets) and not any(los):
                why['lost behind obstacle'] += 1
            elif all(t is None for t in targets):
                why['lost in the open (out of cone/range)'] += 1
            else:
                why['other'] += 1
station = collections.Counter()
for f in sys.argv[1:]:
    r = json.load(open(f))
    for e in r['events']:
        k = e['kind']
        if k in ('bait_assigned', 'successor_assigned', 'successor_in_place', 'bait_died', 'refuge_run', 'refuge_entered', 'refuge_flyby', 'flyby', 'handoff'):
            station[k] += 1
        elif k == 'bait_left':
            station[f"bait_left:{e.get('why')}"] += 1
        elif k == 'hold_ended':
            station[f"hold_ended:{e.get('why')}"] += 1
            station['hold_seconds'] += e.get('held_s', 0)
        elif k == 'hold_started':
            station['hold_started'] += 1
        elif k == 'delivered':
            station['delivered:guide_alive' if e.get('guide_alive') else 'delivered:guide_dead'] += 1
    for role, cnt in (r.get('role_deaths') or {}).items():
        if cnt and role != 'null':
            station[f'death:{role}'] += cnt
    m = r.get('metrics') or (r.get('trap') or {})
    station['max_held_one_station'] = max(station['max_held_one_station'], m.get('trap_max_held_one_station', m.get('max_held_one_station', 0)))
    station['extinct_runs'] += int((r.get('alive') or 0) == 0)
print('station events:', dict(sorted(station.items())))
print(f'runs {n}, score mean {sum(s for s in scores if s is not None) / max(1, len(scores)):.1f}, deliveries started {starts}, '
      f'mean held fraction {held / max(n, 1):.3f}, transfers {transfers}, handoffs {handoffs}, guard interceptions {guards}')
print('outcomes:', dict(outcomes.most_common()))
for g in sorted(by_group):
    print(f'  {g:18s}', dict(by_group[g].most_common()))
print('failure tail classes:', dict(why.most_common()))
print('max phase reached before failure:', dict(phases.most_common()))
print('by required turn (deg -> delivered, failed):', {k: v for k, v in sorted(by_turn.items())})
if durations:
    durations.sort()
    print(f'delivery duration s: median {durations[len(durations) // 2]}, p90 {durations[int(len(durations) * 0.9)]}')
