"""Describe sprint-capture failures using recorded ordinary observations.

These diagnostics identify speed and terrain conditions, not whether an
unobserved alternative route would have survived. Every premature capture
with sprint available remains a benchmark failure, including slow offspring.
"""
import math

TERRAIN = {'forest': 1., 'grassland': 1., 'desert': .8, 'swamp': .5, 'river': .3}
PREDATOR_SPRINT = 15.


def capture_context(case):
    changes = []
    steps = case.get('recent_steps', [])
    for before, after in zip(steps, steps[1:]):
        if before['biome'] == after['biome']:
            continue
        changes.append(dict(time=after['time'],
            from_biome=before['biome'], to_biome=after['biome'],
            slowdown=TERRAIN[after['biome']] < TERRAIN[before['biome']],
            observed_gap_before=before.get('nearest_observed_predator'),
            observed_gap_after=after.get('nearest_observed_predator')))
    factor = TERRAIN[case['biome']]
    speed = case['sprint_speed']
    action = case.get('action') or {}
    requested = max(0., min(action.get('move_distance', 0.), speed))
    slowdown_times = [c['time'] for c in changes if c['slowdown']]
    return dict(
        benchmark_failure=not case['intentional_delivery'],
        same_terrain_sprint_advantage=speed-PREDATOR_SPRINT,
        current_terrain_sprint_distance=speed*factor,
        predator_fast_terrain_sprint_distance=PREDATOR_SPRINT,
        current_terrain_advantage_against_fast_predator=speed*factor-PREDATOR_SPRINT,
        requested_move=requested,
        requested_full_sprint=math.isclose(requested,speed,abs_tol=1e-6),
        expected_move_before_wall_deflection=requested*factor,
        recent_biome_changes=changes,
        seconds_since_last_observed_slowdown=(case['time']-slowdown_times[-1] if slowdown_times else None),
        note='Distances per tick, before wall deflection. Predator terrain is unknown; 15 is a worst case. No recent transition does not exclude older slowdown.')


def report(summary):
    evaluation = summary.get('native_evaluation', {})
    cases = evaluation.get('sprint_available_predator_death_cases')
    if cases is None:
        return dict(seed=summary['seed'], status=summary['status'],
            score=summary['score'], sim_time=summary['sim_time'],
            premature_captures=None, failures_with_recent_observed_slowdown=None,
            failures_without_same_terrain_speed_advantage=None,
            note='All-agent sprint death cases were not recorded in this older replay.')
    failures = [dict(agent=c['agent'], time=c['time'], role=c['role'],
                     biome=c['biome'], **capture_context(c))
                for c in cases if not c['intentional_delivery']]
    return dict(seed=summary['seed'], status=summary['status'],
        score=summary['score'], sim_time=summary['sim_time'],
        premature_captures=len(failures),
        failures_with_recent_observed_slowdown=sum(
            c['seconds_since_last_observed_slowdown'] is not None for c in failures),
        failures_without_same_terrain_speed_advantage=sum(
            c['same_terrain_sprint_advantage'] <= 0 for c in failures),
        intentional_delivery_sacrifices=summary['native_evaluation']['intentional_delivery_sacrifices'],
        failures=failures)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [dict(replay=str(p), **report(json.loads((p/'summary.json').read_text())))
            for p in args.replays]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2)+'\n')
    for row in rows:
        print(f"{row['replay']}: {row['premature_captures']} failures, "
              f"{row['failures_with_recent_observed_slowdown']} after recorded slowdown, "
              f"{row['failures_without_same_terrain_speed_advantage']} without a same-terrain speed advantage")
