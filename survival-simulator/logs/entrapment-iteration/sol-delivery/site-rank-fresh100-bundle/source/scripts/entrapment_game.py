"""Run a native game; record every tick separately from observation-only policy.

From repo root: .venv/bin/python survival-simulator/scripts/entrapment_game.py
The recorder reads world state only AFTER policy evaluation, for replay/metrics.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pygame
from src.core import SimulationCore
from models.core import EntrapmentPolicy


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)): return [clean(v) for v in value]
    if hasattr(value, 'tolist'): return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def atom(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value), allow_nan=False))
    temporary.replace(path)


def drawable(obj, fields):
    return {key: getattr(obj, key) for key in fields}


def record_world(env):
    creature = ('x', 'y', 'direction', 'size', 'color', 'energy', 'max_energy', 'age',
                'hearing_radius', 'vision_radius', 'cone_angle', '_vision_poly')
    return dict(agents=[dict(drawable(a, creature), agent_id=a.agent_id) for a in env.agents],
                predators=[drawable(p, creature) for p in env.predators],
                fruits=[drawable(f, ('x', 'y', 'radius', 'color', 'age')) for f in env.fruits],
                trees=[drawable(t, ('x', 'y', 'radius', 'color', 'age')) for t in env.trees])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3000)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    folder = args.out or ROOT/'logs'/'entrapment_game'/f'seed-{args.seed}'
    if (folder/'summary.json').exists():
        parser.error('Output already contains a run; choose a new --out to preserve it.')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'chunks').mkdir(exist_ok=True)
    start = time.monotonic()
    core = SimulationCore(seed=args.seed)
    policy = EntrapmentPolicy(seed=args.seed)
    env = core.env
    # Native surfaces and draw methods are reused by the viewer. No alternate
    # gameplay engine, teleports, spawned fixtures, or mechanic overrides.
    background = env.static_surface.copy()
    background.blit(env.shadow_surface, (0, 0))
    background.blit(env.obstacle_surface, (0, 0))
    pygame.image.save(background, folder/'background.png')
    sources = [*sorted((ROOT/'models').rglob('*.py')), *sorted((ROOT/'models'/'nikolaj'/'config').glob('*.json')),
               *sorted((ROOT/'src').rglob('*.py')), Path(__file__)]
    atom(folder/'manifest.json', dict(seed=args.seed, horizon=args.seconds, dt=core.dt,
        policy_inputs='Unmodified per-agent DTOs and simulation time only',
        native_defaults=dict(starting_agents=5, starting_predators=0, starting_trees=50),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    atom(folder/'static.json', dict(width=env.width, height=env.height, edges=env.edges))
    states = [env.get_agent_state(a.agent_id) for a in env.agents]
    chunk, history, peak, seen_ids, steps = [], [], len(states), set(), 0
    first_bait_time = None
    bait_gap = longest_gap = gaps_after_first_bait = 0.
    max_near_bait = 0
    active_since = {}
    retained_ids = set()
    predator_ids = {}
    next_pid = 0
    retained_max = 0
    tick = 0
    try:
        for tick in range(round(args.seconds/core.dt)+1):
            now = env.time
            terminal = not states or tick == round(args.seconds/core.dt)
            actions = [] if terminal else policy(states, now)
            debug = policy.snapshot()
            # Evaluator-only metrics. This block cannot affect policy inputs.
            site = policy.site
            holding_ids = set(policy.retired_baits) | {policy.bait, policy.incoming}
            baits = [a for a in env.agents if a.agent_id in holding_ids and
                     policy.site is not None and policy._arrival(a.agent_id)]
            if baits and first_bait_time is None: first_bait_time = now
            if first_bait_time is not None and not baits:
                bait_gap += core.dt
                gaps_after_first_bait += core.dt
                longest_gap = max(longest_gap, bait_gap)
            else: bait_gap = 0.
            near = 0
            for pred in env.predators:
                if pred not in predator_ids:
                    predator_ids[pred] = next_pid
                    next_pid += 1
                pid = predator_ids[pred]
                close = baits and min(math.hypot(pred.x-b.x, pred.y-b.y) for b in baits) <= 40.
                if close:
                    near += 1
                    active_since.setdefault(pid, now)
                    if now-active_since[pid] >= 30.: retained_ids.add(pid)
                else:
                    active_since.pop(pid, None)
            held30 = sum(now-since >= 30. for since in active_since.values())
            retained_max = max(retained_max, held30)
            max_near_bait = max(max_near_bait, near)
            seen_ids.update(s['agent_id'] for s in states)
            peak = max(peak, len(states))
            metrics = dict(agents=len(states), predators=len(env.predators), fruit=len(env.fruits),
                           score=env.score, near_bait=near, held30=held30,
                           bait_present_estimated=bool(baits), bait_energy=[a.energy for a in baits])
            if tick % 10 == 0: history.append(dict(time=now, **metrics))
            chunk.append(clean(dict(tick=tick, time=now, world=record_world(env), policy=debug,
                                    input=states, actions=[a.model_dump() for _, a in actions], evaluation=metrics)))
            if len(chunk) == 100 or terminal:
                first = tick-len(chunk)+1
                with gzip.open(folder/'chunks'/f'{first//100:05d}.json.gz', 'wt', compresslevel=1) as handle:
                    json.dump(chunk, handle, separators=(',', ':'), allow_nan=False)
                chunk = []
                summary = dict(seed=args.seed, status='complete' if terminal else 'running', frames=tick+1,
                    sim_time=now, runtime_seconds=time.monotonic()-start, score=env.score,
                    final_agents=len(states), peak_agents=peak, total_agents_seen=len(seen_ids),
                    predators=len(env.predators), first_bait_time=first_bait_time,
                    maximum_predators_within_40_of_bait=max_near_bait,
                    maximum_predators_continuously_near_bait_30s=retained_max,
                    distinct_predators_ever_near_bait_30s=len(retained_ids),
                    estimated_bait_gap_seconds_after_first_arrival=gaps_after_first_bait,
                    longest_estimated_bait_gap_seconds=longest_gap,
                    policy_metrics=policy.metrics, events=policy.events, history=history,
                    metric_note='Near bait for 30s is a proximity proxy, not proof of permanent capture. Bait occupancy uses estimated localization.')
                atom(folder/'summary.json', summary)
            if tick % 500 == 0:
                print(json.dumps(dict(tick=tick, time=round(now, 1), agents=len(states), predators=len(env.predators),
                      phase=debug['phase'], bait=policy.bait, guides=policy.metrics['guide_assignments'],
                      near_bait=near, elapsed=round(time.monotonic()-start, 1))), flush=True)
            if terminal: break
            state = core.step(actions)
            states = state['observations']
            steps += 1
    except Exception:
        import traceback
        atom(folder/'error.json', dict(tick=tick, error=traceback.format_exc()))
        raise
    print(json.dumps({k: v for k, v in summary.items() if k not in ('history', 'events')}, indent=2))
    print('Replay folder:', folder)


if __name__ == '__main__': main()
