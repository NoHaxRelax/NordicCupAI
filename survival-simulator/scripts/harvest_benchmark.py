"""Offline reproducible harvest validation with truth-only diagnostic counters.

python -m scripts.harvest_benchmark --seeds 1000 --workers 16 --out runs/harvest.jsonl
Policy inputs remain public state dictionaries. Engine truth below is used only
for evaluation and never fed back to the controller or survival substrate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import platform
import time
from types import SimpleNamespace

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')


def run_game(seed, horizon=3000., engine='native', policy_kwargs=None, collect_log=False, arm='candidate'):
    from exploit_lab import safety
    from exploit_lab.reliable_harvest import HarvestPolicy, action
    from fastsim.fastpolicy import PolicySimulationCore
    safety.assert_offline('harvest benchmark')
    policy = HarvestPolicy(horizon=horizon, **(policy_kwargs or {}))
    core = PolicySimulationCore(seed=seed)
    core.policy_init(0, {})  # fixed policy RNG, not the hidden world seed
    if engine == 'python':
        from src.core import SimulationCore
        sim = SimulationCore(seed=seed)
    else:
        sim = core
    st = sim.step([])
    metrics = dict(actual_harvests=0, sleeping_targets=0, early_wakes=0, false_ignores=0,
                   first_tick_harvests=0, failed_farms=0, ignored_sightings=0)
    sleeping = set()
    pending_farms = set()
    log = []
    peak = 0
    start = time.monotonic()
    previous_predators = {}
    target_pids = {}
    previous_predator_poses = {}
    observation_poses = {}
    while st['observations'] and st['sim_time'] < horizon:
        states = st['observations']
        now = st['sim_time']
        filtered = policy.observe(states, now, st['score']) if arm == 'candidate' else states
        native = [action(a, d, di, t, sp) for a, d, di, t, sp in core._engine.policy_observe(filtered, now)]
        out = policy.actions(states, now, native) if arm == 'candidate' else native
        event = policy.last_event
        agents = {a.agent_id: a for a in sim.env.agents}
        predators = sim.env.predators
        pids = {getattr(p, 'predator_id', id(p)): p for p in predators}
        for s in states:
            aid = s['agent_id']
            if aid not in observation_poses or observation_poses[aid][0] != s['age']:
                a = agents[aid]
                observation_poses[aid] = (s['age'], a.x, a.y, a.direction)
        # The controller never receives these coordinates or energy values.
        for aid, tracks in policy.memory.current.items():
            a = agents.get(aid)
            if a is None:
                continue
            for t in tracks:
                if policy.memory.sleeping(t, now):
                    _, ox, oy, heading = observation_poses[aid]
                    point = complex(ox, oy) + t.point * complex(math.cos(heading), math.sin(heading))
                    matches = [p for p in predators if abs(complex(p.x, p.y) - point) < 1e-5
                               and abs(math.remainder(p.direction-heading-t.heading, 2*math.pi)) < 1e-5]
                    metrics['ignored_sightings'] += 1
                    if len(matches) != 1 or not matches[0].resting:
                        metrics['false_ignores'] += 1
                        if collect_log and metrics['false_ignores'] <= 10:
                            log.append(dict(kind='false_ignore', time=now, aid=aid,
                                            energy=a.energy, age=a.age,
                                            public_age=next(s['age'] for s in states if s['agent_id']==aid),
                                            obs=t.obs, stationary=t.stationary,
                                            expected=[point.real,point.imag],
                                            predators=[dict(x=p.x,y=p.y,resting=p.resting,energy=p.energy) for p in predators]))
        if event and event['kind'] in ('attempt', 'retry'):
            fid = event['farm']
            pending_farms.add(fid)
            a = agents[fid]
            t = next(t for t in policy.memory.current[fid] if t.key == policy.pending.target_key)
            heading_velocity = complex(math.cos(t.heading), math.sin(t.heading))*abs(t.motion)
            point = complex(a.x, a.y) + (t.point+heading_velocity) * complex(math.cos(a.direction), math.sin(a.direction))
            if predators:
                if event['kind'] == 'attempt':
                    observed = complex(a.x,a.y) + t.point*complex(math.cos(a.direction),math.sin(a.direction))
                    matches = [pid for pid,(x,y,direction) in previous_predator_poses.items()
                               if abs(complex(x,y)-observed) < 1e-5
                               and abs(math.remainder(direction-a.direction-t.heading, 2*math.pi)) < 1e-5]
                    if len(matches) != 1:
                        raise AssertionError('ambiguous diagnostic target identity')
                    pid = matches[0]
                    target_pids[fid] = pid
                p = pids[target_pids[fid]]
                metrics['sleeping_targets'] += bool(p.resting)
                if collect_log and p.resting:
                    log.append(dict(kind='sleeping_target', time=now, farm=fid,
                                    expected=[point.real,point.imag],
                                    predators=[dict(x=p.x,y=p.y,resting=p.resting,energy=p.energy) for p in predators]))
            if collect_log:
                log.append(dict(event, farm_energy=a.energy, population=len(agents),
                                witnesses=list(policy.pending.witnesses),
                                farm_position=[a.x,a.y,a.direction],
                                target_local=[t.point.real,t.point.imag],
                                velocity_local=[t.motion.real,t.motion.imag],
                                biome=next(s['biome'] for s in states if s['agent_id']==fid),
                                predators=[dict(x=p.x,y=p.y,direction=p.direction,resting=p.resting,energy=p.energy) for p in predators]))
        peak = max(peak, len(out))
        previous_predators = {pid: p.energy for pid, p in pids.items()}
        previous_predator_poses = {pid:(p.x,p.y,p.direction) for pid,p in pids.items()}
        step_actions = [(a['agent_id'], SimpleNamespace(**a) if engine == 'python' else a) for a in out]
        st = sim.step(step_actions)
        if event and event['kind'] in ('attempt', 'retry') and collect_log:
            farm_after = next((a for a in sim.env.agents if a.agent_id == event['farm']), None)
            if farm_after:
                log.append(dict(kind='delivery_survivor', farm=farm_after.agent_id,
                                x=farm_after.x, y=farm_after.y, time=st['sim_time'],
                                nearest=min((math.hypot(p.x-farm_after.x, p.y-farm_after.y) for p in sim.env.predators), default=None)))
        next_agents = {s['agent_id'] for s in st['observations']}
        drained = []
        for p in sim.env.predators:
            pid = getattr(p, 'predator_id', id(p))
            if pid in sleeping and not p.resting:
                metrics['early_wakes'] += 1
                sleeping.remove(pid)
            if p.energy < 0 and previous_predators.get(pid, 0)-p.energy > 100:
                metrics['actual_harvests'] += 1
                sleeping.add(pid)
                drained.append(pid)
                # Mathematical certificate checked against actual post-meal energy.
                if p.energy + 30*(horizon-st['sim_time']+.1) > 100:
                    raise AssertionError('drain cannot cover remaining horizon')
        if event and event['kind'] == 'attempt' and drained:
            metrics['first_tick_harvests'] += 1
        for fid in pending_farms - next_agents:
            if not drained:
                metrics['failed_farms'] += 1
                if collect_log:
                    log.append(dict(kind='failed_delivery', farm=fid, time=st['sim_time'],
                                    predators=[dict(x=p.x, y=p.y, energy=p.energy, resting=p.resting) for p in sim.env.predators]))
        pending_farms.intersection_update(next_agents)
        if engine == 'native':
            sim.pop_events()  # bound diagnostic memory
    # Resolve the final step too; there need not be another action tick.
    if arm == 'candidate':
        policy.observe(st['observations'], st['sim_time'], st['score'])
    return dict(seed=seed, engine=engine, arm=arm, horizon=horizon, score=st['score'],
                sim_time=st['sim_time'], population=len(st['observations']), peak_actions=peak,
                wall=time.monotonic()-start, **policy.metrics, **metrics, log=log)


def summary(rows):
    scores = sorted(r['score'] for r in rows)
    result = dict(n=len(rows), mean_score=statistics.mean(scores), min_score=min(scores),
                  p5_score=scores[int(.05*len(scores))], median_score=statistics.median(scores),
                  mean_survival=statistics.mean(r['sim_time'] for r in rows),
                  full_horizon=sum(r['sim_time'] >= r['horizon']-.01 for r in rows))
    for key in ('attempts', 'confirmed', 'failed', 'actual_harvests', 'first_tick_harvests',
                'failed_farms', 'sleeping_targets', 'early_wakes', 'false_ignores', 'ignored_sightings',
                'retries', 'sacrifices', 'cap_rejections', 'floor_rejections', 'skip_rejections'):
        result[key] = sum(r[key] for r in rows)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=1000)
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--horizon', type=float, default=3000.)
    p.add_argument('--engine', choices=['native', 'python'], default='native')
    p.add_argument('--arm', choices=['candidate', 'control'], default='candidate')
    p.add_argument('--extra-drain-actions', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--log', action='store_true')
    a = p.parse_args()
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [root/'exploit_lab/reliable_harvest.py', root/'exploit_lab/predator_memory.py',
               Path(__file__), root/'fastsim/build-info-policy.json', root/'fastsim/_engine.cpp']
    sources += list((root/'src/elements').glob('*.py')) + [root/'src/utils/simulation.py', root/'src/core.py']
    manifest = dict(arguments=vars(a), python=platform.python_version(), platform=platform.platform(),
                    source_sha256={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    with path.open('w') as f, ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(run_game, s, a.horizon, a.engine, dict(extra_drain_actions=a.extra_drain_actions), a.log, a.arm)
                for s in range(a.start, a.start+a.seeds)]
        for job in as_completed(jobs):
            r = job.result()
            rows.append(r)
            f.write(json.dumps(r)+'\n'); f.flush()
            print(json.dumps(dict(completed=len(rows), seed=r['seed'], wall=round(r['wall'], 1))), flush=True)
    result = summary(rows)
    path.with_suffix('.summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
