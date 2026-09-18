"""Parallel native-game benchmark of the frozen 9059 policy, without fixtures.

Each process owns one native game. No observation, mechanic or decision function
is replaced. Spectator metrics and compact trajectories never enter the policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

# One process per requested vCPU; do not create a BLAS thread pool per game.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE_COMMIT = '9b0c3e8'
ROLE_IDS = {'explorer': 0, 'gatherer': 1, 'avoiding_predator': 2,
            'guide': 3, 'bait': 4, 'replacement_bait': 5, 'retired_bait': 6}

# The benchmark manifest records the original 9059 layout. These moves are
# structural only, so verification maps current files back to their recorded
# names and canonicalizes the rewritten imports before hashing them.
SOURCE_MOVES = {
    'models/entrapment_policy.py': 'models/entrapment/entrapment_policy.py',
    'models/entrapment_sites.py': 'models/entrapment/entrapment_sites.py',
    'models/guide_pathfinding.py': 'models/entrapment/guide_pathfinding.py',
    'models/guide_steering.py': 'models/entrapment/guide_steering.py',
    'models/my_guide.py': 'models/entrapment/my_guide.py',
    'models/observed_trap_sites.py': 'models/entrapment/observed_trap_sites.py',
    'models/predator_following.py': 'models/entrapment/predator_following.py',
    'models/oscar_orchard.py': 'models/survival/oscar_orchard.py',
}
IMPORT_REWRITES = (
    (b'models.exploration', b'models.nikolaj'),
    (b'models.survival.oscar_orchard', b'models.oscar_orchard'),
    (b'models.entrapment.entrapment_policy', b'models.entrapment_policy'),
    (b'models.entrapment.entrapment_sites', b'models.entrapment_sites'),
    (b'models.entrapment.observed_trap_sites', b'models.observed_trap_sites'),
    (b'models.entrapment.my_guide', b'models.my_guide'),
    (b'models.entrapment.guide_pathfinding', b'models.guide_pathfinding'),
    (b'models.entrapment.guide_steering', b'models.guide_steering'),
    (b'models.entrapment.predator_following', b'models.predator_following'),
)


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)): return [clean(v) for v in value]
    if hasattr(value, 'tolist'): return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value), separators=(',', ':'), allow_nan=False))
    temp.replace(path)


def verify_sources():
    manifest = json.loads((ROOT/'docs/entrapment_native_seed0/manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        current_name = (name.replace('models/nikolaj/', 'models/exploration/')
                        if name.startswith('models/nikolaj/') else SOURCE_MOVES.get(name, name))
        source = (ROOT/current_name).read_bytes()
        for current, recorded in IMPORT_REWRITES:
            source = source.replace(current, recorded)
        if hashlib.sha256(source).hexdigest() != expected:
            raise RuntimeError(f'Frozen 9059 source mismatch: {name}')
    return manifest


class Observer:
    """Evaluator-only state; nothing from here is passed to EntrapmentPolicy."""
    def __init__(self, env):
        from models.entrapment.observed_trap_sites import our_sites
        self.static = dict(width=env.width, height=env.height,
                           obstacles=[(o.x, o.y, o.width, o.height) for o in env.obstacles])
        self.true_sites = our_sites(self.static)
        self.first_bait = self.last_bait = None
        self.gap = self.longest_gap = self.current_gap = 0.
        self.near_since = {}
        self.ever_held = set()
        self.max_near = self.max_held = self.max_rear = 0
        self.rear_ticks = 0
        self.peak = 0
        self.agents_seen = set()
        self.pred_ids = {}
        self.agents_catalog = {}
        self.history = []
        self.actual_site = None
        self.mapping_error = None

    def measure(self, env, policy, tick):
        now = env.time
        self.peak = max(self.peak, len(env.agents))
        self.agents_seen.update(a.agent_id for a in env.agents)
        roles, poses = policy.roles, policy.estimator.poses
        baits = []
        self.actual_site = None
        if policy.site is not None and self.true_sites:
            refs = [a for a in env.agents if a.agent_id in poses and poses[a.agent_id].group_id == policy.site_group]
            refs.sort(key=lambda a: (a.agent_id != policy.bait, poses[a.agent_id].uncertainty))
            if refs:
                a = refs[0]; pose = poses[a.agent_id]
                angle = a.direction-pose.heading
                c, s = math.cos(angle), math.sin(angle)
                x, y = policy.site['goal']-pose.position
                projected = (a.x+c*x-s*y, a.y+s*x+c*y)
                candidate = min(self.true_sites, key=lambda q: math.dist(projected, q['goal']))
                self.mapping_error = math.dist(projected, candidate['goal'])
                if self.mapping_error <= 5.:
                    self.actual_site = candidate
                    baits = [a for a in env.agents if roles.get(a.agent_id) in ('bait', 'replacement_bait', 'retired_bait')
                             and math.dist((a.x, a.y), candidate['goal']) <= 3.]
        if baits:
            if self.first_bait is None: self.first_bait = now
            self.last_bait = now
            self.current_gap = 0.
        elif self.first_bait is not None:
            self.gap += .1; self.current_gap += .1
            self.longest_gap = max(self.longest_gap, self.current_gap)
        near, rear, held = 0, 0, 0
        for pred in env.predators:
            if pred not in self.pred_ids: self.pred_ids[pred] = len(self.pred_ids)
            pid = self.pred_ids[pred]
            close = bool(baits) and min(math.hypot(pred.x-a.x, pred.y-a.y) for a in baits) <= 40.
            if close:
                near += 1
                self.near_since.setdefault(pid, now)
                if now-self.near_since[pid] >= 30.:
                    held += 1
                    self.ever_held.add(pid)
            else: self.near_since.pop(pid, None)
            if self.actual_site is not None:
                site = self.actual_site
                rear += (math.dist((pred.x, pred.y), site['other_mouth']) <= 40.
                         and (pred.x-site['mouth'][0])*site['inward'][0]
                         + (pred.y-site['mouth'][1])*site['inward'][1] >= site['overlap']-5.)
        self.max_near, self.max_held, self.max_rear = max(self.max_near, near), max(self.max_held, held), max(self.max_rear, rear)
        self.rear_ticks += bool(rear)
        row = dict(tick=tick, time=now, agents=len(env.agents), predators=len(env.predators),
                   score=env.score, near_bait=near, held30=held, rear=rear, bait_present=bool(baits))
        if tick % 10 == 0: self.history.append(row)
        return row

    def trajectory(self, env, policy, tick):
        births = {}
        for a in env.agents:
            if a.agent_id not in self.agents_catalog:
                data = dict(size=a.size, color=a.color, max_energy=a.max_energy,
                            hearing_radius=a.hearing_radius, vision_radius=a.vision_radius,
                            cone_angle=a.cone_angle, speed=a.speed, sprint_speed=a.sprint_speed)
                self.agents_catalog[a.agent_id] = data
                births[a.agent_id] = data
        return dict(tick=tick, time=env.time, births=births,
                    # Compact but lossless physical state at EVERY native tick.
                    agents=[[a.agent_id, a.x, a.y, a.direction, a.energy, a.age,
                             ROLE_IDS.get(policy.roles.get(a.agent_id), -1)] for a in env.agents],
                    predators=[[self.pred_ids[p], p.x, p.y, p.direction, p.energy] for p in env.predators])


def run_case(task):
    seed, seconds, output, record = task
    folder = Path(output)/f'{seed:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    cpu_start = time.process_time()
    trajectory = None
    result = dict(seed=seed, baseline_commit=BASELINE_COMMIT, horizon=seconds, policy_rng_seed=0)
    try:
        from src.core import SimulationCore
        from models.entrapment.entrapment_policy import EntrapmentPolicy
        core = SimulationCore(seed=seed)
        env = core.env
        policy = EntrapmentPolicy(seed=0)  # Independent of the world-generation seed.
        observer = Observer(env)
        write_json(folder/'static.json', dict(observer.static, role_ids=ROLE_IDS,
                    trajectory_format='agents: id,x,y,heading,energy,age,role; predators: id,x,y,heading,energy'))
        if record: trajectory = gzip.open(folder/'trajectory.jsonl.gz', 'wt', compresslevel=1)
        states = [env.get_agent_state(a.agent_id) for a in env.agents]
        limit = round(seconds/core.dt)
        last_progress = 0.
        for tick in range(limit+1):
            terminal = not states or tick == limit
            actions = [] if terminal else policy(states, env.time)
            last = observer.measure(env, policy, tick)
            if trajectory is not None:
                trajectory.write(json.dumps(clean(observer.trajectory(env, policy, tick)), separators=(',', ':'), allow_nan=False)+'\n')
            wall = time.monotonic()
            if wall-last_progress > 10.:
                write_json(folder/'progress.json', dict(last, elapsed_seconds=wall-start))
                last_progress = wall
            if terminal: break
            state = core.step(actions)
            states = state['observations']
        write_json(folder/'events.json', policy.events)
        write_json(folder/'history.json', observer.history)
        result.update(status='survived_horizon' if env.agents else 'extinct',
            sim_time=env.time, frames=tick+1, score=env.score, final_agents=len(env.agents),
            peak_agents=observer.peak, total_agents=len(observer.agents_seen),
            predators=len(env.predators), true_map_eligible_sites=len(observer.true_sites),
            trap_discovered=policy.metrics['site_discoveries'] > 0,
            first_trap_time=next((e['time'] for e in policy.events if e['kind']=='our_trap_discovered'), None),
            first_physical_bait_time=observer.first_bait, last_physical_bait_time=observer.last_bait,
            bait_gap_seconds=observer.gap, longest_bait_gap_seconds=observer.longest_gap,
            maximum_predators_near_bait=observer.max_near,
            maximum_predators_held30=observer.max_held, distinct_predators_held30=len(observer.ever_held),
            final_predators_held30=last['held30'], rear_occupied_seconds=observer.rear_ticks*.1,
            maximum_predators_at_rear=observer.max_rear,
            policy_metrics=policy.metrics)
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    finally:
        if trajectory is not None: trajectory.close()
    result.update(wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu_start)
    write_json(folder/'result.json', result)
    return result


def aggregate(cases):
    complete = [r for r in cases if r['status'] != 'error']
    baited = [r for r in complete if r['first_physical_bait_time'] is not None]
    def dist(key):
        values = sorted(r[key] for r in complete)
        if not values: return None
        return dict(mean=statistics.mean(values), median=statistics.median(values), minimum=values[0], maximum=values[-1])
    assigned = sum(r['policy_metrics']['guide_assignments'] for r in complete)
    delivered = sum(r['policy_metrics']['delivery_arrivals'] for r in complete)
    return dict(cases=len(cases), completed_games=len(complete), errors=len(cases)-len(complete),
        survived_horizon=sum(r['status']=='survived_horizon' for r in complete),
        maps_with_true_eligible_sites=sum(r['true_map_eligible_sites'] > 0 for r in complete),
        maps_with_discovered_traps=sum(r['trap_discovered'] for r in complete),
        maps_with_physical_bait=len(baited),
        baited_games_without_subsequent_gap=sum(r['bait_gap_seconds'] < .05 for r in baited),
        baited_games_with_rear_occupation=sum(r['rear_occupied_seconds'] > 0 for r in baited),
        guide_assignments=assigned, delivery_arrivals=delivered,
        delivery_arrival_fraction=None if not assigned else delivered/assigned,
        survival_seconds=dist('sim_time'), score=dist('score'),
        maximum_predators_held30=dist('maximum_predators_held30'),
        per_case_wall_seconds=dist('wall_seconds'), per_case_cpu_seconds=dist('cpu_seconds'),
        note='Arrival fraction counts assignments, not independent captures. Held30 means within 40 of physically verified bait for 30s, not permanent retention. Every seed counts; errors remain explicit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3000.)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--no-trajectories', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards: parser.error('Invalid shard index')
    manifest = verify_sources()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed_start+i for i in range(args.count) if i % args.shards == args.shard_index]
    write_json(args.out/'protocol.json', dict(baseline_commit=BASELINE_COMMIT, seeds=seeds,
        seconds=args.seconds, workers=args.workers, shard_index=args.shard_index, shards=args.shards,
        source_hashes=manifest['sources'], policy_rng_seed=0,
        env_threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        traces='Every-tick lossless agent/predator trajectories; not a full fruit/tree/observation replay.' if not args.no_trajectories else 'Per-game results, events, one-second time series only.'))
    cases, pending = [], []
    for seed in seeds:
        result = args.out/f'{seed:06d}'/'result.json'
        if result.exists(): cases.append(json.loads(result.read_text()))
        else: pending.append((seed, args.seconds, str(args.out), not args.no_trajectories))
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_case, task): task[0] for task in pending}
        for future in as_completed(futures):
            seed = futures[future]
            try: result = future.result()
            except Exception: result = dict(seed=seed, status='error', error=traceback.format_exc())
            cases.append(result)
            progress = dict(aggregate(cases), total_planned=len(seeds), elapsed_seconds=time.monotonic()-start)
            write_json(args.out/'aggregate.json', progress)
            print(json.dumps(dict(seed=seed, status=result['status'], sim_time=result.get('sim_time'),
                                 completed=len(cases), planned=len(seeds), elapsed_seconds=round(time.monotonic()-start, 2))), flush=True)
    write_json(args.out/'cases.json', sorted(cases, key=lambda r:r['seed']))
    write_json(args.out/'aggregate.json', dict(aggregate(cases), total_planned=len(seeds),
                                            elapsed_seconds=time.monotonic()-start, finished=True))


if __name__ == '__main__': main()
