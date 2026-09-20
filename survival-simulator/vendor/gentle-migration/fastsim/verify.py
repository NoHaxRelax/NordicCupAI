"""Lockstep equivalence check: Python engine vs native engine.

The Python engine iterates sets of Agent/Fruit/Tree/Predator objects, whose hashes
come from memory addresses, so its own runs are not reproducible. Here those
classes get a creation-counter hash (the same stand-in the native engine uses);
nothing else is changed. Both engines then receive identical actions and every
step compares the returned state (observations bit for bit), the full world
state and the random generator state.

    python fastsim/verify.py --seeds 1 2 3 --horizon 300 --policy random
    python fastsim/verify.py --seeds 1 --horizon 3000 --policy orchard --no-predators
"""
import os, sys, math, time, json, random, argparse, itertools, pathlib
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# works both in this project (survival/vendor/survival-simulator/src) and in the team repo
# (survival-simulator/src next to fastsim/); the orchard policy comes from fastsim/policy/
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor' / 'survival-simulator'), str(HERE / 'policy')]

from src.core import SimulationCore as PySim  # noqa: E402
from src.elements.creature import Creature  # noqa: E402
from src.elements.fruit import Fruit  # noqa: E402
from src.elements.tree import Tree  # noqa: E402
from src.elements.environment import Environment  # noqa: E402
from src.utils.DTOs import ActionRequest  # noqa: E402
import fastsim  # noqa: E402

_serial = {'c': itertools.count(1)}


def _patch_hash(cls):
    orig = cls.__init__

    def init(self, *a, **k):
        self._serial = next(_serial['c'])
        orig(self, *a, **k)
    cls.__init__ = init
    cls.__hash__ = lambda self: self._serial


for _cls in (Creature, Fruit, Tree):
    _patch_hash(_cls)

_orig_spawn_predator = Environment.spawn_predator


class RandomPolicy:
    """Exercises every action path: sprinting, None direction, big turns, spawning."""

    def __init__(self, seed):
        self.r = random.Random(seed * 7919 + 1)

    def __call__(self, states, t):
        acts = []
        for s in states:
            r = self.r
            d = r.choice([0, r.uniform(0, 12), r.uniform(8, 45), -3, 10])
            direction = None if r.random() < 0.1 else r.uniform(-4, 4)
            turn = r.choice([0.0, r.uniform(-0.5, 0.5), r.uniform(-7, 7)])
            spawn = s['energy'] > 110 and r.random() < 0.05
            acts.append((s['agent_id'], ActionRequest(agent_id=s['agent_id'], move_distance=d, move_direction=direction if direction is not None else 0.0,
                                                      turn_angle=turn, spawn_agent=spawn)))
            if direction is None:
                acts[-1][1].__dict__['move_direction'] = None
        # an action for a dead/unknown id must be ignored by both engines
        acts.append((10 ** 6, ActionRequest(agent_id=10 ** 6, move_distance=5, move_direction=0, turn_angle=0, spawn_agent=False)))
        return acts


def world_py(env):
    return dict(
        agents=[(a.agent_id, a.x, a.y, a.direction, a.age, a.energy, a.max_energy, a.speed, a.sprint_speed,
                 a.hearing_radius, a.vision_radius, a.cone_angle, a.max_age) for a in env.agents],
        predators=[(p.x, p.y, p.direction, p.energy, p.resting) for p in env.predators],
        fruits=[(f.fruit_id, f.x, f.y, f.energy, f.age, f.radius) for f in env.fruits],
        trees=[(t.x, t.y, t.radius, t.age) for t in env.trees],
        time=env.time, score=env.score, next_id=env._next_agent_id)


def world_fast(sim):
    e = sim._engine
    info = e.info()
    return dict(
        agents=[(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[12], a[13]) for a in e.agents()],
        predators=[tuple(p) for p in e.predators()],
        fruits=[f[:6] for f in e.fruits()],
        trees=[tuple(t) for t in e.trees()],
        time=info['time'], score=info['score'], next_id=info['next_agent_id'])


def same(a, b):
    """Exact equality, NaN-safe, including int/float kind for top-level agent traits."""
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        return float(a) == float(b) and math.copysign(1, float(a)) == math.copysign(1, float(b)) or (a != a and b != b)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def first_diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a.keys()) != list(b.keys()):
            return f'{path}: keys {list(a.keys())} != {list(b.keys())}'
        for k in a:
            d = first_diff(a[k], b[k], f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f'{path}: len {len(a)} != {len(b)}\n  py={a}\n  fast={b}'
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if same(a, b) else f'{path}: py={a!r} fast={b!r}'


def type_kinds(state):
    """Python type names of the per-agent scalar fields (int vs float)."""
    return [[type(s[k]).__name__ for k in ('speed', 'sprint_speed', 'hearing_radius', 'vision_range', 'max_energy')]
            for s in state['observations']]


def run(seed, horizon, policy_name, predators, starting_predators, check_every):
    _serial['c'] = itertools.count(1)
    Environment.spawn_predator = _orig_spawn_predator if predators else (lambda self, *a, **k: None)
    t0 = time.perf_counter()
    py = PySim(seed=seed, starting_predators=starting_predators)
    t1 = time.perf_counter()
    fast = fastsim.SimulationCore(seed=seed, starting_predators=starting_predators, predators=predators)
    t2 = time.perf_counter()
    if policy_name == 'random':
        pol = RandomPolicy(seed)
    else:
        import orchard_ref as orchard
        pol = orchard.OrchardPolicy(seed=seed)

    d = first_diff(world_py(py.env), world_fast(fast))
    if d:
        return dict(seed=seed, ok=False, step=0, diff='init world ' + d)
    if py.env.rng.getstate() != fast._engine.rng_state():
        return dict(seed=seed, ok=False, step=0, diff='init rng state')

    actions = []
    steps = 0; tpy = tfast = 0.0; max_agents = 0; max_preds = 0; events = 0
    while True:
        a = time.perf_counter(); sp = py.step(actions); b = time.perf_counter(); sf = fast.step(actions); c = time.perf_counter()
        tpy += b - a; tfast += c - b; steps += 1
        events += len(fast.pop_events())
        max_agents = max(max_agents, sp['num_agents']); max_preds = max(max_preds, len(py.env.predators))
        d = first_diff(sp, sf)
        if not d and type_kinds(sp) != type_kinds(sf):
            d = f'int/float kinds {type_kinds(sp)} != {type_kinds(sf)}'
        if not d and steps % check_every == 0:
            d = first_diff(world_py(py.env), world_fast(fast))
            if not d and py.env.rng.getstate() != fast._engine.rng_state():
                d = 'rng state'
        if d:
            return dict(seed=seed, ok=False, step=steps, t=sp['sim_time'], diff=d)
        if not sp['num_agents'] or sp['sim_time'] >= horizon:
            break
        actions = pol(sp['observations'], sp['sim_time'])
    d = first_diff(world_py(py.env), world_fast(fast))
    if not d and py.env.rng.getstate() != fast._engine.rng_state():
        d = 'rng state'
    return dict(seed=seed, ok=d is None, diff=d, steps=steps, t=round(sp['sim_time'], 1), alive=sp['num_agents'],
                max_agents=max_agents, max_predators=max_preds, events=events, score=sp['score'],
                init_py_s=round(t1 - t0, 2), init_fast_s=round(t2 - t1, 3),
                step_py_ms=round(tpy / steps * 1e3, 3), step_fast_ms=round(tfast / steps * 1e3, 4),
                speedup=round(tpy / max(tfast, 1e-12), 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[1])
    p.add_argument('--horizon', type=float, default=200)
    p.add_argument('--policy', choices=['random', 'orchard'], default='random')
    p.add_argument('--no-predators', action='store_true')
    p.add_argument('--starting-predators', type=int, default=0)
    p.add_argument('--check-every', type=int, default=1, help='compare full world + RNG every N steps')
    a = p.parse_args()
    bad = 0
    for s in a.seeds:
        r = run(s, a.horizon, a.policy, not a.no_predators, a.starting_predators, a.check_every)
        bad += not r['ok']
        print(json.dumps(r), flush=True)
    sys.exit(1 if bad else 0)
