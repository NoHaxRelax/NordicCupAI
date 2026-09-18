"""Diagnostic: why is each fruit eaten at the energy it has? (claimed/unclaimed, known age, hunger)."""
import os, sys, pathlib, math, json, collections
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy, INF
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 900
KW = json.loads(os.environ.get('ORCHARD_KW', '{}'))
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed, **KW)
reasons = collections.Counter(); energy_by = collections.defaultdict(float)
orig_remove = env.remove_fruit
def remove_fruit(fruit):
    eater = next((a for a in env.agents if math.dist((a.x, a.y), (fruit.x, fruit.y)) < a.size+fruit.radius), None)
    if eater is not None:
        m = pol.minds.get(eater.agent_id)
        why = 'no_mind'
        if m is not None:
            g = pol.groups[m.group]
            # find memory entry closest to this fruit in the group frame is not possible without the frame;
            # instead classify by the eater's claim and its memory record
            fm = g.fruits.get(m.fruit) if m.fruit is not None else None
            if fm is None:
                why = 'unclaimed_touch'
            else:
                s_energy = eater.energy
                if fm.born_lo == -INF: why = 'claimed_unknown_age'
                elif fm.born_hi-fm.born_lo > pol.P['wait_tol']: why = 'claimed_uncertain_age'
                elif env.time >= fm.born_lo+pol.P['ripen_wait']: why = 'claimed_ripe_rule'
                elif eater.age > eater.max_age: why = 'claimed_old'
                else: why = 'claimed_hungry'
        ripe = fruit.energy >= 59.9
        reasons[(why, 'ripe' if ripe else 'unripe')] += 1; energy_by[why] += fruit.energy
    return orig_remove(fruit)
env.remove_fruit = remove_fruit
state = sim.step([])
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time']); state = sim.step(actions)
tot = sum(reasons.values())
for (why, r), n in sorted(reasons.items(), key=lambda x: -x[1]): print(f"{why:<24}{r:<7}{n:>6}  {100*n/tot:5.1f}%")
