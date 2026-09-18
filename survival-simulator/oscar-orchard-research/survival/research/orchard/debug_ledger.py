"""Diagnostic: where does the species' energy go? Read-only engine accounting per category."""
import os, sys, pathlib, math, json
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 1200
KW = json.loads(os.environ.get('ORCHARD_KW', '{}'))
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed, **KW)
state = sim.step([])
L = dict(passive=0., walk=0., sprint=0., turn=0., births=0., old_drain=0., fruit_in=0., child_start=0.)
orig_remove = env.remove_fruit
def remove_fruit(fruit):
    if any(math.dist((a.x, a.y), (fruit.x, fruit.y)) < a.size+fruit.radius for a in env.agents): L['fruit_in'] += fruit.energy
    return orig_remove(fruit)
env.remove_fruit = remove_fruit
mode_ticks = {}
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    ag = {a.agent_id: a for a in env.agents}
    n_before = env._next_agent_id
    for aid, act in actions:
        a = ag[aid]; d = max(0., min(act.move_distance, a.sprint_speed))
        if a.energy < a.max_energy/5 and d > a.speed: d = a.speed
        if d <= a.speed: L['walk'] += d*0.05
        else: L['walk'] += a.speed*0.05; L['sprint'] += (d-a.speed)*0.5
        L['turn'] += min(math.pi, abs(act.turn_angle))/(2*math.pi)
        if a.age > a.max_age: L['old_drain'] += 0.01*a.age
        L['passive'] += 0.1
        mode = pol.decisions.get(aid, ('?',))[0]; mode_ticks[mode] = mode_ticks.get(mode, 0)+1
    state = sim.step(actions)
    nb = env._next_agent_id-n_before; L['births'] += 100.*nb; L['child_start'] += 75.*nb
tot = L['passive']+L['walk']+L['sprint']+L['turn']+L['births']+L['old_drain']
print(f"seed {seed} to t={state['sim_time']:.0f} alive={state['num_agents']} score={state['score']:.0f}")
print('income: fruit', round(L['fruit_in']), '+ founders 750 ; children received', round(L['child_start']))
for k in ('passive', 'walk', 'sprint', 'turn', 'births', 'old_drain'):
    print(f"  {k:<10}{L[k]:>9.0f}  {100*L[k]/tot:5.1f}%")
print('ticks by mode:', dict(sorted(mode_ticks.items(), key=lambda x: -x[1])))
