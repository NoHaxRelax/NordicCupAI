"""Diagnostic: fruit spawned vs eaten vs rotted per window; where rotted fruit was relative to agents/knowledge."""
import os, sys, pathlib, math, json, collections
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 1200
KW = json.loads(os.environ.get('ORCHARD_KW', '{}'))
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed, **KW)
W = 300.
stats = collections.defaultdict(lambda: collections.Counter())
orig_spawn = env.spawn_fruit; orig_remove = env.remove_fruit
def spawn_fruit(*a, **k):
    f = orig_spawn(*a, **k)
    if f is not None: stats[int(env.time//W)]['spawned'] += 1
    return f
def remove_fruit(fruit):
    w = stats[int(env.time//W)]
    eater = next((a for a in env.agents if math.dist((a.x, a.y), (fruit.x, fruit.y)) < a.size+fruit.radius), None)
    if eater is not None:
        w['eaten'] += 1; w['eaten_energy'] += fruit.energy
        if fruit.energy >= 59.9: w['eaten_ripe'] += 1
    else:
        w['rotted'] += 1
        dmin = min((math.dist((a.x, a.y), (fruit.x, fruit.y)) for a in env.agents), default=9e9)
        if dmin < 100: w['rot_near100'] += 1
        elif dmin < 250: w['rot_near250'] += 1
        # was it in some group's map?
        known = False
        for g in pol.groups.values():
            if not g.anchored: continue
            if any(math.dist(f.p, (fruit.x, fruit.y)) < 6 for f in g.fruits.values()): known = True; break
        if known: w['rot_known'] += 1
    return orig_remove(fruit)
env.spawn_fruit = spawn_fruit; env.remove_fruit = remove_fruit
state = sim.step([])
agent_ticks = collections.Counter()
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    w = int(state['sim_time']//W)
    stats[w]['agent_ticks'] += len(actions)
    for aid, _ in actions: stats[w]['mode_'+pol.decisions.get(aid, ('?',))[0]] += 1
    state = sim.step(actions)
print(f"seed {seed} kw {KW} survived {state['sim_time']:.0f}")
print(f"{'win':>5}{'agents':>7}{'spawn':>7}{'eaten':>7}{'ripe':>6}{'avgE':>6}{'rot':>6}{'rot<100':>8}{'rot<250':>8}{'rotKnown':>9}  modes")
for w in sorted(stats):
    s = stats[w]; n = s['agent_ticks']/(W*10)
    modes = {k[5:]: round(v/max(1, s['agent_ticks']), 2) for k, v in s.items() if k.startswith('mode_') and v/max(1, s['agent_ticks']) >= 0.03}
    print(f"{w*W:>5.0f}{n:>7.1f}{s['spawned']:>7}{s['eaten']:>7}{s['eaten_ripe']:>6}{s['eaten_energy']/max(1,s['eaten']):>6.0f}{s['rotted']:>6}{s['rot_near100']:>8}{s['rot_near250']:>8}{s['rot_known']:>9}  {modes}")
