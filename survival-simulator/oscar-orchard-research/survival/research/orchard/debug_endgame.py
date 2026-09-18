"""Diagnostic (engine truth, never fed back): why does the colony die with trees left?
Every `every` seconds after `start`, per agent: mode, energy, age, true distance to the nearest live
tree and to the nearest fruit, and whether the agent's family map knows a live tree (and how far it is)."""
import os, sys, pathlib, math, json, collections
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
start = float(sys.argv[2]) if len(sys.argv) > 2 else 1800
every = float(sys.argv[3]) if len(sys.argv) > 3 else 50
KW = json.loads(os.environ.get('ORCHARD_KW', '{}'))
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed, **KW)
state = sim.step([]); nxt = start; tally = collections.Counter()
while state['num_agents'] and state['sim_time'] < 3000:
    actions = pol(state['observations'], state['sim_time'])
    if state['sim_time'] >= nxt-1e-6:
        nxt += every; t = state['sim_time']
        live = [(tr.x, tr.y, tr.age) for tr in env.trees]; fruits = [(f.x, f.y, f.energy) for f in env.fruits]
        print(f"== t={t:.0f} agents={len(env.agents)} trees={len(live)} fruits={len(fruits)}")
        for a in sorted(env.agents, key=lambda a: a.agent_id):
            m = pol.minds.get(a.agent_id); g = pol.groups[m.group] if m else None
            dtree = min((math.dist((a.x, a.y), (x, y)) for x, y, _ in live), default=None)
            dfruit = min((math.dist((a.x, a.y), (x, y)) for x, y, _ in fruits), default=None)
            known = None
            if g is not None and g.anchored:
                kl = [math.dist((a.x, a.y), tr.p) for tr in g.trees.values() if not tr.dead]
                known = min(kl) if kl else None
            mode = pol.decisions.get(a.agent_id, ('?',))[0]
            print(f"  a{a.agent_id} {mode:<8} e={a.energy:5.0f} age={a.age:4.0f} vis={a.vision_radius:.0f}/{a.cone_angle:.2f} "
                  f"trueTree={dtree and round(dtree)} trueFruit={dfruit and round(dfruit)} knownLiveTree={known and round(known)} groupLiveTrees={sum(1 for tr in g.trees.values() if not tr.dead) if g else '-'}")
            tally[mode] += 1
    state = sim.step(actions)
print(f"died at {state['sim_time']:.0f}; late-phase mode ticks {dict(tally)}")
