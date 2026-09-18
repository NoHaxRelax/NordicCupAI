"""Diagnostic: when does an anchored agent's absolute position error jump, and why?"""
import os, sys, pathlib, math, collections
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 200
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed)
state = sim.step([]); prev_err = {}; shown = 0; jumps = collections.Counter(); persist = collections.Counter()
while state['num_agents'] and state['sim_time'] < horizon:
    obs_by = {o['agent_id']: o for o in state['observations']}
    actions = pol(state['observations'], state['sim_time'])
    truth = {a.agent_id: (a.x, a.y) for a in env.agents}
    for aid, m in pol.minds.items():
        g = pol.groups[m.group]
        if not g.anchored or aid not in truth: continue
        e = math.hypot(m.pose.p[0]-truth[aid][0], m.pose.p[1]-truth[aid][1])
        pe = prev_err.get(aid, 0.)
        if e > 5: persist[aid] += 1
        if e-pe > 1.0:
            o = obs_by[aid]; kinds = collections.Counter(x['type'] for x in o['observations'])
            la = m.last_action
            jumps['born' if la is None else ('moved' if la[0] > 0 else 'still')] += 1
            if shown < 14 and la is not None:
                shown += 1
                print(f"t={state['sim_time']:.1f} a{aid} err {pe:.2f}->{e:.2f} age={o['age']:.1f} act=(d={la[0]:.1f},dir={la[1]:.2f}) biome={la[3]} obs={dict(kinds)} marks={len(m.prev_marks)} group={m.group} truepos=({truth[aid][0]:.0f},{truth[aid][1]:.0f})")
        prev_err[aid] = e
    state = sim.step(actions)
print('jumps by state', dict(jumps))
print('agents with >5 error ticks:', sorted(persist.items(), key=lambda x: -x[1])[:10])
