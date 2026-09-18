import os, sys, pathlib, math
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1'); os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor'/'survival-simulator')); sys.path.insert(0, str(ROOT/'research'/'orchard'))
from src.core import SimulationCore
from src.elements.environment import Environment
Environment.spawn_predator = lambda self, *a, **k: None
from orchard import OrchardPolicy, wrap
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 200
sim = SimulationCore(seed=seed); env = sim.env; pol = OrchardPolicy(seed=seed)
# record group heading offset per group at anchoring/merge to detect inconsistent members
state = sim.step([]); flagged = set(); shown = 0
prev_groups = {}
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    truth = {a.agent_id: (a.x, a.y, a.direction) for a in env.agents}
    # per group: offset between estimated and true heading for each member; consistent group => all equal
    for g in pol.groups.values():
        offs = {aid: wrap(pol.minds[aid].pose.theta-truth[aid][2]) for aid in g.agents if aid in truth}
        if not offs: continue
        vals = list(offs.values()); ref = vals[0]
        bad = [aid for aid, o in offs.items() if abs(wrap(o-ref)) > 1e-6]
        if len(bad) > len(offs)/2: ref = offs[bad[0]]; bad = [aid for aid, o in offs.items() if abs(wrap(o-ref)) > 1e-6]
        for aid in bad:
            if aid in flagged: continue
            flagged.add(aid); m = pol.minds[aid]
            if shown < 15:
                shown += 1
                print(f"t={state['sim_time']:.1f} group{g.id} anchored={g.anchored} a{aid} age={m.born and round(state['sim_time']-m.born,1)} off={offs[aid]:.3f} ref={ref:.3f} diff={wrap(offs[aid]-ref):.3f} members={len(offs)}")
    state = sim.step(actions)
print('flagged', len(flagged))
