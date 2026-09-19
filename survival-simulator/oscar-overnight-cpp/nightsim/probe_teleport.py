"""Probe engine-bug exploits in the UNMODIFIED Python engine (ground truth).
T1 overflow teleport: turn X then move with move_direction X -> direction+X = inf -> NaN position -> clamped to (W-5,H-5).
T2 predators cannot reach the corner prisoner.
T3 armed-tick observation safety (shapely Polygon) for several X.
T4 sprint-40 boundary entry.
T5 JSON/pydantic round trip of the values."""
import sys, json, math, random
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else 'vendor/survival-simulator')
import numpy as np
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

def act(aid, md=0., mdir=0., turn=0., spawn=False):
    return (aid, ActionRequest(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=spawn))

X = 1e308
# T5
raw = json.dumps({"actions": [{"agent_id": 0, "move_distance": 0.0, "move_direction": X, "turn_angle": -X, "spawn_agent": False}]})
a = ActionRequest(**json.loads(raw)["actions"][0]); print("T5 json", raw[:120], "->", a.move_direction, a.turn_angle)

for seed in [3, 4]:
    sim = SimulationCore(seed=seed)
    env = sim.env
    ag = env.agents[0]; aid = ag.agent_id
    print(f"\nseed {seed} agent {aid} start ({ag.x:.1f},{ag.y:.1f}) dir {ag.direction:.3f} E {ag.energy:.1f}")
    sim.step([act(aid, turn=X)])
    print("  armed dir", ag.direction, "E", round(ag.energy, 3))
    sim.step([act(aid, md=0., mdir=X, turn=-X)])
    print(f"  after teleport pos ({ag.x},{ag.y}) dir {ag.direction} E {ag.energy:.3f}")
    # T2: predator placed at nearest free point, awake, full energy
    from src.elements.predator import Predator
    p = Predator(1540, 1140, rng=env.rng); p2 = Predator(1555, 1100, rng=env.rng)
    for q in (p, p2): q.resting = False; q.energy = 200; env.predators.append(q)
    env._update_predator_grid()
    mind = 1e9
    for t in range(600):
        st = sim.step([act(aid)])
        if aid not in env.agents_dict: print("  PRISONER DIED at", round(env.time, 1), "E", ag.energy); break
        for q in (p, p2): mind = min(mind, math.hypot(q.x - ag.x, q.y - ag.y))
    print(f"  after 60 s: alive={aid in env.agents_dict} E={ag.energy:.1f} min predator dist={mind:.1f} preds at ({p.x:.1f},{p.y:.1f}) ({p2.x:.1f},{p2.y:.1f}) score {env.score:.2f}")
    # prisoner child
    ag.energy = 400
    sim.step([act(aid, spawn=True)])
    kids = [a for a in env.agents if a.agent_id != aid and abs(a.x - ag.x) < 1 and abs(a.y - ag.y) < 1]
    print("  child at prisoner position:", [(k.agent_id, k.x, k.y, round(k.energy)) for k in kids])

# T3: armed-tick observation for many X with objects in view, check wrap value
print("\nT3 wrapped angle for armed direction")
for Xc in [1e308, 9e307, 1.2e308, 1.5e308, 1.7e308, 5e307+1e292]:
    v = float((np.float64(0.3) - np.float64(Xc) + np.pi) % (2*np.pi) - np.pi)
    v2 = float((np.float64(-2.0) - np.float64(Xc) + np.pi) % (2*np.pi) - np.pi)
    print(f"  X={Xc:.3e} wrap(-X)={v:.4f} (obj angle 0.3) {v2:.4f} (obj -2.0) cos(X)={math.cos(Xc):.3f} sum_overflow={math.isinf(Xc+Xc)}")
# stress: many agents armed each tick on a crowded map, 300 ticks, see if anything throws
sim = SimulationCore(seed=7); env = sim.env
for k in range(20): env.spawn_agent(x=random.uniform(100, 1500), y=random.uniform(100, 1100))
errs = 0
for t in range(300):
    acts = []
    for a in list(env.agents):
        acts.append(act(a.agent_id, md=5, turn=(X if t % 2 == 0 else -X)))
    try: sim.step(acts)
    except Exception as e: errs += 1; print("  EXC", type(e).__name__, e); break
print("T3 stress armed/disarmed every tick, errors:", errs, "agents", len(env.agents))

# T4: sprint-40 boundary entry
sim = SimulationCore(seed=1); env = sim.env
ag = env.agents[0]; aid = ag.agent_id
ag.x, ag.y, ag.direction = 35.0, 600.0, math.pi; ag.sprint_speed = 40; ag.energy = 400
env._update_agent_grid()
sim.step([act(aid, md=40)])
print(f"\nT4 sprint40 west from x=35: now ({ag.x:.2f},{ag.y:.2f}) E {ag.energy:.1f}")
sim.step([act(aid, md=30, mdir=math.pi)])
print(f"   exit east 30: ({ag.x:.2f},{ag.y:.2f})")
