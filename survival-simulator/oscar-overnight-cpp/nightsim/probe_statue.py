"""NaN-statue predator trap probe (unmodified Python engine).
Statue: two turns of 1e308 -> heading inf; afterwards the statue sends NO action (so it never moves/teleports).
Predators observe rel_dir = wrap(atan2 - inf) = NaN -> chase test fails; at d >= 90 the pivot branch uses sign(NaN) ->
NaN move direction -> predator clamped to (W-10,H-10) with NaN heading forever."""
import sys, math, json
sys.path.insert(0, 'vendor/survival-simulator')
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
from src.elements.predator import Predator
X = 1e308
def act(aid, md=0., mdir=0., turn=0.): return (aid, ActionRequest(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=False))
print('json 1e999 ->', json.loads('{"a": 1e999}')['a'], ActionRequest(**json.loads('{"agent_id":0,"move_distance":0,"move_direction":1e999,"turn_angle":0,"spawn_agent":false}')).move_direction)
for D in (70, 100, 150, 220):
    for seed in (1, 2, 3):
        sim = SimulationCore(seed=seed); env = sim.env
        st = env.agents[0]; sid = st.agent_id
        for a in list(env.agents[1:]): env.kill_agent(a)
        st.energy = 400
        sim.step([act(sid, turn=X)]); sim.step([act(sid, turn=X)])
        # predator placed D ahead of the statue along +x-ish free direction, facing it
        placed = None
        for k in range(36):
            b = k * math.pi / 18; px, py = st.x + D * math.cos(b), st.y + D * math.sin(b)
            if env._is_position_free(px - 12, py - 12, 24, 24):
                p = Predator(px, py, rng=env.rng); p.direction = b + math.pi; p.resting = False; p.energy = 150
                env.predators.append(p); env._update_predator_grid(); placed = p; break
        if not placed: print(D, seed, 'no spot'); continue
        for t in range(300):
            sim.step([])
            if sid not in env.agents_dict: break
        print(f"D={D} seed={seed} statue_dir={st.direction} statue_alive={sid in env.agents_dict} pos=({st.x:.0f},{st.y:.0f}) "
              f"pred=({placed.x:.1f},{placed.y:.1f}) pdir={placed.direction} E={placed.energy:.0f} rest={placed.resting}")
# robustness: statue + many agents + predators, 3000 ticks, no exceptions?
sim = SimulationCore(seed=9); env = sim.env
ids = [a.agent_id for a in env.agents]; st = ids[0]
sim.step([act(st, turn=X)]); sim.step([act(st, turn=X)])
for k in range(8): env.spawn_predator()
err = None
try:
    for t in range(1500):
        sim.step([act(i, md=5, turn=0.05) for i in ids[1:] if i in env.agents_dict])
except Exception as e: err = repr(e)
print('stress', err, 'preds', [(round(p.x), round(p.y), p.direction) for p in env.predators][:10], 'score', round(env.score, 2))
