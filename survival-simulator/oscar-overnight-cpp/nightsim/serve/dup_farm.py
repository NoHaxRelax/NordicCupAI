"""Use 2: negative-energy harvest. doomed (index i) dies -> farm (i+1) skips the energy<=0 check -> predator eats it."""
import sys, math, json
sys.path.insert(0, '/opt/nordiccup')
from src.core import SimulationCore
from DTOs import ActionRequest
from src.elements.predator import Predator
def A(aid, md=0., mdir=0., turn=0.): return (aid, ActionRequest(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=False))
for N in (0, 2000, 10000, 20000):
    sim = SimulationCore(seed=12); env = sim.env
    keep = env.agents[:2]
    for a in list(env.agents[2:]): env.kill_agent(a)
    doomed, farm = keep[0], keep[1]          # list order: doomed dies first, farm is skipped
    env.agents[:] = [doomed, farm]
    doomed.x, doomed.y = 700., 600.; doomed.energy = 0.005
    farm.x, farm.y = 800., 600.; farm.energy = 75.        # a newborn-sized tank
    p = Predator(802., 600., rng=env.rng); p.resting = False; p.energy = 150.
    env.predators.append(p); env._update_predator_grid()
    s0 = env.score; pe0 = p.energy
    st = sim.step([A(farm.agent_id, 0., 0., math.pi)] * N)
    fe = farm.energy
    print(f'N={N:6d}: farm energy {fe:9.2f} alive={farm.agent_id in env.agents_dict} | score {s0:.2f} -> {env.score:+.2f} (delta {env.score-s0:+.2f}) | predator {pe0:.1f} -> {p.energy:.1f} resting={p.resting}')
