"""Duplicate-action mechanics in the UNMODIFIED reference engine."""
import sys, json, math
sys.path.insert(0, '/opt/nordiccup')
from src.core import SimulationCore
from DTOs import ActionRequest, StepResponse
def A(aid, md=0., mdir=0., turn=0., sp=False): return (aid, ActionRequest(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=sp))
def strict(state):
    b = StepResponse(game_status='running', score=state['score'], sim_time=state['sim_time'], n_agents=state['num_agents'], agent_status=state['observations']).model_dump()
    json.dumps(b, allow_nan=False)

# 1. does the engine execute N duplicates, and what do they cost?
print('--- 1. duplicates per tick (speed/sprint/energy) ---')
for N, dist in ((1, 20.), (2, 10.), (10, 20.), (20, 10.), (200, 10.)):
    sim = SimulationCore(seed=11); env = sim.env; a = env.agents[0]
    aid = a.agent_id; a.x, a.y = 200., 600.; a.direction = 0.; e0, x0 = a.energy, a.x
    st = sim.step([A(aid, dist, 0., 0.)] * N); strict(st)
    a = env.agents_dict[aid]
    print(f'N={N:4d} d={dist:5.1f} speed={a.speed} sprint={a.sprint_speed}: moved {a.x-x0:7.2f}px  energy {e0-a.energy:8.3f}  per px {(e0-a.energy)/max(a.x-x0,1e-9):.4f}')

# 2. negative energy: does the agent survive the tick, and does score flip when eaten?
print('--- 2. score on death with negative energy (needs the list-mutation skip) ---')
from src.elements.predator import Predator
for drain in (0, 150, 2000):
    sim = SimulationCore(seed=12); env = sim.env
    victims = env.agents[:2]
    for a in list(env.agents[2:]): env.kill_agent(a)
    farm, doomed = victims[0], victims[1]
    farm.x, farm.y = 800., 600.; doomed.x, doomed.y = 805., 600.
    p = Predator(810., 600., rng=env.rng); p.resting = False; p.energy = 150.
    env.predators.append(p); env._update_predator_grid()
    s0 = env.score
    if drain: sim.step([A(farm.agent_id, 0., 0., math.pi)] * drain)
    fe = env.agents_dict.get(farm.agent_id).energy if farm.agent_id in env.agents_dict else None
    for _ in range(3): st = sim.step([]); strict(st)
    print(f'drain={drain:5d}: farm energy after drain {fe}, alive_after={farm.agent_id in env.agents_dict}, score {s0:.2f} -> {env.score:.2f}, predator energy {p.energy:.1f} resting={p.resting}')
