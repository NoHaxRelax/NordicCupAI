import sys, math
sys.path.insert(0, 'vendor/survival-simulator')
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest
X = 1e308
def act(aid, md=0., mdir=0., turn=0.): return (aid, ActionRequest(agent_id=aid, move_distance=md, move_direction=mdir, turn_angle=turn, spawn_agent=False))
for seed in (1, 2, 5):
    sim = SimulationCore(seed=seed); env = sim.env; ag = env.agents[0]; aid = ag.agent_id
    ag.sprint_speed = 33.; ag.energy = 400.
    sim.step([act(aid, turn=X)]); sim.step([act(aid, mdir=X, turn=-X)])
    print(seed, 'in', (ag.x, ag.y), 'dir', ag.direction)
    dy = math.sqrt(32.2**2 - 10.5**2)
    for k in range(5):
        sim.step([act(aid, md=32.2, mdir=math.atan2(-dy, 10.5))]); print('  slide', round(ag.x, 2), round(ag.y, 2), 'E', round(ag.energy, 1))
    sim.step([act(aid, md=31.2, mdir=math.pi)]); print('  exit', round(ag.x, 2), round(ag.y, 2), 'E', round(ag.energy, 1), 'pred-reachable (x<=1565?)', ag.x <= 1565)
