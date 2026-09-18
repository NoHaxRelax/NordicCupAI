"""Decision-equivalence check: hash the full action stream of a deterministic native-engine run."""
import os, sys, json, hashlib, time
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, 'vendor/survival-simulator'); sys.path.insert(0, 'research/orchard'); sys.path.insert(0, '.')
import fastsim
from orchard import OrchardPolicy
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3; horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 400
kw = json.load(open('results/orchard/best-config.json'))
sim = fastsim.SimulationCore(seed=seed, predators=False); pol = OrchardPolicy(seed=seed, **kw)
state = sim.step([]); h = hashlib.sha256(); t0 = time.perf_counter(); n = 0
while state['num_agents'] and state['sim_time'] < horizon:
    actions = pol(state['observations'], state['sim_time'])
    for aid, a in actions:
        h.update(f"{aid}:{a.move_distance:.9g}:{a.move_direction:.9g}:{a.turn_angle:.9g}:{int(a.spawn_agent)};".encode())
    state = sim.step(actions); n += 1
print(f"seed {seed} steps {n} agents {state['num_agents']} score {state['score']:.3f} actions-sha {h.hexdigest()[:16]} wall {time.perf_counter()-t0:.1f}s")
