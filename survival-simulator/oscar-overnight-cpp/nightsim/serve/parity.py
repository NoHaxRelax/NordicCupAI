"""In-engine policy_act vs API path (state -> JSON wire -> policy_act_ext) on the same game; actions must match."""
import json, math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nightsim
from night_policy import NightPolicy
cfg = json.load(open(sys.argv[1])); seeds = [int(x) for x in sys.argv[2].split(',')]; T = float(sys.argv[3])
def same(a, b):
    return all(x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y)) for x, y in zip(a, b))
for seed in seeds:
    sim = nightsim.SimulationCore(seed=seed, predators=True); eng = sim._engine
    state = sim.step([]); eng.policy_init(nightsim.seed_key(seed), dict(cfg))
    ext = NightPolicy(cfg, seed=seed); ticks = bad = omitted = 0
    while state['observations'] and state['sim_time'] < T:
        a1 = eng.policy_act()
        wire = json.loads(json.dumps(state['observations']))   # allow_nan like the stdlib default
        a2 = ext.raw(wire, state['sim_time'])
        if len(a1) != len(a2) or not all(same(x, y) for x, y in zip(a1, a2)):
            bad += 1
            if bad <= 3: print('MISMATCH', seed, state['sim_time'], len(a1), len(a2))
        omitted += len(state['observations']) - len(a1)
        state = sim.step([(a, dict(agent_id=a, move_distance=d, move_direction=di, turn_angle=t, spawn_agent=sp)) for a, d, di, t, sp in a1])
        ticks += 1
    info = eng.info()
    print(f'seed {seed}: ticks {ticks} mismatches {bad} omitted-actions {omitted} t={info["time"]:.1f} score={info["score"]:.1f} statues={eng.dbg_tp()} frozen={sum(1 for p in eng.predators() if p[0]==1590. and p[1]==1190.)}', flush=True)
