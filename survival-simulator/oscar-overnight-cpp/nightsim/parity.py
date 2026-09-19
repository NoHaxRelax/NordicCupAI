"""Parity check: the Python OrchardPolicy (research/orchard/orchard.py) vs the native nightsim policy, in lockstep on one
nightsim engine. usage: parity.py --seeds 1-8 --horizon 600 [--kw cfg.json:label]
Each tick: native decisions via policy_act() (no step), Python decisions via the policy; compare (aid, dist, dir, turn,
spawn) with tolerance; step the engine with the Python actions. Reports the first divergent tick per seed."""
import argparse, json, math, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'research' / 'orchard')); sys.path.insert(0, str(ROOT / 'vendor' / 'survival-simulator'))
import nightsim
from orchard import OrchardPolicy
from nightsim.run import parse_seeds

def one(seed, horizon, kw):
    sim = nightsim.SimulationCore(seed=seed, predators=False); eng = sim._engine
    state = sim.step([]); eng.pop_events()
    eng.policy_init(nightsim.seed_key(seed), kw)
    pol = OrchardPolicy(seed=seed, **kw)
    steps = 0; first_div = None
    while state['num_agents'] and state['sim_time'] < horizon:
        nat = {a: (d, dr, t, bool(sp)) for (a, d, dr, t, sp) in eng.policy_act()}
        py = pol(state['observations'], state['sim_time'])
        pyd = {aid: (act.move_distance, act.move_direction, act.turn_angle, bool(act.spawn_agent)) for aid, act in py}
        if first_div is None:
            for aid, pv in pyd.items():
                nv = nat.get(aid)
                if nv is None or any(abs(x - y) > 1e-6 for x, y in zip(pv[:3], nv[:3])) or pv[3] != nv[3]:
                    first_div = (round(state['sim_time'], 1), aid, pv, nv); break
            if first_div is None and set(nat) != set(pyd): first_div = (round(state['sim_time'], 1), 'ids', sorted(nat)[:5], sorted(pyd)[:5])
        state = sim.step(py); steps += 1
    return dict(seed=seed, steps=steps, surv=round(state['sim_time'], 1), score=round(state['score'], 3), first_div=first_div)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', nargs='+', required=True); ap.add_argument('--horizon', type=float, default=600.); ap.add_argument('--kw', default='')
    a = ap.parse_args()
    kw = {}
    if a.kw:
        f, lab = a.kw.split(':'); kw = json.load(open(f))[lab]
    for s in parse_seeds(a.seeds):
        r = one(s, a.horizon, kw); print(json.dumps(r), flush=True)
