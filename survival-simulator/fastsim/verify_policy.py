"""Lockstep check: Python orchard policy vs the native (C++) port, on one fastsim engine.

Every step both policies see the same state; their actions must be bit-identical
(aid, move_distance, move_direction, turn_angle, spawn_agent). The Python actions
drive the engine. Also prints the action-stream hash in the format of
research/orchard/equiv.py for comparison with recorded hashes.

    python fastsim/verify_policy.py --seeds 3 --horizon 400
    python fastsim/verify_policy.py --seeds 1 2 3 --horizon 3000 --ref pre_opt --predators
"""
import os, sys, json, time, math, struct, hashlib, argparse, pathlib, importlib
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor' / 'survival-simulator'), str(HERE / 'policy')]
import fastsim  # noqa: E402

REFS = {'current': 'orchard_ref', 'pre_opt': 'orchard_pre_opt_ref'}


def bits(x):
    return struct.pack('<d', float(x))


def run(seed, horizon, ref, cfg, predators):
    mod = importlib.import_module(REFS[ref])
    sim = fastsim.SimulationCore(seed=seed, predators=predators)
    sim._engine.policy_init(fastsim.seed_key(seed), cfg)
    pol = mod.OrchardPolicy(seed=seed, **cfg)
    state = sim.step([])
    h = hashlib.sha256(); n = 0; tpy = tnat = 0.; max_agents = 0
    while state['num_agents'] and state['sim_time'] < horizon:
        a = time.perf_counter(); py = pol(state['observations'], state['sim_time'])
        b = time.perf_counter(); nat = sim._engine.policy_act(); c = time.perf_counter()
        tpy += b - a; tnat += c - b
        pyt = [(aid, act.move_distance, act.move_direction, act.turn_angle, act.spawn_agent) for aid, act in py]
        if len(pyt) != len(nat) or any(p[0] != q[0] or bits(p[1]) != bits(q[1]) or bits(p[2]) != bits(q[2])
                                       or bits(p[3]) != bits(q[3]) or bool(p[4]) != bool(q[4]) for p, q in zip(pyt, nat)):
            diff = next(((p, q) for p, q in zip(pyt, nat) if p != q or any(bits(x) != bits(y) for x, y in zip(p[1:4], q[1:4]))),
                        (len(pyt), len(nat)))
            return dict(seed=seed, ok=False, step=n, t=round(state['sim_time'], 1), diff=repr(diff))
        for aid, d, dr, t, sp in pyt:
            h.update(f"{aid}:{d:.9g}:{dr:.9g}:{t:.9g}:{int(sp)};".encode())
        state = sim.step(py); n += 1
        max_agents = max(max_agents, state['num_agents'])
    return dict(seed=seed, ok=True, steps=n, t=round(state['sim_time'], 1), alive=state['num_agents'],
                max_agents=max_agents, score=round(state['score'], 4), actions_sha=h.hexdigest()[:16],
                py_policy_ms=round(tpy / max(n, 1) * 1e3, 3), native_policy_ms=round(tnat / max(n, 1) * 1e3, 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[3])
    p.add_argument('--horizon', type=float, default=400)
    p.add_argument('--ref', choices=list(REFS), default='current')
    p.add_argument('--config', default=str(HERE / 'policy' / 'best-config.json'))
    p.add_argument('--predators', action='store_true')
    a = p.parse_args()
    cfg = json.load(open(a.config)) if a.config else {}
    bad = 0
    for s in a.seeds:
        r = run(s, a.horizon, a.ref, cfg, a.predators)
        bad += not r['ok']
        print(json.dumps(r), flush=True)
    sys.exit(1 if bad else 0)
