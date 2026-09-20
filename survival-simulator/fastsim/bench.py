"""Engine-only speed benchmark.

1. `record`: run the orchard policy on the native engine and save its actions.
2. `replay`: replay those actions on the native engine (and optionally the Python
   engine for the first N steps) and report ms/step. The native engine is
   deterministic, so the replay retraces the recorded run exactly.

    python fastsim/bench.py record --seed 1 --horizon 600 --out /tmp/acts.pkl
    python fastsim/bench.py replay --file /tmp/acts.pkl --python-steps 1000
"""
import os, sys, time, pickle, argparse, pathlib
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
# works both in this project (survival/vendor/survival-simulator/src) and in the team repo
# (survival-simulator/src next to fastsim/); the orchard policy comes from fastsim/policy/
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor' / 'survival-simulator'), str(HERE / 'policy')]
import fastsim  # noqa: E402
from src.utils.DTOs import ActionRequest  # noqa: E402


def record(seed, horizon, out, predators):
    import orchard_ref as orchard
    sim = fastsim.SimulationCore(seed=seed, predators=predators)
    pol = orchard.OrchardPolicy(seed=seed)
    st = sim.step([]); log = [[]]; tp = te = 0.0
    while st['num_agents'] and st['sim_time'] < horizon:
        a = time.perf_counter(); acts = pol(st['observations'], st['sim_time']); b = time.perf_counter()
        st = sim.step(acts); c = time.perf_counter(); tp += b - a; te += c - b
        log.append([(aid, act.move_distance, act.move_direction, act.turn_angle, act.spawn_agent) for aid, act in acts])
    pickle.dump(dict(seed=seed, predators=predators, actions=log, final_score=st['score'], final_time=st['sim_time']),
                open(out, 'wb'))
    n = len(log)
    print(f'recorded {n} steps to t={st["sim_time"]:.1f} alive={st["num_agents"]}; policy {tp/n*1e3:.2f} ms/step, '
          f'native engine {te/n*1e3:.3f} ms/step')


def replay(path, python_steps):
    d = pickle.load(open(path, 'rb'))
    steps = [[(aid, ActionRequest(agent_id=aid, move_distance=m, move_direction=md, turn_angle=t, spawn_agent=s))
              for aid, m, md, t, s in acts] for acts in d['actions']]
    t0 = time.perf_counter()
    sim = fastsim.SimulationCore(seed=d['seed'], predators=d['predators'])
    t1 = time.perf_counter()
    agents = 0
    for acts in steps:
        st = sim.step(acts); agents += st['num_agents']
    t2 = time.perf_counter()
    if not (st['score'] == d['final_score'] and st['sim_time'] == d['final_time']):
        # expected across CPUs: numpy's math kernels (and so the engine) can differ by platform
        print('note: replay diverged from the recording (recorded on another platform?)')
    fast_ms = (t2 - t1) / len(steps) * 1e3
    print(f'native: init {t1-t0:.3f}s, {len(steps)} steps, {fast_ms:.4f} ms/step, mean agents {agents/len(steps):.1f}')
    if python_steps:
        from src.core import SimulationCore as PySim
        from src.elements.environment import Environment
        if not d['predators']:
            Environment.spawn_predator = lambda self, *a, **k: None
        a = time.perf_counter(); py = PySim(seed=d['seed']); b = time.perf_counter()
        fs = fastsim.SimulationCore(seed=d['seed'], predators=d['predators'])
        n = min(python_steps, len(steps)); tpy = tf = 0.0
        for acts in steps[:n]:
            c = time.perf_counter(); py.step(acts); e = time.perf_counter(); fs.step(acts); f = time.perf_counter()
            tpy += e - c; tf += f - e
        print(f'python: init {b-a:.2f}s; first {n} steps python {tpy/n*1e3:.3f} ms/step vs native {tf/n*1e3:.4f} '
              f'ms/step -> {tpy/tf:.0f}x (Python runs are not set-order identical, timing only)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record'); r.add_argument('--seed', type=int, default=1); r.add_argument('--horizon', type=float, default=600)
    r.add_argument('--out', required=True); r.add_argument('--predators', action='store_true')
    q = sub.add_parser('replay'); q.add_argument('--file', required=True); q.add_argument('--python-steps', type=int, default=0)
    a = p.parse_args()
    if a.cmd == 'record':
        record(a.seed, a.horizon, a.out, a.predators)
    else:
        replay(a.file, a.python_steps)
