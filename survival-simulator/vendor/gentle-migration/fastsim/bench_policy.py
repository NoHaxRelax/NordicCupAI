"""Wall-clock comparison for a full game: native policy vs Python policy.

Four configurations, all on the same fastsim engine and the same seed, so the games
are identical tick for tick (the native policy is bit-identical - see
fastsim/verify_evasion.py):

  native      run_policy(): decide + step entirely in C++, ONE Python call per game.
  native+prof the same, with per-phase nanosecond accounting turned on. The gap
              between this and `native` is the cost of the clock reads themselves.
  native/tick the C++ policy driven from Python one tick at a time: policy_act() then
              step(). Same decisions, same engine - the only difference from `native`
              is that state and actions cross into Python 30000 times. This is what
              the in-process interface costs.
  python      the Python OrchardEvasionPolicy driven the same way. The baseline.

    python fastsim/bench_policy.py --seed 1 --horizon 3000
    python fastsim/bench_policy.py --seed 1 --horizon 300 --skip python
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

import fastsim  # noqa: E402
from fastsim.fastpolicy import PolicySimulationCore  # noqa: E402
from models.notrap_config import defaults, orchard_kwargs  # noqa: E402
from models.orchard_evasion_policy import OrchardEvasionPolicy  # noqa: E402


def native(seed, horizon, cfg, predators, profile):
    sim = PolicySimulationCore(seed=seed, predators=predators)
    sim.policy_init(seed, cfg)
    sim.step([])
    t = time.perf_counter()
    steps, peak, ns_i, ns_p, ns_e = sim._engine.run_policy(float(horizon), float(horizon), profile)
    wall = time.perf_counter() - t
    out = dict(wall_s=round(wall, 3), steps=steps, peak_agents=peak,
               sim_time=round(sim.env.time, 1), score=round(sim.env.score, 4),
               us_per_tick=round(wall / max(steps, 1) * 1e6, 1))
    if profile:
        tot = ns_i + ns_p + ns_e
        out['phase_s'] = dict(interface=round(ns_i / 1e9, 3), policy=round(ns_p / 1e9, 3),
                              engine=round(ns_e / 1e9, 3))
        out['phase_pct'] = dict(interface=round(100 * ns_i / max(tot, 1), 2),
                                policy=round(100 * ns_p / max(tot, 1), 2),
                                engine=round(100 * ns_e / max(tot, 1), 2))
    return out


def per_tick(seed, horizon, cfg, predators, use_native):
    """One Python call per tick, as the current production loop does."""
    if use_native:
        sim = PolicySimulationCore(seed=seed, predators=predators)
        sim.policy_init(seed, cfg)
        # step() wants [(agent_id, ActionRequest)], so the packed tuples have to be
        # reboxed. That reboxing IS the per-tick interface, so it is timed with the
        # policy, not hidden.
        def decide():
            return [(aid, dict(move_distance=d, move_direction=dr, turn_angle=t, spawn_agent=sp))
                    for aid, d, dr, t, sp in sim.policy_act()]
    else:
        sim = fastsim.SimulationCore(seed=seed, predators=predators)
        pol = OrchardEvasionPolicy(seed=seed, **cfg)
        decide = None
    state = sim.step([])
    steps = 0
    t_policy = 0.
    peak = state['num_agents']
    t0 = time.perf_counter()
    while state['num_agents'] and state['sim_time'] < horizon:
        a = time.perf_counter()
        acts = decide() if use_native else pol(state['observations'], state['sim_time'])
        t_policy += time.perf_counter() - a
        state = sim.step(acts)
        steps += 1
        peak = max(peak, state['num_agents'])
    wall = time.perf_counter() - t0
    return dict(wall_s=round(wall, 3), steps=steps, peak_agents=peak,
                sim_time=round(state['sim_time'], 1), score=round(state['score'], 4),
                us_per_tick=round(wall / max(steps, 1) * 1e6, 1),
                policy_s=round(t_policy, 3),
                policy_pct=round(100 * t_policy / max(wall, 1e-9), 2),
                engine_and_glue_s=round(wall - t_policy, 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--horizon', type=float, default=3000)
    p.add_argument('--no-predators', dest='predators', action='store_false')
    p.add_argument('--skip', nargs='*', default=[])
    a = p.parse_args()
    cfg = orchard_kwargs(defaults())
    runs = [('native', lambda: native(a.seed, a.horizon, cfg, a.predators, False)),
            ('native+prof', lambda: native(a.seed, a.horizon, cfg, a.predators, True)),
            ('native/tick', lambda: per_tick(a.seed, a.horizon, cfg, a.predators, True)),
            ('python', lambda: per_tick(a.seed, a.horizon, cfg, a.predators, False))]
    results = {}
    for name, fn in runs:
        if name in a.skip:
            continue
        r = fn()
        results[name] = r
        print(json.dumps({name: r}), flush=True)
    if 'python' in results and 'native' in results:
        py, nat = results['python']['wall_s'], results['native']['wall_s']
        print(json.dumps(dict(speedup_end_to_end=round(py / nat, 2),
                              games_per_hour_native=round(3600 / nat, 1),
                              games_per_hour_python=round(3600 / py, 1))), flush=True)


if __name__ == '__main__':
    main()
