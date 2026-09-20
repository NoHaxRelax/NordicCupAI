"""Lockstep check: models/orchard_evasion_policy.py vs its native port, one fastsim engine.

Every tick both policies see the same engine state and must return bit-identical
actions - (agent_id, move_distance, move_direction, turn_angle, spawn_agent) compared
as raw IEEE-754 doubles, not with a tolerance. The PYTHON actions drive the engine, so
a divergence cannot hide by steering the two runs into different states.

This is fastsim/verify_policy.py from survival-simulator/oscar-fastsim 51ca0680 with
the gap that mattered closed: that verifier only ever ran the no-predator fixture, so
the evasion branch it was checking did not exist and would not have been exercised
anyway. Here predators are ON by default and the run reports how many ticks actually
entered the flee and face branches - a run that never fled has verified nothing about
evasion, so `evasion_ticks` is part of the result, not a footnote.

    python fastsim/verify_evasion.py --seeds 1 2 3 --horizon 600
    python fastsim/verify_evasion.py --seeds 1 --horizon 3000 --no-predators
"""
import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
import time

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from fastsim.fastpolicy import PolicySimulationCore  # noqa: E402
from models.notrap_config import defaults, orchard_kwargs  # noqa: E402
from models.orchard_evasion_policy import OrchardEvasionPolicy  # noqa: E402


def bits(x):
    return struct.pack('<d', float(x))


def same(p, q):
    return (p[0] == q[0] and bits(p[1]) == bits(q[1]) and bits(p[2]) == bits(q[2])
            and bits(p[3]) == bits(q[3]) and bool(p[4]) is bool(q[4]))


def run(seed, horizon, cfg, predators, verbose=False):
    sim = PolicySimulationCore(seed=seed, predators=predators)
    sim.policy_init(seed, cfg)
    pol = OrchardEvasionPolicy(seed=seed, **cfg)
    state = sim.step([])
    h = hashlib.sha256()
    n = divergences = 0
    tpy = tnat = 0.
    first = None
    max_agents = 0
    while state['num_agents'] and state['sim_time'] < horizon:
        a = time.perf_counter(); py = pol(state['observations'], state['sim_time'])
        b = time.perf_counter(); nat = sim.policy_act(); c = time.perf_counter()
        tpy += b - a; tnat += c - b
        pyt = [(aid, act.move_distance, act.move_direction, act.turn_angle, act.spawn_agent)
               for aid, act in py]
        if len(pyt) != len(nat):
            divergences += 1
            if first is None:
                first = dict(step=n, t=round(state['sim_time'], 1), kind='length',
                             py=len(pyt), native=len(nat))
        else:
            bad = [(p, q) for p, q in zip(pyt, nat) if not same(p, q)]
            if bad:
                divergences += len(bad)
                if first is None:
                    first = dict(step=n, t=round(state['sim_time'], 1), kind='action',
                                 n_agents=len(pyt), py=repr(bad[0][0]), native=repr(bad[0][1]))
                if verbose:
                    print(f'  step {n} t={state["sim_time"]:.1f}: {len(bad)} of {len(pyt)} differ',
                          flush=True)
        for aid, d, dr, t, sp in pyt:
            h.update(f'{aid}:{d:.9g}:{dr:.9g}:{t:.9g}:{int(sp)};'.encode())
        state = sim.step(py)
        n += 1
        max_agents = max(max_agents, state['num_agents'])
    m = pol.metrics
    nm = sim.policy_metrics()
    return dict(seed=seed, predators=predators, ok=divergences == 0, divergences=divergences,
                first_divergence=first, steps=n, t=round(state['sim_time'], 1),
                alive=state['num_agents'], max_agents=max_agents, score=round(state['score'], 4),
                actions_sha=h.hexdigest()[:16],
                py_evasion=dict(flee=m['flee_ticks'], face=m['face_ticks'], sprint=m['sprint_ticks']),
                native_evasion=dict(flee=nm['flee_ticks'], face=nm['face_ticks'],
                                    sprint=nm['sprint_ticks']),
                evasion_ticks=m['flee_ticks'] + m['face_ticks'],
                # Deflection counters come from the Python side only. That is enough:
                # the actions are compared bit-for-bit, so if Python steered on a tick
                # and the two agreed, the native policy steered identically. What the
                # count is for is proving the branch RAN - a wall_mode>0 run with
                # steer_ticks==0 has verified nothing about deflection, which is
                # exactly how five dead wall_* parameters survived every check.
                steer_ticks=m['steer_ticks'],
                py_steer=dict(ticks=m['steer_ticks'], locked=m['steer_locked'],
                              flips=m['steer_flips'], aborts=m['steer_aborts'],
                              no_target=m['steer_no_target'], not_target=m['steer_not_target']),
                wall_mode=int(cfg.get('wall_mode', 0)),
                py_policy_ms=round(tpy / max(n, 1) * 1e3, 3),
                native_policy_ms=round(tnat / max(n, 1) * 1e3, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--horizon', type=float, default=600)
    p.add_argument('--no-predators', dest='predators', action='store_false')
    p.add_argument('--config', default=None, help='JSON file of OrchardEvasionPolicy keywords')
    # Checking the default config only is what let the deflection layer ship
    # unported: at wall_mode=0 every wall_* value is dead, so Python and native
    # agree no matter what the native side implements. Sweeping the modes is the
    # gate, not an option.
    p.add_argument('--wall-modes', type=int, nargs='+', default=None,
                   help='verify at each of these wall_mode values (e.g. 0 1 2)')
    # Any policy keyword, so a deploy gate can verify a mechanism at its ON value.
    # Checking defaults alone is what let five wall_* keys ship unimplemented: at the
    # default the feature is off, so the two policies agree no matter what native does.
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='override a policy keyword, e.g. --set pred_evade_closest=1')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()
    base = json.load(open(a.config)) if a.config else orchard_kwargs(defaults())
    for item in a.overrides:
        key, _, raw = item.partition('=')
        key = key.strip()
        if key not in base:
            p.error(f'--set {key}: not a policy keyword')
        base[key] = type(base[key])(float(raw)) if isinstance(base[key], (int, float)) else raw
    modes = a.wall_modes if a.wall_modes is not None else [int(base.get('wall_mode', 0))]
    bad = 0
    unexercised = []
    for mode in modes:
        cfg = dict(base, wall_mode=mode)
        evasion_seen = steer_seen = 0
        for s in a.seeds:
            r = run(s, a.horizon, cfg, a.predators, a.verbose)
            bad += not r['ok']
            evasion_seen += r['evasion_ticks']
            steer_seen += r['steer_ticks']
            print(json.dumps(r), flush=True)
        if a.predators and not evasion_seen:
            print(f'WARNING: wall_mode={mode}: no flee or face tick occurred; '
                  'the evasion branch was never exercised', file=sys.stderr)
        if mode and not steer_seen:
            unexercised.append(mode)
            print(f'WARNING: wall_mode={mode}: no steer tick occurred; deflection was '
                  'never exercised, so agreement here proves nothing about it', file=sys.stderr)
    # A deflection mode that never steered is not a pass. Treat it as a failure so a
    # deploy gate cannot be satisfied by a run that simply never reached the branch.
    if unexercised:
        print(f'FAIL: deflection never exercised for wall_mode={unexercised}', file=sys.stderr)
    sys.exit(1 if (bad or unexercised) else 0)


if __name__ == '__main__':
    main()
