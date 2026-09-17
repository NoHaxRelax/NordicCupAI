"""Compare the observation-only estimator with the oracle on generated maps.

    ../.venv/bin/python scripts/trapper/check_estimator.py --seeds 1 2 --seconds 300

Reports when each agent became absolutely localized, position/heading errors
over time, rectangles recovered vs. true, and predator track errors.
"""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core import SimulationCore                                # noqa: E402
from models.trapper.society_base import SocietyPolicy              # noqa: E402
from models.trapper.oracle import OracleWorld                      # noqa: E402
from models.trapper.estimator import EstimatedWorld                # noqa: E402
from models.trapper.geometry import dist, wrap                     # noqa: E402


def run(seed, seconds, verbose):
    sim = SimulationCore(seed=seed)
    env = sim.env
    pol = SocietyPolicy(seed=seed)
    oracle = OracleWorld(env)
    est = EstimatedWorld()
    state = sim.step([])
    first_abs = None
    pos_err = []
    head_err = []
    pred_err = []
    n_true_rects = len(env.obstacles) - 4
    while state['num_agents'] and state['sim_time'] < seconds:
        obs = state['observations']
        ew = est.update(obs, state['sim_time'])
        ow = oracle.update(obs)
        f = ew._frame
        if f is not None and f.absolute and first_abs is None:
            first_abs = round(state['sim_time'], 1)
        if f is not None and f.absolute:
            for aid, a in ew.agents.items():
                o = ow.agents.get(aid)
                if o is None:
                    continue
                pos_err.append(dist(a.p, o.p))
                head_err.append(abs(wrap(a.heading - o.heading)))
            for p in ew.predators:
                if p.fresh:
                    best = min((dist(p.p, q.p) for q in ow.predators), default=None)
                    if best is not None:
                        pred_err.append(best)
        actions = pol(obs, state['sim_time'])
        est.note_actions(actions)
        state = sim.step(actions)
        if verbose and round(state['sim_time'], 1) % 30 == 0:
            f = ew._frame
            print(f"t={state['sim_time']:.0f} frames={len(est.frames)} abs={f.absolute if f else None} members={len(f.members) if f else 0} "
                  f"rects={len(f.rects) if f else 0}/{n_true_rects} pos_err(last)={pos_err[-1] if pos_err else None} metrics={est.metrics}")
    f = ew._frame
    def q(v, k):
        v = sorted(v)
        return round(v[int(len(v) * k)], 2) if v else None
    return dict(seed=seed, first_absolute=first_abs, frames=len(est.frames), members=len(f.members) if f else 0,
                rects=len(f.rects) if f else 0, true_rects=n_true_rects,
                pos_err_median=q(pos_err, .5), pos_err_p95=q(pos_err, .95), pos_err_max=round(max(pos_err), 2) if pos_err else None,
                head_err_p95=q(head_err, .95), pred_err_median=q(pred_err, .5), pred_err_p95=q(pred_err, .95), metrics=est.metrics)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    for seed in a.seeds:
        print(run(seed, a.seconds, a.verbose))
