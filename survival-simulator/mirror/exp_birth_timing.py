"""Can a seed-aware policy TIME its births to delay predators?

exp_birth_phase.py showed a single birth moves the next predator spawn 98% of the time,
but only delays it 57% of the time -- a blind birth is a coin flip. That is the wrong
statistic for a policy with a synchronized model, because such a policy does not flip a
coin: it evaluates the branches and picks one. What matters is the value of CHOOSING.

This is the fair version of that question. Both arms breed the SAME NUMBER of times from
the SAME state, so population and energy effects cancel and only the timing differs:

  arm A (blind)  : breed on the first tick a parent is eligible
  arm B (seeded) : try each of the next WINDOW eligible ticks in the mirror and breed on
                   whichever pushes the next predator spawn furthest away

Reported: the extra predator-free time arm B buys over arm A.

    PYTHONPATH=survival-simulator python mirror/exp_birth_timing.py
"""
import statistics
import time

import numpy as np

from fastsim import _mirror

_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)

W, H = 1600, 1200
WINDOW = 12             # candidate birth ticks arm B may choose between
LOOK = 900              # ticks of lookahead used to score a candidate
DECISIONS = (300, 600, 900, 1200)
SEEDS = list(range(1, 41))
ENERGY_GATE = 115.0


def forage(state, birth_ids=()):
    """Deterministic, RNG-free: walk at the nearest fruit, else hold."""
    acts = []
    for a in state["observations"]:
        best_d, best_ang = None, 0.0
        for o in a["observations"]:
            if o["type"] == "Fruit" and (best_d is None or o["distance"] < best_d):
                best_d, best_ang = o["distance"], o["angle"]
        move, direction = (0.0, 0.0) if best_d is None else (min(a["speed"], best_d), best_ang)
        acts.append((int(a["agent_id"]), {
            "move_distance": move, "move_direction": direction,
            "turn_angle": 0.0, "spawn_agent": int(a["agent_id"]) in birth_ids,
        }))
    return acts


def eligible(state):
    return [int(a["agent_id"]) for a in state["observations"] if a["energy"] > ENERGY_GATE]


def score_plan(e, snap, state, delay, horizon):
    """Restore, hold for `delay` ticks, breed, then report ticks until the next predator.

    Returns (ticks_to_next_predator or horizon, actually_bred)."""
    e.restore(snap)
    st, n0, bred = state, len(e.predators()), False
    for t in range(horizon):
        ids = ()
        if not bred and t >= delay:
            el = eligible(st)
            if el:
                ids, bred = {el[0]}, True
        st = e.step(forage(st, ids))
        if len(e.predators()) > n0:
            return t, bred
    return horizon, bred


def main():
    t0 = time.perf_counter()
    gains, a_times, b_times, chosen = [], [], [], []
    rollouts = 0

    for seed in SEEDS:
        e = _mirror.Engine([seed], env_width=W, env_height=H,
                           starting_agents=5, starting_fruits=60, starting_trees=60)
        st, tick = e.state(), 0
        for T in DECISIONS:
            while tick < T:
                st = e.step(forage(st))
                tick += 1
            if not eligible(st):
                continue
            snap = e.snapshot()

            # arm A: breed at the first opportunity
            a_t, a_bred = score_plan(e, snap, st, 0, LOOK)
            rollouts += 1
            if not a_bred:
                e.restore(snap)
                continue

            # arm B: search the window, keep the delay that pushes the spawn furthest
            best_t, best_d = a_t, 0
            for d in range(1, WINDOW):
                t_d, bred = score_plan(e, snap, st, d, LOOK)
                rollouts += 1
                if bred and t_d > best_t:
                    best_t, best_d = t_d, d

            e.restore(snap)                     # back to the trunk
            a_times.append(a_t)
            b_times.append(best_t)
            gains.append(best_t - a_t)
            chosen.append(best_d)

    n = len(gains)
    print(f"decision points        : {n}   ({rollouts} rollouts of up to {LOOK} ticks)")
    print(f"window                 : {WINDOW} candidate birth ticks ({WINDOW/10:.1f} s of slack)")
    print(f"lookahead              : {LOOK} ticks ({LOOK/10:.0f} s)\n")
    print(f"arm A (blind)  mean ticks to next predator : {statistics.mean(a_times):7.1f} "
          f"({statistics.mean(a_times)/10:.1f} s)")
    print(f"arm B (seeded) mean ticks to next predator : {statistics.mean(b_times):7.1f} "
          f"({statistics.mean(b_times)/10:.1f} s)")
    print(f"\nmean gain              : {statistics.mean(gains):+.1f} ticks "
          f"({statistics.mean(gains)/10:+.1f} s)")
    print(f"median gain            : {statistics.median(gains):+.1f} ticks")
    improved = sum(1 for g in gains if g > 0)
    print(f"decisions improved     : {improved}/{n} ({100*improved/n:.0f}%)")
    print(f"never worse            : {all(g >= 0 for g in gains)}  (arm A is in arm B's set)")
    maxed = sum(1 for b in b_times if b >= LOOK)
    print(f"pushed past lookahead  : {maxed}/{n} ({100*maxed/n:.0f}%)  "
          f"vs {sum(1 for a in a_times if a >= LOOK)}/{n} blind")
    print(f"chosen delay (ticks)   : median {statistics.median(chosen):.0f}, "
          f"0 chosen {chosen.count(0)}/{n} times")
    print(f"\nwall clock: {time.perf_counter()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
