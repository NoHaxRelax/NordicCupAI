"""Does birth-phase dodging COMPOUND, or does it only re-roll locally?

exp_birth_timing.py showed that with 1.2 s of scheduling slack a seed-aware policy can
push the next predator spawn past a 90 s horizon 99% of the time, against 48% for blind
breeding. That is a local result and it does not settle the real question.

The worry: predator spawns are a Bernoulli process whose rate, p(t) = (1/P)*dt*t*1e-4
(_engine.cpp:1543), depends on sim time and predator count -- not on RNG phase. Re-phasing
the stream changes WHICH draws fire, not the long-run rate. So dodging once may simply
move the spawn just past the horizon and buy nothing over a whole game.

The counter-argument: repeatedly choosing the later branch is rejection sampling against
the spawn process, and selection does bias the realized outcome.

This settles it by counting predators actually spawned over a long run, same seeds, same
policy, differing only in whether births are phase-timed.

    PYTHONPATH=survival-simulator python mirror/exp_birth_game.py
"""
import statistics
import time

import numpy as np

from fastsim import _mirror

_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)

W, H = 1600, 1200
HORIZON = 12000         # ticks per game (1200 sim seconds)
WINDOW = 12             # candidate birth ticks
LOOK = 600              # lookahead used to score a candidate
COOLDOWN = 400          # ticks between phase-timed birth decisions
SEEDS = list(range(1, 21))
ENERGY_GATE = 115.0


def forage(state, birth_ids=()):
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


def ticks_to_predator(e, snap, state, delay, horizon):
    """Score a candidate birth delay: ticks until the next predator, capped at horizon."""
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


def play(seed, timed):
    """One game. `timed`=False breeds at the first opportunity after each cooldown;
    `timed`=True searches the window and breeds on the phase that delays predators most."""
    e = _mirror.Engine([seed], env_width=W, env_height=H,
                       starting_agents=5, starting_fruits=60, starting_trees=60)
    st = e.state()
    next_decision, tick, births, rollouts = 0, 0, 0, 0

    while tick < HORIZON and st["num_agents"] > 0:
        ids = ()
        if tick >= next_decision and eligible(st):
            if not timed:
                ids = {eligible(st)[0]}
                births += 1
                next_decision = tick + COOLDOWN
            else:
                snap = e.snapshot()
                best_t, best_d = -1, 0
                for d in range(WINDOW):
                    t_d, bred = ticks_to_predator(e, snap, st, d, LOOK)
                    rollouts += 1
                    if bred and t_d > best_t:
                        best_t, best_d = t_d, d
                e.restore(snap)
                if best_t < 0:                 # no candidate actually bred
                    next_decision = tick + COOLDOWN
                else:
                    # commit: hold for best_d ticks, then breed
                    for _ in range(best_d):
                        st = e.step(forage(st))
                        tick += 1
                        if st["num_agents"] == 0:
                            break
                    if st["num_agents"] == 0:
                        break
                    el = eligible(st)
                    ids = {el[0]} if el else ()
                    if ids:
                        births += 1
                    next_decision = tick + COOLDOWN
        st = e.step(forage(st, ids))
        tick += 1

    return dict(preds=len(e.predators()), score=e.info()["score"], ticks=tick,
                births=births, rollouts=rollouts, alive=st["num_agents"])


def main():
    t0 = time.perf_counter()
    blind, timed = [], []
    for seed in SEEDS:
        b = play(seed, False)
        t = play(seed, True)
        blind.append(b)
        timed.append(t)
        print(f"  seed {seed:>3}: blind preds={b['preds']:>3} score={b['score']:8.1f} "
              f"ticks={b['ticks']:>6} | timed preds={t['preds']:>3} "
              f"score={t['score']:8.1f} ticks={t['ticks']:>6}", flush=True)

    def col(rows, k):
        return [r[k] for r in rows]

    print(f"\n{'':22}{'blind':>10}{'timed':>10}{'delta':>10}")
    for k in ("preds", "score", "ticks", "births"):
        b, t = statistics.mean(col(blind, k)), statistics.mean(col(timed, k))
        print(f"  mean {k:<16}{b:10.1f}{t:10.1f}{t-b:+10.1f}")

    pd = [t - b for b, t in zip(col(blind, "preds"), col(timed, "preds"))]
    sd = [t - b for b, t in zip(col(blind, "score"), col(timed, "score"))]
    fewer = sum(1 for d in pd if d < 0)
    print(f"\n  paired predator delta : mean {statistics.mean(pd):+.2f}, "
          f"median {statistics.median(pd):+.1f}, fewer in {fewer}/{len(pd)} seeds")
    print(f"  paired score delta    : mean {statistics.mean(sd):+.1f}, "
          f"median {statistics.median(sd):+.1f}, better in "
          f"{sum(1 for d in sd if d > 0)}/{len(sd)} seeds")
    print(f"\n  rollouts spent (timed): {sum(col(timed,'rollouts'))}")
    print(f"  wall clock: {time.perf_counter()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
