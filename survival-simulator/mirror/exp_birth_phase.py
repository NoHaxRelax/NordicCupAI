"""Quantify the birth-phase lever on predator spawn timing.

Claim under test: because a birth consumes 24-36 MT words in the agent phase, and the
predator-spawn coin is the LAST draw of the tick (_engine.cpp:1544), choosing to breed on
tick T re-phases every world draw from T onward -- so a seed-aware policy can deliberately
shift when the next predator appears.

Method: at tick T take a snapshot, then run two branches from it under an identical
foraging policy -- one that forces a birth on tick T, one that does not -- and record when
the next predator spawns in each. Everything else is identical, so the difference is
attributable to the birth alone.

This measures the LEVER, not a policy. It answers "how much predator-free time can one
birth buy, and is the effect controllable or just chaotic reshuffling?"

    PYTHONPATH=survival-simulator python mirror/exp_birth_phase.py
"""
import math
import statistics
import sys
import time

import numpy as np

from fastsim import _mirror

_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)

W, H = 1600, 1200
HORIZON = 1500          # ticks to watch after the decision point
DECISIONS = (300, 600, 900)
SEEDS = list(range(1, 33))


def forage(state, force_birth_ids=()):
    """Deterministic: walk at nearest fruit, else hold. No RNG, so it cannot itself
    perturb the stream -- movement and turning consume zero draws (_engine.cpp:1067-1098)."""
    acts = []
    for a in state["observations"]:   # per-agent public state list
        best_d, best_ang = None, 0.0
        for o in a["observations"]:
            if o["type"] == "Fruit" and (best_d is None or o["distance"] < best_d):
                best_d, best_ang = o["distance"], o["angle"]
        if best_d is None:
            move, direction = 0.0, 0.0
        else:
            move, direction = min(a["speed"], best_d), best_ang
        acts.append((int(a["agent_id"]), {
            "move_distance": move,
            "move_direction": direction,
            "turn_angle": 0.0,
            "spawn_agent": int(a["agent_id"]) in force_birth_ids,
        }))
    return acts


def run_branch(e, snap, state, force_ids):
    """Restore to the decision point, act, and report the tick of the next predator spawn."""
    e.restore(snap)
    n0 = len(e.predators())
    st = e.step(forage(state, force_ids))
    for t in range(1, HORIZON):
        if len(e.predators()) > n0:
            return t, len(e.predators()) - n0
        st = e.step(forage(st))
    return None, len(e.predators()) - n0


def main():
    t_start = time.perf_counter()
    rows = []
    no_birth_possible = 0

    for seed in SEEDS:
        e = _mirror.Engine([seed], env_width=W, env_height=H,
                           starting_agents=5, starting_fruits=60, starting_trees=60)
        st = e.state()
        tick = 0
        for T in DECISIONS:
            while tick < T:
                st = e.step(forage(st))
                tick += 1
            # a birth needs a parent above 100 energy after its move (_engine.cpp:1389)
            parents = [int(a["agent_id"]) for a in st["observations"] if a["energy"] > 115]
            if not parents:
                no_birth_possible += 1
                continue
            snap = e.snapshot()
            base_t, _ = run_branch(e, snap, st, ())
            bir_t, _ = run_branch(e, snap, st, {parents[0]})
            e.restore(snap)                       # back to the trunk for the next T
            rows.append((seed, T, base_t, bir_t))

    shifts = [(b - a) for _, _, a, b in rows if a is not None and b is not None]
    both_none = sum(1 for _, _, a, b in rows if a is None and b is None)
    gained = sum(1 for _, _, a, b in rows if a is not None and b is None)
    lost = sum(1 for _, _, a, b in rows if a is None and b is not None)

    print(f"decision points evaluated : {len(rows)}  "
          f"(skipped {no_birth_possible}: no parent above the energy threshold)")
    print(f"both branches quiet       : {both_none}")
    print(f"birth pushed spawn beyond the {HORIZON}-tick horizon : {gained}")
    print(f"birth pulled a spawn INTO the horizon                : {lost}")
    if shifts:
        moved = sum(1 for s in shifts if s != 0)
        print(f"\ncomparable pairs          : {len(shifts)}")
        print(f"  next spawn moved at all : {moved} ({100*moved/len(shifts):.0f}%)")
        print(f"  mean shift              : {statistics.mean(shifts):+.1f} ticks "
              f"({statistics.mean(shifts)/10:+.1f} s)")
        print(f"  median shift            : {statistics.median(shifts):+.1f} ticks")
        print(f"  best  (latest  spawn)   : {max(shifts):+d} ticks ({max(shifts)/10:+.1f} s)")
        print(f"  worst (earliest spawn)  : {min(shifts):+d} ticks ({min(shifts)/10:+.1f} s)")
        later = sum(1 for s in shifts if s > 0)
        print(f"  birth delayed the spawn : {later}/{len(shifts)} "
              f"({100*later/len(shifts):.0f}%)")
        big = sorted(shifts, reverse=True)[:5]
        print(f"  top 5 delays            : {[f'{s/10:+.1f}s' for s in big]}")

    print(f"\nwall clock: {time.perf_counter()-t_start:.1f} s for "
          f"{2*len(rows)} branch rollouts of up to {HORIZON} ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
