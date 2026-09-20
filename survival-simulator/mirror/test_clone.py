"""Correctness gate for fastsim._mirror Engine.clone().

A shadow model is worthless unless a clone is (a) bit-exact with its parent and
(b) fully independent of it. Both are checked here, plus the cost of a clone.

    python mirror/test_clone.py
"""
import hashlib
import random
import sys
import time

import numpy as np

from fastsim import _mirror

_mirror.set_numpy_loops(np.sin, np.cos, np.arctan2, np.hypot)

W, H = 1600, 1200


def fingerprint(e):
    """Everything that defines the engine's future: RNG state plus all dynamic entities."""
    st = e.rng_state()
    h = hashlib.sha256()
    h.update(repr(st).encode())
    h.update(repr(e.info()).encode())
    for row in (e.agents(), e.predators(), e.fruits(), e.trees()):
        h.update(repr(row).encode())
    return h.hexdigest()


def make(seed, **kw):
    return _mirror.Engine([seed], env_width=W, env_height=H, **kw)


def actions_for(e, rng):
    # Engine.step wants a sequence of (agent_id, action) pairs; the action may be a dict.
    out = []
    for a in e.agents():
        out.append((int(a[0]), {
            "move_distance": rng.uniform(0, 8),
            "move_direction": rng.uniform(-3.14, 3.14),
            "turn_angle": rng.uniform(-1, 1),
            "spawn_agent": False,
        }))
    return out


def drive(e, ticks, rng):
    for _ in range(ticks):
        e.step(actions_for(e, rng))


def main():
    failures = []

    # ---------- 1. a clone is bit-exact and stays bit-exact under identical actions ----
    for seed in (1, 2, 3, 614466944):
        e = make(seed, starting_predators=6)
        drive(e, 300, random.Random(seed))          # get into a messy mid-game state
        c = e.clone()

        if fingerprint(e) != fingerprint(c):
            failures.append(f"seed {seed}: clone differs from parent immediately")
            continue

        # identical action streams must keep them identical for a long run
        ra, rb = random.Random(77), random.Random(77)
        for t in range(2000):
            e.step(actions_for(e, ra))
            c.step(actions_for(c, rb))
            if t % 250 == 0 and fingerprint(e) != fingerprint(c):
                failures.append(f"seed {seed}: diverged at tick {t}")
                break
        else:
            if fingerprint(e) != fingerprint(c):
                failures.append(f"seed {seed}: diverged by tick 2000")
            else:
                print(f"  seed {seed:>10}: 2000 ticks bit-identical  "
                      f"(score {e.info()['score']:.6f})")

    # ---------- 2. a clone is independent: diverging it must not touch the parent ------
    e = make(7, starting_predators=6)
    drive(e, 400, random.Random(7))
    before = fingerprint(e)
    c = e.clone()
    drive(c, 500, random.Random(999))               # completely different actions
    after = fingerprint(e)
    if before != after:
        failures.append("clone mutated its parent")
    elif fingerprint(c) == before:
        failures.append("diverged clone still matches parent (clone is a no-op?)")
    else:
        print("  independence: parent unchanged after clone ran 500 different ticks")

    # ---------- 3. branch-and-replay: clone, act, and the parent can reproduce it ------
    e = make(11, starting_predators=6)
    drive(e, 300, random.Random(11))
    snap = e.clone()
    log = []
    r = random.Random(4)
    for _ in range(400):
        acts = actions_for(e, r)
        log.append(acts)
        e.step(acts)
    target = fingerprint(e)
    for acts in log:                                 # replay the exact log into the snapshot
        snap.step(acts)
    if fingerprint(snap) != target:
        failures.append("replaying a recorded action log into a snapshot did not reproduce it")
    else:
        print("  branch-and-replay: 400-tick log reproduced exactly from a snapshot")

    # ---------- 4. snapshot/restore is equivalent to clone, and reversible ------------
    for seed in (1, 614466944):
        e = make(seed, starting_predators=6)
        drive(e, 300, random.Random(seed))
        snap = e.snapshot()
        at_snap = fingerprint(e)

        ra = random.Random(21)
        want = []
        for _ in range(600):                      # run forward, remember the trajectory
            e.step(actions_for(e, ra))
            want.append(fingerprint(e))

        e.restore(snap)                           # rewind
        if fingerprint(e) != at_snap:
            failures.append(f"seed {seed}: restore did not reproduce the snapshot state")
            continue
        ra = random.Random(21)
        got = []
        for _ in range(600):                      # and re-run: must retrace exactly
            e.step(actions_for(e, ra))
            got.append(fingerprint(e))
        if got != want:
            bad = next(i for i, (x, y) in enumerate(zip(got, want)) if x != y)
            failures.append(f"seed {seed}: retrace after restore diverged at tick {bad}")
        else:
            print(f"  seed {seed:>10}: restore + 600-tick retrace bit-identical")

    # ---------- 5. restore must also work across a DIFFERENT branch -------------------
    e = make(3, starting_predators=6)
    drive(e, 300, random.Random(3))
    snap = e.snapshot()
    drive(e, 400, random.Random(1234))            # wander off down some other branch
    e.restore(snap)
    ref = make(3, starting_predators=6)
    drive(ref, 300, random.Random(3))
    if fingerprint(e) != fingerprint(ref):
        failures.append("restore after diverging did not return to the snapshot state")
    else:
        print("  restore recovers exactly after an unrelated 400-tick branch")

    # ---------- 6. cost ---------------------------------------------------------------
    # Drive with a foraging policy, not random actions: random actions starve the colony
    # and the costs below would then be measured at zero population.
    def forage(state):
        acts = []
        for a in state["observations"]:
            best_d, best_ang = None, 0.0
            for o in a["observations"]:
                if o["type"] == "Fruit" and (best_d is None or o["distance"] < best_d):
                    best_d, best_ang = o["distance"], o["angle"]
            move, direction = (0.0, 0.0) if best_d is None else (min(a["speed"], best_d), best_ang)
            acts.append((int(a["agent_id"]), {"move_distance": move, "move_direction": direction,
                                              "turn_angle": 0.0, "spawn_agent": False}))
        return acts

    e = make(1, starting_predators=8, starting_fruits=60, starting_trees=60)
    st = e.state()
    for _ in range(600):
        st = e.step(forage(st))
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        e.clone()
    clone_ms = (time.perf_counter() - t0) / n * 1e3

    t0 = time.perf_counter()
    for _ in range(n):
        s = e.snapshot()
    snap_ms = (time.perf_counter() - t0) / n * 1e3

    s = e.snapshot()
    t0 = time.perf_counter()
    for _ in range(n):
        e.restore(s)
    rest_ms = (time.perf_counter() - t0) / n * 1e3

    acts = forage(st)
    t0 = time.perf_counter()
    for _ in range(200):
        e.step(acts)
    step_ms = (time.perf_counter() - t0) / 200 * 1e3
    info = e.info()
    print(f"\n  population: {len(e.agents())} agents, {len(e.predators())} predators, "
          f"{len(e.trees())} trees, {len(e.fruits())} fruits")
    print(f"  clone     : {clone_ms:8.3f} ms   ({clone_ms / step_ms:6.1f} engine ticks)")
    print(f"  snapshot  : {snap_ms:8.3f} ms   ({snap_ms / step_ms:6.1f} engine ticks)")
    print(f"  restore   : {rest_ms:8.3f} ms   ({rest_ms / step_ms:6.1f} engine ticks)")
    print(f"  step      : {step_ms:8.3f} ms")
    print(f"  snapshot+restore is {clone_ms / (snap_ms + rest_ms):.1f}x cheaper than clone")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL CLONE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
