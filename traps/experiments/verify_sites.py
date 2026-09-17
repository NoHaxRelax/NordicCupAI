"""End-to-end: every site trap_sites.py reports must actually hold a predator."""

import argparse
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import require_simulator          # noqa: E402

require_simulator()

from src.elements.agent import Agent
from src.elements.environment import Environment
from trap_sites import find_trap_sites

Environment._render_biome_surface = lambda self: None   # drawing only
W, H, CHUNK, DT = 1600, 1200, 400, 0.1


def build(seed):
    env = Environment(W, H, CHUNK, random.Random(seed))
    for _ in range(W // 20):
        env.spawn_obstacle()
    env.agents_dict = {}
    return env


def trial(seed, site, steps):
    env = build(seed)
    env.spawn_tree = lambda *a, **k: None
    env.spawn_fruit = lambda *a, **k: None

    bx, by = site.bait
    px, py = site.predator_side
    bait = Agent(x=float(bx), y=float(by), rng=env.rng, energy=150)
    bait.agent_id = env._next_agent_id
    env._next_agent_id += 1
    env.agents.append(bait)
    env.agents_dict[bait.agent_id] = bait
    env._update_agent_grid()
    bait.max_age = math.inf
    bait.energy = bait.max_energy
    bait.direction = math.atan2(py - by, px - bx)      # face the predator

    # Place the predator directly: Environment.spawn_predator applies a
    # stricter 10x10 box test than the radius test the engine uses to move.
    if env._in_obstacle((px, py), 10, env.obstacles):
        return "predator_spot_illegal", 0.0
    from src.elements.predator import Predator
    predator = Predator(float(px), float(py), rng=env.rng)
    env.predators.append(predator)
    env._update_predator_grid()
    predator.direction = math.atan2(by - py, bx - px)
    env.spawn_predator = lambda *a, **k: None

    maxd = math.hypot(predator.x - bait.x, predator.y - bait.y)
    engaged = False
    for _ in range(steps):
        bait.energy = bait.max_energy
        env.non_agent_step(DT)
        if bait not in env.agents:
            return "EATEN", maxd
        bait.energy = bait.max_energy
        d = math.hypot(predator.x - bait.x, predator.y - bait.y)
        maxd = max(maxd, d)
        if d <= predator.hearing_radius:
            engaged = True
        elif engaged and d > predator.hearing_radius + 40:
            return "ESCAPED", maxd
    return ("held" if engaged else "no_contact"), maxd


def work(job):
    seed, safety, steps, cap, kinds = job
    env = build(seed)
    sites = find_trap_sites(env.obstacles, W, H, safety=safety,
                            kinds=tuple(kinds.split(",")))
    out = []
    for s in sites[:cap]:
        status, maxd = trial(seed, s, steps)
        out.append((s, status, maxd))
    return seed, len(sites), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=40)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--safety", default="safe")
    ap.add_argument("--kinds", default="wall,slot")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    jobs = [(s, a.safety, a.steps, a.cap, a.kinds) for s in range(a.maps)]
    tally, n_sites_total, fails = {}, 0, []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for seed, n_sites, rows in pool.map(work, jobs):
            n_sites_total += n_sites
            for s, status, maxd in rows:
                key = f"{s.kind}:{status}"
                tally[key] = tally.get(key, 0) + 1
                if status not in ("held",):
                    fails.append((seed, s, status, round(maxd, 1)))

    print(f"\nsafety={a.safety}  maps={a.maps}  steps={a.steps}")
    print(f"sites found: {n_sites_total} ({n_sites_total / a.maps:.2f}/map), "
          f"tested {sum(tally.values())}")
    for k, v in sorted(tally.items()):
        print(f"   {k:<22} {v}")
    for f in fails[:15]:
        print(f"   FAIL seed={f[0]} {f[2]} maxD={f[3]} "
              f"thick={f[1].thickness} sep={f[1].separation} margin={f[1].margin}")


if __name__ == "__main__":
    main()
