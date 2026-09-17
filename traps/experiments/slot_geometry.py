"""Quantify the two-rock 'slot' trap: a gap an agent fits in but a predator cannot.

Geometry (a vertical corridor):

        rock A          gap g          rock B
    +-----------+   .           .   +-----------+
    |           |   .  agent    .   |           |     length L
    +-----------+   .           .   +-----------+
                    ^ mouth (open end)

An agent (radius 5) fits in the gap when g >= 10; a predator (radius 10) is
excluded when g < 20. Predation needs distance < 15, and every predator position
must clear rocks by 10, so the agent is untouchable exactly when its distance to
the nearest predator-legal point is >= 15.

Sweeps gap, corridor length, and how deep the agent sits, running the real
engine each time. Nothing about the predator is modified.
"""

import argparse
import itertools
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

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.elements import biome as biome_mod
from src.elements.agent import Agent
from src.elements.biome import Grassland_biome
from src.elements.environment import Environment

DT = 0.1
WORLD = 400
CX = CY = 200
ROCK_W = 60            # rock thickness; irrelevant to the slot mechanism
PRED_R, AGENT_R = 10, 5
TOUCH = 15             # predator.size + agent.size

_GRASS = Grassland_biome()


def _flat(self):
    a = np.empty((self.width, self.height), dtype=object)
    a[:, :] = _GRASS
    return a


biome_mod.Map_generator.generate = _flat
Environment._render_biome_surface = lambda self: None


def clearance_to_predator_space(rects, px, py, width, height):
    """Distance from a point to the nearest position a predator may occupy."""
    blocked = np.zeros((width, height), dtype=bool)
    for (x, y, w, h) in rects:
        x0 = max(int(math.floor(x - PRED_R)) + 1, 0)
        x1 = min(int(math.ceil(x + w + PRED_R)) - 1, width - 1)
        y0 = max(int(math.floor(y - PRED_R)) + 1, 0)
        y1 = min(int(math.ceil(y + h + PRED_R)) - 1, height - 1)
        if x1 >= x0 and y1 >= y0:
            blocked[x0:x1 + 1, y0:y1 + 1] = True
    # Outside the world is not predator space either.
    dist = distance_transform_edt(blocked)
    return float(dist[int(round(px)), int(round(py))])


class Slot(Environment):
    def __init__(self, gap, length, depth, seed=1, pred_dx=0.0, pred_dy=-35.0):
        super().__init__(width=WORLD, height=WORLD, chunk_size=400,
                         rng=random.Random(seed))
        self.biome_map.fill(_GRASS)
        top = CY - length / 2.0
        self.rock_a = self.spawn_obstacle(x=CX - gap / 2.0 - ROCK_W, y=top,
                                          width=ROCK_W, height=length)
        self.rock_b = self.spawn_obstacle(x=CX + gap / 2.0, y=top,
                                          width=ROCK_W, height=length)
        self.rects = [(o.x, o.y, o.width, o.height) for o in self.obstacles]
        self.mouth_top = top
        self.agent_pos = (float(CX), float(top + depth))

        self.fits = not self._in_obstacle(self.agent_pos, AGENT_R, self.obstacles)
        self.clearance = clearance_to_predator_space(
            self.rects, self.agent_pos[0], self.agent_pos[1], WORLD, WORLD)
        if not self.fits:
            return

        bait = Agent(x=self.agent_pos[0], y=self.agent_pos[1],
                     rng=self.rng, energy=150)
        bait.agent_id = self._next_agent_id
        self._next_agent_id += 1
        self.agents.append(bait)
        self.agents_dict[bait.agent_id] = bait
        self._update_agent_grid()
        bait.max_age = math.inf
        bait.energy = bait.max_energy
        bait.direction = -math.pi / 2          # look out of the mouth
        self.bait = bait

        # Predator starts outside the top mouth, on the corridor centre line.
        self.predator = Environment.spawn_predator(
            self, x=CX + pred_dx, y=top + pred_dy)
        if self.predator is None:
            self.predator = Environment.spawn_predator(self, x=CX, y=top - 35)
        self.predator.direction = math.atan2(
            self.bait.y - self.predator.y, self.bait.x - self.predator.x)
        self.ticks = 0
        self.min_dist = self.distance
        self.max_dist = self.distance
        self.engaged = False
        self.in_range = 0

    def spawn_tree(self, *a, **k):
        return None

    def spawn_predator(self, *a, **k):
        return None

    def spawn_fruit(self, *a, **k):
        return None

    @property
    def distance(self):
        return math.hypot(self.predator.x - self.bait.x,
                          self.predator.y - self.bait.y)

    @property
    def alive(self):
        return self.bait in self.agents

    def step(self):
        self.bait.energy = self.bait.max_energy
        self.non_agent_step(DT)
        self.ticks += 1
        if not self.alive:
            return
        self.bait.energy = self.bait.max_energy
        d = self.distance
        self.min_dist = min(self.min_dist, d)
        self.max_dist = max(self.max_dist, d)
        if d <= self.predator.hearing_radius:
            self.engaged = True
            self.in_range += 1


def run(gap, length, depth, seed, steps, pdx=0.0, pdy=-35.0):
    trial = Slot(gap, length, depth, seed, pdx, pdy)
    if not trial.fits:
        return {"gap": gap, "length": length, "depth": depth, "seed": seed,
                "fits": False, "clearance": round(trial.clearance, 2),
                "status": "agent_does_not_fit"}
    for _ in range(steps):
        trial.step()
        if not trial.alive:
            break
    if not trial.alive:
        status = "EATEN"
    elif trial.in_range / max(1, trial.ticks) > 0.5:
        status = "trapped"           # predator stuck at the mouth
    elif trial.engaged:
        status = "engaged_then_left"
    else:
        status = "never_engaged"
    return {"gap": gap, "length": length, "depth": depth, "seed": seed,
            "fits": True, "clearance": round(trial.clearance, 2),
            "status": status, "min_dist": round(trial.min_dist, 2),
            "ticks": trial.ticks}


def _w(a):
    return run(*a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="gap")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    seeds = list(range(1, a.seeds + 1))

    if a.mode == "gap":
        combos = [(g, 140.0, 70.0) for g in
                  [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19.5,
                   20, 20.5, 21, 22, 24, 26, 30]]
    elif a.mode == "depth":
        combos = [(14.0, 200.0, d) for d in
                  [0, 2, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 80, 100]]
    elif a.mode == "stress":
        combos = [(g, 200.0, d)
                  for g in (10.0, 12.0, 14.0, 16.0, 18.0, 19.0)
                  for d in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0)]
    elif a.mode == "length":
        combos = [(14.0, L, L / 2) for L in
                  [10, 14, 20, 24, 30, 40, 50, 60, 80, 100, 140, 200]]
    else:
        raise SystemExit("bad mode")

    if a.mode == "stress":
        starts = [(0.0, -35.0), (40.0, -30.0), (-40.0, -30.0),
                  (0.0, -80.0), (70.0, 60.0), (-70.0, 120.0)]
    else:
        starts = [(0.0, -35.0)]
    jobs = [(g, L, d, s, a.steps, pdx, pdy)
            for (g, L, d), s, (pdx, pdy) in
            itertools.product(combos, seeds, starts)]
    print(f"{len(jobs)} trials x up to {a.steps} ticks\n", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(_w, jobs, chunksize=2))

    agg = {}
    for r in res:
        k = (r["gap"], r["length"], r["depth"])
        d = agg.setdefault(k, {"n": 0, "clear": r["clearance"],
                               "status": {}, "mind": []})
        d["n"] += 1
        d["status"][r["status"]] = d["status"].get(r["status"], 0) + 1
        if "min_dist" in r:
            d["mind"].append(r["min_dist"])

    print(f"{'gap':>6} {'len':>6} {'depth':>6} {'clearance':>10} "
          f"{'closest':>8}  outcome")
    for k in sorted(agg):
        d = agg[k]
        mind = f"{min(d['mind']):.2f}" if d["mind"] else "-"
        outcome = ", ".join(f"{v}x {s}" for s, v in sorted(d["status"].items()))
        print(f"{k[0]:>6} {k[1]:>6} {k[2]:>6} {d['clear']:>10.2f} "
              f"{mind:>8}  {outcome}")
    print(f"\nagent fits when gap >= {2 * AGENT_R}; predator excluded when "
          f"gap < {2 * PRED_R}; predation needs distance < {TOUCH}")


if __name__ == "__main__":
    main()
