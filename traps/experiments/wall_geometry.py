"""Which rock geometries actually HOLD an engaged predator?

The predator starts already pinned against the far face of the rock, which is
the state a lure delivers it into. From there the unmodified engine runs:
predator AI, energy, rest cycles, collision sliding and sensing are original.

Separates the two failure modes:
  * never engaged  - predator never got within smell range (acquisition problem)
  * escaped        - predator was engaged, then rounded the rock / ate the bait
"""

import argparse
import itertools
import json
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

from src.elements import biome as biome_mod
from src.elements.biome import Grassland_biome
from src.elements.environment import Environment

DT = 0.1
WORLD = 400
CENTRE = 200

# Uniform grassland everywhere and nothing is drawn, so the generated biome map
# and per-pixel terrain render are pure setup cost. Physics is untouched.
_GRASS = Grassland_biome()


def _flat_map(self):
    arr = np.empty((self.width, self.height), dtype=object)
    arr[:, :] = _GRASS
    return arr


biome_mod.Map_generator.generate = _flat_map
Environment._render_biome_surface = lambda self: None


class RockHold(Environment):
    def __init__(self, rock_w, rock_h, gap, bait_off, pred_off,
                 predator_awake, seed):
        super().__init__(width=WORLD, height=WORLD, chunk_size=400,
                         rng=random.Random(seed))
        self.biome_map.fill(_GRASS)

        self.rock = self.spawn_obstacle(x=CENTRE, y=CENTRE - rock_h / 2.0,
                                        width=rock_w, height=rock_h)
        self.rock_top = self.rock.y
        self.rock_bottom = self.rock.y + rock_h

        # Bait on the far (+x) face; bait_off slides it along the face toward a corner.
        self.bait_position = (self.rock.x + rock_w + gap, CENTRE + bait_off)
        self.spawn_agent(x=self.bait_position[0], y=self.bait_position[1])
        self.bait = self.agents[0]
        self.bait.direction = math.pi
        self.bait.max_age = math.inf
        self.bait.energy = self.bait.max_energy

        # Predator starts pinned on the near (-x) face, offset along it.
        self.predator = Environment.spawn_predator(
            self, x=self.rock.x - 10, y=CENTRE + bait_off + pred_off)
        if self.predator is None:
            raise RuntimeError("predator start not free")
        self.predator.direction = 0.0
        if predator_awake:
            self.predator.energy = self.predator.max_energy
            self.predator.resting = False

        self.ticks = 0
        self.engaged = False
        self.escaped = False
        self.eaten = False
        self.in_range_ticks = 0
        self.engaged_ticks = 0
        self.min_distance = self.max_distance = self.distance

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
            self.eaten = True
            self.escaped = True
            return
        self.bait.energy = self.bait.max_energy
        d = self.distance
        self.min_distance = min(self.min_distance, d)
        self.max_distance = max(self.max_distance, d)
        if d <= self.predator.hearing_radius:
            self.engaged = True
            self.in_range_ticks += 1
        if self.engaged:
            self.engaged_ticks += 1
            # Past the far face means it got round to the bait's side.
            if self.predator.x > self.rock.x + self.rock.width:
                self.escaped = True


def run_trial(rock_w, rock_h, gap, bait_off, pred_off, awake, seed, steps):
    margin = rock_h / 2.0 - abs(bait_off)
    try:
        t = RockHold(rock_w, rock_h, gap, bait_off, pred_off, awake, seed)
    except RuntimeError:
        return None
    for _ in range(steps):
        t.step()
        if t.escaped:
            break
    hold_frac = t.in_range_ticks / max(1, t.engaged_ticks)
    return {
        "w": round(rock_w, 2), "h": round(rock_h, 2), "gap": gap,
        "bait_off": bait_off, "pred_off": pred_off, "margin": round(margin, 2),
        "awake": awake, "seed": seed, "ticks": t.ticks,
        "engaged": t.engaged, "escaped": t.escaped, "eaten": t.eaten,
        "hold_frac": round(hold_frac, 4),
        "maxd": round(t.max_distance, 3), "mind": round(t.min_distance, 3),
        # Held == engaged, never escaped, and stayed pinned the whole time.
        "held": bool(t.engaged and not t.escaped and hold_frac > 0.99),
    }


def _worker(a):
    return run_trial(*a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="width")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = list(range(1, a.seeds + 1))
    pred_offs = [-20.0, -8.0, 0.0, 8.0, 20.0]
    awake = [False, True]

    if a.mode == "width":
        combos = [(w, 100.0, 5.0, 0.0) for w in
                  [30 + 1.25 * i for i in range(0, 25)]]
    elif a.mode == "gap":
        combos = [(w, 100.0, g, 0.0)
                  for w in (30.0, 35.0, 40.0, 45.0)
                  for g in (5.0, 10.0, 15.0, 20.0, 25.0)]
    elif a.mode == "margin":
        # Slide the bait toward a corner, and shrink the rock, to find the
        # perpendicular extent needed on each side of the bait.
        combos = [(w, h, 5.0, off)
                  for w in (30.0, 35.0)
                  for h, off in [(h, off) for h in (30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
                                 for off in (0.0,)]
                  ] + [(30.0, 100.0, 5.0, off) for off in (10.0, 20.0, 30.0, 35.0, 40.0, 45.0)]
    elif a.mode == "margin2":
        # h = 2*margin with the bait centred, so 'margin' is the rock extent
        # on each side of the bait axis. Predator starts well inside the face
        # so this isolates holding from "arrived past the rock's end".
        combos = [(w, 2 * m, 5.0, 0.0)
                  for w in (30.0, 32.5, 35.0, 37.5)
                  for m in (20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 40.0, 45.0)]
        pred_offs = [-12.0, -6.0, 0.0, 6.0, 12.0]
    else:
        raise SystemExit("bad mode")

    jobs = [(w, h, g, bo, po, aw, s, a.steps)
            for (w, h, g, bo), po, aw, s in itertools.product(
                combos, pred_offs, awake, seeds)]
    print(f"{len(jobs)} trials x up to {a.steps} ticks", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        res = [r for r in pool.map(_worker, jobs, chunksize=4) if r]

    agg = {}
    for r in res:
        k = (r["w"], r["h"], r["gap"], r["bait_off"], r["margin"])
        d = agg.setdefault(k, {"n": 0, "held": 0, "eng": 0, "esc": 0,
                               "eaten": 0, "maxd": 0.0})
        d["n"] += 1
        d["held"] += r["held"]
        d["eng"] += r["engaged"]
        d["esc"] += r["escaped"]
        d["eaten"] += r["eaten"]
        d["maxd"] = max(d["maxd"], r["maxd"])

    print(f"\n{'w':>6} {'h':>6} {'gap':>4} {'boff':>5} {'marg':>5} "
          f"{'held':>8} {'engaged':>8} {'escaped':>7} {'eaten':>5} {'maxD':>7}")
    for k in sorted(agg):
        d = agg[k]
        print(f"{k[0]:>6} {k[1]:>6} {k[2]:>4} {k[3]:>5} {k[4]:>5} "
              f"{d['held']:>3}/{d['n']:<4} {d['eng']:>8} {d['esc']:>7} "
              f"{d['eaten']:>5} {d['maxd']:>7.2f}")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
