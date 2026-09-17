"""Find every place on a map where an agent can stand and not be caught.

Both kinds of site exploit the same engine facts: a predator smells through
walls (`Creature.observe` applies no occlusion test inside `hearing_radius`,
60 for predators), `Predator.step` always charges the nearest agent it senses
within `hearing_radius * 1.5`, and `Environment.update_entity_position` resolves
a blocked move by rotating the direction while keeping the distance.

WALL sites -- behavioural. A bait parked just across a thin rock is charged
forever and never reached, because the predator slides along the face instead
of pathing round it. It *could* walk around; its AI does not. Requires:

  separation  bait-to-predator distance must stay inside the 60-unit smell
              radius even at the far end of the predator's oscillation, which
              adds about 7.5 units. Measured limit 52.5.
  thickness   at most 37.5 units along the trap axis (the separation limit with
              the bait hugging the face at its minimum 5-unit standoff).
  margin      at least 32.5 units of rock either side of the bait, or the
              predator slides round the end. 40 covers every thickness.

SLOT sites -- geometric, and therefore stronger. A gap at least 10 wide takes an
agent (radius 5) but a gap under 20 excludes a predator (radius 10). An agent is
eaten only at distance < 15, and every predator position clears rocks by 10, so
the agent simply cannot be touched when:

  clearance   distance from the agent to the nearest position a predator may
              legally occupy is at least 15. Measured exactly across 648 runs:
              caught at 14, safe at 15, in every gap width from 10 to 19.

  Gap width itself is not an independent requirement -- it only decides whether
  clearance can exceed zero. Above clearance ~45 the predator stops reliably
  staying in smell range, so the spot shelters the agent but traps nothing;
  those are reported as kind 'shelter' and are not returned by default.

Verification inflates rocks by slightly less than the predator radius, so gaps
above about 18 are rejected even though the true limit is 20. That is deliberate:
near 20 the legal region is a sliver a predator can still squeeze into, and a
naive grid reports a huge clearance for a spot it walks straight into. The scan
therefore under-reports clearance rather than ever over-reporting it.

Slots are found with a distance transform rather than by pairing rocks, so this
also catches notches formed by overlapping rocks, dead-end corners, and gaps
between a rock and the map boundary.

Usage:
    from trap_sites import find_trap_sites
    sites = find_trap_sites(env.obstacles, env.width, env.height)
    best = sites[0]        # sorted, safest first; slots outrank walls
    bait_x, bait_y = best.bait
    if best.guaranteed:    # slot: capture is geometrically impossible
        ...

CLI:
    python trap_sites.py --seed 3            # scan a generated map
    python trap_sites.py --seed 3 --render out.png
    python trap_sites.py --seed 3 --kinds slot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, label

# Engine constants (src/elements/predator.py, creature.py, agent.py).
PREDATOR_SMELL = 60.0
PREDATOR_RADIUS = 10.0
AGENT_RADIUS = 5.0
# Environment.non_agent_step eats an agent when distance < predator.size +
# agent.size. Every predator position must clear rocks by PREDATOR_RADIUS, so an
# agent further than this from the nearest predator-legal point cannot be caught.
TOUCH_DISTANCE = PREDATOR_RADIUS + AGENT_RADIUS     # 15.0

# Measured in simulation; see the module docstring.
SEPARATION_LIMIT = 52.5     # bait-to-predator distance a wall trap still holds
OSCILLATION = 7.5           # how far past that the predator swings
# Past this the predator does not reliably stay: it drifts out of smell range
# and wanders off. The agent is still safe, but the spot shelters rather than
# traps. Measured on real maps: holds at 45, unreliable from about 50 up.
SLOT_HOLD_MAX = 45.0

PRESETS = {
    # The empirical knees. ~99% of wall sites hold, but sites at the boundary
    # are one bad bounce from failing.
    "measured": {"max_thickness": 37.5, "min_margin": 32.5,
                 "max_separation": 52.5, "min_clearance": 15.0},
    # Recommended. Keeps a buffer on every limit; costs sites, not reliability.
    "safe": {"max_thickness": 35.0, "min_margin": 40.0,
             "max_separation": 50.0, "min_clearance": 18.0},
    # Only the most comfortable geometry.
    "paranoid": {"max_thickness": 33.0, "min_margin": 45.0,
                 "max_separation": 48.0, "min_clearance": 22.0},
}


@dataclass(frozen=True)
class TrapSite:
    """One place to park a bait agent so a predator cannot reach it.

    Two kinds, with different strength of guarantee:

    'wall'  Behavioural. The predator *could* walk round the rock but its AI
            charges straight at what it smells, so it pins itself against the
            face. Depends on thickness, margin and separation.
    'slot'  Geometric. The gap is wide enough for an agent (radius 5) but not
            for a predator (radius 10), so no legal predator position is within
            touching distance. Depends only on `clearance`, and holds no matter
            what the predator's AI does.
    """

    kind: str                           # 'wall' or 'slot'
    bait: Tuple[float, float]           # where the bait agent stands
    predator_side: Tuple[float, float]  # closest spot a predator can occupy
    separation: float                   # bait-to-predator distance
    score: float                        # higher is safer
    axis: str = ""                      # wall: 'x' if thin along x
    thickness: float = 0.0              # wall: rock thickness; slot: gap width
    margin: float = 0.0                 # wall: rock extent each side of bait
    wall_span: Tuple[float, float] = (0.0, 0.0)
    predator_clearance: float = 0.0     # wall: open depth behind the predator
    clearance: float = 0.0              # slot: distance to predator-legal space

    @property
    def worst_case_distance(self) -> float:
        """Closest the predator ever gets.

        For a wall that is the separation plus its oscillation; for a slot the
        predator simply cannot beat the clearance.
        """
        if self.kind in ("slot", "shelter"):
            return self.clearance
        return self.separation + OSCILLATION

    @property
    def guaranteed(self) -> bool:
        """True when geometry alone prevents capture, regardless of predator AI."""
        return (self.kind in ("slot", "shelter")
                and self.clearance >= TOUCH_DISTANCE)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["worst_case_distance"] = round(self.worst_case_distance, 2)
        d["guaranteed"] = self.guaranteed
        return d


def _as_rects(obstacles: Iterable) -> List[Tuple[float, float, float, float]]:
    """Accept Obstacle objects or plain (x, y, width, height) tuples."""
    rects = []
    for o in obstacles:
        if hasattr(o, "width") and hasattr(o, "x"):
            rects.append((float(o.x), float(o.y), float(o.width), float(o.height)))
        else:
            x, y, w, h = o
            rects.append((float(x), float(y), float(w), float(h)))
    return rects


def _solid(rects, width, height) -> np.ndarray:
    """Cells lying inside a rock. Run extent equals the rock's true size."""
    grid = np.zeros((width, height), dtype=bool)
    for (x, y, w, h) in rects:
        x0, x1 = max(int(math.ceil(x)), 0), min(int(math.floor(x + w)), width - 1)
        y0, y1 = max(int(math.ceil(y)), 0), min(int(math.floor(y + h)), height - 1)
        if x1 >= x0 and y1 >= y0:
            grid[x0:x1 + 1, y0:y1 + 1] = True
    return grid


def _blocked(rects, width, height, radius) -> np.ndarray:
    """Mirrors Environment._in_obstacle: a creature of the given radius at an
    integer point is inside iff x - r < px < x + w + r (strict, hence +1/-1)."""
    grid = np.zeros((width, height), dtype=bool)
    for (x, y, w, h) in rects:
        x0 = max(int(math.floor(x - radius)) + 1, 0)
        x1 = min(int(math.ceil(x + w + radius)) - 1, width - 1)
        y0 = max(int(math.floor(y - radius)) + 1, 0)
        y1 = min(int(math.ceil(y + h + radius)) - 1, height - 1)
        if x1 >= x0 and y1 >= y0:
            grid[x0:x1 + 1, y0:y1 + 1] = True
    return grid


def _runs(occ: np.ndarray):
    """First and last index of each cell's contiguous run along axis 0."""
    n = occ.shape[0]
    idx = np.arange(n, dtype=np.int32)[:, None]

    prev = np.empty_like(occ)
    prev[0] = False
    prev[1:] = occ[:-1]
    left = np.where(occ & ~prev, idx, np.int32(-1))
    np.maximum.accumulate(left, axis=0, out=left)

    nxt = np.empty_like(occ)
    nxt[-1] = False
    nxt[:-1] = occ[1:]
    right = np.where(occ & ~nxt, idx, np.int32(n))
    right = np.minimum.accumulate(right[::-1], axis=0)[::-1]
    return left, right


EPS = 0.01          # clears the engine's strict "<" obstacle test


def _free_point(rects, px, py, radius) -> bool:
    """Environment._in_obstacle, evaluated in continuous coordinates."""
    for (x, y, w, h) in rects:
        if x - radius < px < x + w + radius and y - radius < py < y + h + radius:
            return False
    return True


def _exact_span(rects, row, lo, hi):
    """True extent of the wall crossing `row`, merged across touching rocks.

    The raster only locates the wall to the nearest unit; rocks have float
    coordinates, so the legal standing points must come from the rectangles
    themselves or marginal walls get rejected for a rounding error.
    """
    touching = [r for r in rects
                if r[1] - EPS <= row <= r[1] + r[3] + EPS
                and r[0] <= hi + 1 and r[0] + r[2] >= lo - 1]
    if not touching:
        return None
    span_lo = min(r[0] for r in touching)
    span_hi = max(r[0] + r[2] for r in touching)
    # Absorb anything else on this row that overlaps the growing span.
    changed = True
    while changed:
        changed = False
        for (x, y, w, h) in rects:
            if not (y - EPS <= row <= y + h + EPS):
                continue
            if x <= span_hi and x + w >= span_lo:
                if x < span_lo:
                    span_lo, changed = x, True
                if x + w > span_hi:
                    span_hi, changed = x + w, True
    return span_lo, span_hi


def _scan_axis(occ, free_bait, free_pred, rects, cfg, axis_name) -> List[TrapSite]:
    """Walls thin along axis 0 and long along axis 1."""
    max_t = cfg["max_thickness"]
    min_m = cfg["min_margin"]
    max_s = cfg["max_separation"]

    left, right = _runs(occ)
    thickness = right - left
    thin = occ & (thickness <= max_t)
    if not thin.any():
        return []

    # The cross-section must be unchanged on the next row, so the predator stays
    # blocked by the same wall as it slides along the face.
    same = np.zeros_like(occ)
    same[:, :-1] = (thin[:, :-1] & thin[:, 1:] &
                    (left[:, :-1] == left[:, 1:]) &
                    (right[:, :-1] == right[:, 1:]))

    # Length of that unchanged stretch, measured downward from each cell.
    need = int(math.ceil(2 * min_m))
    run = np.zeros(occ.shape, dtype=np.int32)
    acc = np.zeros(occ.shape[0], dtype=np.int32)
    for j in range(occ.shape[1] - 1, -1, -1):
        acc = np.where(same[:, j], acc + 1, 0)
        run[:, j] = acc

    long_enough = thin & (run >= need)
    if not long_enough.any():
        return []

    # Keep only the first row of each stretch, then one entry per wall.
    top = np.zeros_like(long_enough)
    top[:, 0] = long_enough[:, 0]
    top[:, 1:] = long_enough[:, 1:] & ~same[:, :-1]
    xs, ys = np.nonzero(top)
    if xs.size == 0:
        return []
    keys = (left[xs, ys].astype(np.int64) * 100000 + right[xs, ys]) * 100000 + ys
    _, uniq = np.unique(keys, return_index=True)
    xs, ys = xs[uniq], ys[uniq]

    nx, ny = occ.shape
    sites: List[TrapSite] = []
    for x, y in zip(xs, ys):
        lo, hi = int(left[x, y]), int(right[x, y])
        length = int(run[x, y]) + 1
        mid = y + length // 2                    # centre the bait on the stretch
        if mid >= ny:
            continue
        margin = length / 2.0

        # Resolve the wall's real faces, then stand just clear of each.
        span = _exact_span(rects, float(mid), lo, hi)
        if span is None:
            continue
        span_lo, span_hi = span
        t = span_hi - span_lo
        if t > max_t:
            continue
        bx = span_hi + AGENT_RADIUS + EPS
        px = span_lo - PREDATOR_RADIUS - EPS
        sep = bx - px
        if sep > max_s or not (0 <= px and bx < nx):
            continue
        if not _free_point(rects, bx, float(mid), AGENT_RADIUS):
            continue
        if not _free_point(rects, px, float(mid), PREDATOR_RADIUS):
            continue

        # How much open room the predator has behind its resting spot. Tight
        # corridors still trap, but leave it less room to settle.
        clearance = 0.0
        step = int(math.floor(px))
        while step - 1 >= 0 and clearance < 60 and free_pred[step - 1, mid]:
            step -= 1
            clearance += 1.0

        # Prefer short separation and generous margin.
        score = (max_s - sep) + min(margin, 80.0) * 0.5 + min(clearance, 30.0) * 0.1
        bait_xy = (round(float(bx), 2), float(mid))
        pred_xy = (round(float(px), 2), float(mid))
        if axis_name == "y":
            bait_xy, pred_xy = bait_xy[::-1], pred_xy[::-1]
        sites.append(TrapSite(
            kind="wall",
            bait=bait_xy, predator_side=pred_xy, axis=axis_name,
            thickness=round(t, 2), separation=round(sep, 2),
            margin=round(margin, 1), wall_span=(float(y), float(y + length)),
            predator_clearance=round(clearance, 1), score=round(float(score), 2)))
    return sites


# The candidate grid uses the true predator radius: under-inflating it there
# would throw away perfectly good gaps of 18-19, which genuinely exclude a
# 10-radius predator. All the conservatism lives in the verification below,
# which is what actually decides whether a site is accepted.
VERIFY_SHRINK = 0.5     # verify with rocks inflated slightly under the radius
VERIFY_STEP = 0.25      # sub-unit sampling for the verification


def _legal_distance(rects, px, py, radius, width, height,
                    shrink=VERIFY_SHRINK, step=VERIFY_STEP) -> float:
    """Distance from (px, py) to the nearest point a predator may occupy.

    Returns `inf` when nothing legal lies within `radius`.

    A plain integer grid is not safe here: where a gap is close to the predator's
    20-unit diameter the legal region is a sliver that can fall between grid
    points, so the naive distance transform reports a huge clearance for a spot
    a predator walks straight into. Sampling at a quarter unit while inflating
    rocks by slightly LESS than the predator radius guarantees the opposite
    bias -- any genuinely legal point is detected, so this under-reports
    clearance rather than over-reporting it.
    """
    r = PREDATOR_RADIUS - shrink
    lo_x, hi_x = px - radius, px + radius
    lo_y, hi_y = py - radius, py + radius
    near = [q for q in rects
            if q[0] - r <= hi_x and q[0] + q[2] + r >= lo_x
            and q[1] - r <= hi_y and q[1] + q[3] + r >= lo_y]

    xs = np.arange(lo_x, hi_x + step, step)
    ys = np.arange(lo_y, hi_y + step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    # Outside the world is not somewhere a predator can be either.
    blocked = (gx < 0) | (gx > width - 1) | (gy < 0) | (gy > height - 1)
    for (x, y, w, h) in near:
        blocked |= ((gx > x - r) & (gx < x + w + r) &
                    (gy > y - r) & (gy < y + h + r))
    if blocked.all():
        return math.inf
    d = np.hypot(gx - px, gy - py)
    d[blocked] = np.inf
    best = float(d.min())
    return best if best <= radius else math.inf


def _find_slots(rects, width, height, cfg, kinds) -> List[TrapSite]:
    """Pockets an agent fits into but a predator cannot reach into.

    Works off a distance transform rather than pairs of rocks, so it catches
    every shape of pocket: a gap between two rocks, a notch where rocks overlap,
    a slot between a rock and the boundary wall, a dead-end corner.

    Measured thresholds (slot_experiment.py, 648 trials): the agent is caught at
    clearance 14 and safe at 15 in every gap width from 10 to 19, matching
    TOUCH_DISTANCE exactly. Gap width itself turned out not to matter -- it only
    decides whether clearance can exceed zero at all.
    """
    min_clear = cfg["min_clearance"]

    # Candidates only; each survivor is confirmed against exact geometry below.
    blocked_pred = _blocked(rects, width, height, PREDATOR_RADIUS)
    blocked_agent = _blocked(rects, width, height, AGENT_RADIUS)
    solid = _solid(rects, width, height)

    clearance = distance_transform_edt(blocked_pred)
    to_rock = distance_transform_edt(~solid)     # half the local corridor width

    # A pocket is anywhere an agent can stand but a predator cannot. Label these
    # independently of the threshold: labelling the thresholded mask instead
    # would split one pocket into several as the preset tightens, reporting the
    # same place more than once and making a stricter preset look richer.
    pocket = (~blocked_agent) & blocked_pred
    usable = pocket & (clearance >= min_clear)
    if not usable.any():
        return []
    components, n = label(pocket)
    if n == 0:
        return []

    sites: List[TrapSite] = []
    for comp in range(1, n + 1):
        mask = (components == comp) & usable
        if not mask.any():
            continue
        # Prefer deep cells that a predator can still smell, and among equally
        # deep ones the most centred, so the agent has room on both sides.
        scores = np.where(mask & (clearance <= SLOT_HOLD_MAX),
                          np.floor(clearance) * 1000.0 + to_rock, -1.0)
        if not (scores > 0).any():
            scores = np.where(mask, np.floor(clearance) * 1000.0 + to_rock, -1.0)
        order = np.argsort(scores, axis=None)[::-1]
        n_cand = int((scores > 0).sum())
        if n_cand == 0:
            continue
        # Spread the tries across the pocket instead of taking adjacent cells.
        stride = max(1, n_cand // 8)
        tries = [order[i] for i in range(0, min(n_cand, 8 * stride), stride)]

        for idx in tries:
            bx, by = np.unravel_index(int(idx), clearance.shape)
            # Authoritative check: is anything predator-legal too close?
            near = _legal_distance(rects, float(bx), float(by),
                                   min_clear, width, height)
            if near < min_clear:
                continue
            exact = _legal_distance(rects, float(bx), float(by),
                                    SLOT_HOLD_MAX, width, height)
            # Where the predator actually ends up. Located against full-radius
            # geometry, unlike the deliberately conservative safety test above,
            # so a spot with no real predator position nearby is a shelter.
            spot = _nearest_predator_spot(rects, float(bx), float(by),
                                          SLOT_HOLD_MAX + 6, width, height)
            holds = math.isfinite(exact) and spot is not None
            if holds:
                kind, c = "slot", exact
            else:
                # Safe, but too deep for the predator to ever smell: a shelter
                # rather than a trap. Re-measure coarsely to report a real value.
                kind = "shelter"
                c = _legal_distance(rects, float(bx), float(by), 4 * SLOT_HOLD_MAX,
                                    width, height, step=0.5)
                if not math.isfinite(c):
                    c = 4 * SLOT_HOLD_MAX
            if kind not in kinds:
                continue
            px, py = spot if spot is not None else (float(bx), float(by))
            sites.append(TrapSite(
                kind=kind, bait=(float(bx), float(by)),
                predator_side=(float(px), float(py)),
                separation=round(float(c), 2),
                # Slots outrank walls: capture is geometrically impossible.
                score=round((100.0 if kind == "slot" else 10.0)
                            + min(float(c), SLOT_HOLD_MAX), 2),
                thickness=round(float(to_rock[bx, by]) * 2.0, 2),
                clearance=round(float(c), 2),
                margin=round(float(c) - TOUCH_DISTANCE, 2)))
            break
    return sites


def _nearest_predator_spot(rects, bx, by, radius, width, height):
    """The closest point a predator may actually occupy.

    Checked against continuous geometry, not the grid: a cell the raster calls
    free can still fail Environment._in_obstacle by a fraction of a unit.
    """
    step = 0.5
    xs = np.arange(bx - radius, bx + radius + step, step)
    ys = np.arange(by - radius, by + radius + step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    blocked = (gx < 0) | (gx > width - 1) | (gy < 0) | (gy > height - 1)
    for (x, y, w, h) in rects:
        blocked |= ((gx > x - PREDATOR_RADIUS) & (gx < x + w + PREDATOR_RADIUS) &
                    (gy > y - PREDATOR_RADIUS) & (gy < y + h + PREDATOR_RADIUS))
    if blocked.all():
        return None
    d = np.hypot(gx - bx, gy - by)
    d[blocked] = np.inf
    idx = int(np.argmin(d))
    return float(gx.flat[idx]), float(gy.flat[idx])


def find_trap_sites(obstacles: Iterable, width: int = 1600, height: int = 1200,
                    safety: str = "safe", limit: int | None = None,
                    kinds: Sequence[str] = ("wall", "slot")) -> List[TrapSite]:
    """All viable trap locations on a map, safest first.

    Args:
        obstacles: Obstacle objects (env.obstacles) or (x, y, w, h) tuples.
                   Partial knowledge is fine -- pass what has been mapped so far.
        width, height: world size.
        safety: 'measured', 'safe' (default) or 'paranoid'; see PRESETS.
        limit: keep only the best N sites.

    Overlapping rocks are handled: the scan runs on a rasterised occupancy grid,
    so two rocks that merge into one thick wall are correctly rejected, and two
    that merge into one long wall are correctly accepted. The scan is
    conservative -- it only accepts walls with an unchanging cross-section along
    the whole stretch -- so it under-reports rather than over-reports.
    """
    if safety not in PRESETS:
        raise ValueError(f"safety must be one of {sorted(PRESETS)}")
    cfg = PRESETS[safety]
    rects = _as_rects(obstacles)
    if not rects:
        return []

    occ = _solid(rects, width, height)
    free_bait = ~_blocked(rects, width, height, AGENT_RADIUS)
    free_pred = ~_blocked(rects, width, height, PREDATOR_RADIUS)

    sites: List[TrapSite] = []
    if "wall" in kinds:
        flipped = [(y, x, h, w) for (x, y, w, h) in rects]
        sites += _scan_axis(occ, free_bait, free_pred, rects, cfg, "x")
        sites += _scan_axis(np.ascontiguousarray(occ.T),
                            np.ascontiguousarray(free_bait.T),
                            np.ascontiguousarray(free_pred.T), flipped, cfg, "y")
    if "slot" in kinds or "shelter" in kinds:
        sites += _find_slots(rects, width, height, cfg, kinds)
    sites.sort(key=lambda s: s.score, reverse=True)
    return sites[:limit] if limit else sites


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _real_map(seed: int, width: int, height: int):
    """The obstacles of the actual seeded game map.

    Builds a real Environment, so the biome generator consumes the RNG stream
    first, exactly as in a real game. Slower than _sampled_map but exact.
    """
    import random
    from _bootstrap import headless, require_simulator
    headless()
    require_simulator()
    from src.elements.environment import Environment
    env = Environment(width, height, 400, random.Random(seed))
    for _ in range(width // 20):
        env.spawn_obstacle()
    return [(o.x, o.y, o.width, o.height) for o in env.obstacles]


def _sampled_map(seed: int, width: int, height: int):
    """Obstacles drawn from the same distribution as create_environment.

    The draws match spawn_obstacle exactly, but because a real Environment
    consumes the RNG for biome generation first, a given seed does NOT produce
    the same map as a real game. Use --real for that.
    """
    import random
    rng = random.Random(seed)
    rects = [(0.0, 0.0, float(width), 30.0),
             (0.0, height - 30.0, float(width), 30.0),
             (0.0, 0.0, 30.0, float(height)),
             (width - 30.0, 0.0, 30.0, float(height))]
    for _ in range(width // 20):
        w = rng.uniform(30, 100)
        h = rng.uniform(30, 100)
        x = rng.uniform(0, width - w)
        y = rng.uniform(0, height - h)
        rects.append((x, y, w, h))
    return rects


def _render(rects, sites, path, width, height):
    from _bootstrap import headless
    headless()
    import pygame
    pygame.init()
    surf = pygame.Surface((width, height))
    surf.fill((32, 40, 32))
    for (x, y, w, h) in rects:
        pygame.draw.rect(surf, (120, 120, 120), pygame.Rect(x, y, w, h))
    for s in sites:
        # Green marks a slot (capture geometrically impossible); amber a wall.
        line = (0, 230, 120) if s.kind == "slot" else (255, 200, 0)
        pygame.draw.line(surf, line, s.predator_side, s.bait, 2)
        if s.kind == "slot":
            pygame.draw.circle(surf, (0, 230, 120),
                               [int(v) for v in s.bait], 10, 2)
        pygame.draw.circle(surf, (80, 200, 255), [int(v) for v in s.bait], 6)
        pygame.draw.circle(surf, (255, 60, 60),
                           [int(v) for v in s.predator_side], 8)
    pygame.image.save(surf, path)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="Map to scan.")
    ap.add_argument("--real", action="store_true",
                    help="Build the actual seeded game map (slower, exact). "
                         "Without it the obstacles are drawn from the same "
                         "distribution but are not that seed's real map.")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--safety", default="safe", choices=sorted(PRESETS))
    ap.add_argument("--kinds", default="wall,slot",
                    help="Comma-separated: wall, slot, or both.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render", help="Save an annotated PNG of the map.")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    args = ap.parse_args()

    builder = _real_map if args.real else _sampled_map
    rects = builder(args.seed, args.width, args.height)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    sites = find_trap_sites(rects, args.width, args.height, args.safety,
                            args.limit, kinds)

    if args.json:
        print(json.dumps([s.as_dict() for s in sites], indent=2))
    else:
        n_slot = sum(s.kind == "slot" for s in sites)
        print(f"seed {args.seed}  safety={args.safety}  {len(sites)} site(s): "
              f"{len(sites) - n_slot} wall, {n_slot} slot\n")
        if sites:
            print(f"{'#':>2} {'kind':>5} {'bait':>15} {'predator':>15} "
                  f"{'thick/gap':>10} {'worst':>6} {'margin':>7} {'safe?':>6}")
            for i, s in enumerate(sites, 1):
                print(f"{i:>2} {s.kind:>5} "
                      f"{str(tuple(int(v) for v in s.bait)):>15} "
                      f"{str(tuple(int(v) for v in s.predator_side)):>15} "
                      f"{s.thickness:>10.1f} {s.worst_case_distance:>6.1f} "
                      f"{s.margin:>7.1f} {('YES' if s.guaranteed else '-'):>6}")
            print(f"\n'worst' is the closest the predator ever gets. It must stay "
                  f"above {TOUCH_DISTANCE:.0f} (capture) and, to keep the")
            print(f"predator attracted rather than wandering off, under the "
                  f"{PREDATOR_SMELL:.0f}-unit smell radius.")
            print("'safe? YES' means capture is geometrically impossible, "
                  "whatever the predator does.")
    if args.render:
        _render(rects, sites, args.render, args.width, args.height)
        print(f"\nwrote {args.render}")


if __name__ == "__main__":
    main()
