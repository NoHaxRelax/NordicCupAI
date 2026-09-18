"""Pure static-map selector for short predator-excluding gaps.

The selector uses arena dimensions and obstacle rectangles only.  A qualifying
site has the v5 short-site predator runup plus a radius-5.01 path entering the
opposite end of the same channel.  That second path is geometric access for a
future bait; it does not demonstrate a safe handoff around a live predator.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

Point = tuple[float, float]
Rect = tuple[float, float, float, float]


class SiteSelectionError(ValueError):
    pass


def _point(value: Sequence[float]) -> Point:
    return float(value[0]), float(value[1])


def _rect(value) -> Rect:
    if isinstance(value, Mapping):
        return tuple(float(value[k]) for k in ("x", "y", "width", "height"))
    return tuple(float(v) for v in value)


def _segment_rect(a: Point, b: Point, r: Rect) -> bool:
    x, y, w, h = r
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x), (dx, x + w - a[0]),
                 (-dy, a[1] - y), (dy, y + h - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


class _Geometry:
    def __init__(self, width: float, height: float, rects: Iterable):
        self.width, self.height = float(width), float(height)
        self.rects = [_rect(row) for row in rects]

    def free(self, p: Point, radius: float, ignore=()) -> bool:
        ignored = set(ignore)
        if not (radius <= p[0] <= self.width - radius
                and radius <= p[1] <= self.height - radius):
            return False
        return not any(i not in ignored
                       and x - radius < p[0] < x + w + radius
                       and y - radius < p[1] < y + h + radius
                       for i, (x, y, w, h) in enumerate(self.rects))

    def clear(self, a: Point, b: Point, radius: float, ignore=()) -> bool:
        ignored = set(ignore)
        if not self.free(a, radius, ignored) or not self.free(b, radius, ignored):
            return False
        return not any(i not in ignored and _segment_rect(
            a, b, (x - radius, y - radius, w + 2 * radius, h + 2 * radius))
            for i, (x, y, w, h) in enumerate(self.rects))


def _static_parts(static_map: Mapping) -> tuple[float, float, list[Rect]]:
    try:
        width, height = float(static_map["width"]), float(static_map["height"])
        rects = [_rect(row) for row in static_map["obstacles"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("static_map requires width, height, and obstacle rectangles") from exc
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError("static map dimensions must be finite and positive")
    return width, height, rects


def enumerate_sites(static_map: Mapping, *, min_gap: float = 10.1,
                    max_gap: float = 19.9, min_overlap: float = 20.0,
                    bait_depth: float = 5.0,
                    require_second_access: bool = True,
                    allow_offset_approach: bool = True) -> list[dict]:
    """Return compatible site dictionaries, best first.

    Arena boundaries participate whenever they are present in ``obstacles``;
    native maps represent them as indices 0--3.  ``second_access_clear`` means
    only that a radius-5.01 disc has a straight static path from 20 units beyond
    the other mouth to the bait goal.
    """
    width, height, rects = _static_parts(static_map)
    g = _Geometry(width, height, rects)
    def is_boundary(index):
        x, y, w, h = rects[index]
        return ((x <= 0 and w >= width) or (y <= 0 and h >= height)
                or (x + w >= width and w >= width)
                or (y + h >= height and h >= height))
    sites = []
    for i, a in enumerate(rects):
        for j, b in enumerate(rects):
            if i == j:
                continue
            for axis in (0, 1):
                ax, ay, aw, ah = a if axis == 0 else (a[1], a[0], a[3], a[2])
                bx, by, bw, bh = b if axis == 0 else (b[1], b[0], b[3], b[2])
                gap = bx - (ax + aw)
                low, high = max(ay, by), min(ay + ah, by + bh)
                overlap = high - low
                if not (min_gap <= gap <= max_gap and overlap >= min_overlap):
                    continue
                cross = ax + aw + gap / 2

                def xy(u, v):
                    return (u, v) if axis == 0 else (v, u)

                for sign in (1, -1):
                    mouth = xy(cross, low if sign == 1 else high)
                    inward = xy(0.0, float(sign))
                    add_depth = lambda p, d: (p[0] + inward[0] * d,
                                              p[1] + inward[1] * d)
                    goal = add_depth(mouth, bait_depth)
                    if not g.free(goal, 5.01):
                        continue

                    # The legacy lane is the channel centreline. Arena-wall
                    # pairs need a small cross-channel shift after the finite
                    # obstacle ends because the boundary wall continues.
                    offsets = [0.0]
                    if allow_offset_approach:
                        offsets += [sign * d for d in
                                    (2., 4., 6., 8., 10., 12., 15., 18., 22., 26.)
                                    for sign in (1., -1.)]
                    approach = None
                    cross_dir = (-inward[1], inward[0])
                    for offset in offsets:
                        shifted = lambda p: (p[0] + cross_dir[0] * offset,
                                             p[1] + cross_dir[1] * offset)
                        far = shifted(add_depth(mouth, -125.0))
                        hold = shifted(add_depth(mouth, -75.0))
                        if not (g.free(far, 11.01) and g.free(hold, 11.01)
                                and g.clear(far, hold, 11.01)):
                            continue
                        runups = [shifted(add_depth(mouth, -d))
                                  for d in (250., 225., 200., 180., 160., 145.)]
                        clear_runups = [r for r in runups if g.clear(far, r, 11.0)]
                        if not clear_runups:
                            continue
                        if not g.clear(hold, mouth, 11.0, ignore=(i, j)):
                            continue
                        approach = (far, hold, clear_runups[0], offset)
                        break
                    if approach is None:
                        continue
                    far, hold, runup, approach_offset = approach

                    # Enter from beyond the opposite overlap endpoint and pass
                    # down the complete channel to the depth-5 goal.
                    other_mouth = xy(cross, high if sign == 1 else low)
                    second_staging = add_depth(other_mouth, 20.0)
                    second_clear = g.clear(second_staging, goal, 5.01)
                    if require_second_access and not second_clear:
                        continue
                    margin = min(far[0], far[1], width - far[0], height - far[1])
                    boundary_indices = [k for k in (i, j) if is_boundary(k)]
                    site = dict(
                        mouth=mouth, inward=inward,
                        cross=(-inward[1], inward[0]), goal=goal,
                        far=far, hold=hold, gap=gap, overlap=overlap,
                        runup=runup, approach_lane_offset=approach_offset,
                        offset_approach=bool(approach_offset),
                        obstacle_indices=[i, j], axis=axis,
                        other_mouth=other_mouth,
                        replacement_entry=second_staging,
                        second_access_clear=second_clear,
                        second_access_radius=5.01,
                        bait_depth=bait_depth,
                        boundary_indices=boundary_indices,
                        geometric_replacement_access_only=True,
                    )
                    sites.append((overlap + .05 * margin, site))
    sites.sort(key=lambda row: row[0], reverse=True)
    return [{k: list(v) if isinstance(v, tuple) else v for k, v in site.items()}
            for _, site in sites]


def select_site(static_map: Mapping, **kwargs) -> dict:
    """Select the highest-scoring site; suitable as a policy constructor helper."""
    sites = enumerate_sites(static_map, **kwargs)
    if not sites:
        raise SiteSelectionError(
            "static map has no depth-5 short-gap site with clear predator runup "
            "and geometric bait access through the opposite mouth")
    return sites[0]
