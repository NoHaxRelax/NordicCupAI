"""Trap sites from rectangles: thin walls and narrow gaps.

Wall site: rectangle 30-35.5 thick and >= 70 long. The predator is held against
the *front* face; the holder stands 5.1 off the midpoint of the *back* face.
Gap site: two rectangles whose facing sides are 11-19 apart over >= 55 units of
overlap. An agent (radius 5) fits, a predator (radius 10) does not. The bait
stands ``depth`` inside one mouth; predators pile up outside that mouth.

All numbers come from the tested ranges in Oscar's and Lucas's research; the
detector is deliberately conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Rect, add, mul, sub, dot, unit, perp, dist, free_point, path_clear
from .world import AGENT_RADIUS, PREDATOR_RADIUS

WALL_THIN = (30.0, 35.5)
WALL_MIN_LONG = 70.0
GAP_RANGE = (10.5, 19.5)  # strict engine tests: an agent (radius 5) passes above 10, a predator (10) below 20
GAP_MIN_OVERLAP = 30.0    # passage length; the bait must be > 15 from any point either mouth lets the predator reach
HOLDER_OFF = 5.1          # holder center from the back face
FRONT_OFF = 5.1           # guide sacrifice point from the front face
CORRIDOR = 150.0          # straight approach distance in front of the trap
GAP_CORRIDOR_MIN = 100.0  # gap sites: shortest acceptable straight approach
GAP_DEPTH = 5.0           # minimum bait depth inside the mouth
GAP_FLYBY_OUT = 14.0      # where a guide leaves the axis in front of a staffed mouth
KILL_MARGIN = 1.5         # extra distance beyond the 15 kill radius


def gap_reach(width):
    """How far outside a mouth of this width the predator's center must stay (its radius is 10
    and the mouth corners block it): sqrt(10^2 - (w/2)^2)."""
    import math
    return math.sqrt(max(0.0, 100.0 - (width / 2) ** 2))


def gap_depth(width, length):
    """Bait depth from the near mouth that keeps it out of the kill radius from both mouths,
    or None if the passage is too short for that."""
    reach = gap_reach(width)
    lo = max(GAP_DEPTH, 15.0 + KILL_MARGIN - reach)
    hi = length + reach - 15.0 - KILL_MARGIN
    if lo > hi:
        return None
    return lo


@dataclass
class Site:
    kind: str                 # 'wall' or 'gap'
    key: str
    rect: Rect | None         # the wall (wall sites)
    rects: tuple              # the two obstacles (gap sites)
    axis: tuple               # unit vector along the wall length / passage
    normal: tuple             # unit vector from the trap toward the predator (front) side
    front_mid: tuple          # midpoint of the front face / the mouth center line at the mouth
    holder: tuple             # holder / bait position
    front: tuple              # guide's final point on the front side
    corridor_start: tuple     # where the straight approach begins
    successor: tuple          # staging point for a replacement, behind the holder
    guard: tuple | None       # guard station, further behind (wall only)
    thickness: float
    length: float
    lateral: float            # half-width of the usable front zone along the axis
    far_mouth: tuple | None = None    # gap: the other entrance (successor route) or None if closed
    far_mouth_open: bool = True
    score: float = 0.0
    extra: dict = field(default_factory=dict)

    def in_front_zone(self, p, margin=0.0):
        """Point lies on the predator side within the held zone around the front face."""
        d = sub(p, self.front_mid)
        along = abs(dot(d, self.axis))
        out = dot(d, self.normal)
        return -5 <= out <= 40 + margin and along <= self.lateral + margin

    def held_center(self):
        return add(self.front_mid, mul(self.normal, PREDATOR_RADIUS))


def _is_boundary(r: Rect, width, height):
    return r.x <= 0 or r.y <= 0 or r.x2 >= width or r.y2 >= height


def _corridor_clear(start, normal, length, others, width, height, radius=PREDATOR_RADIUS, band=25.0):
    """The strip from ``start`` outward along ``normal`` is free for the predator."""
    tangent = perp(normal)
    for lat in (-band, 0.0, band):
        a = add(start, mul(tangent, lat))
        b = add(a, mul(normal, length))
        if not free_point(a, radius, others, width, height) or not free_point(b, radius, others, width, height):
            return False
        if not path_clear(a, b, radius, others):
            return False
    return True


def find_wall_sites(rects, width, height):
    sites = []
    for i, r in enumerate(rects):
        if _is_boundary(r, width, height):
            continue
        thin, long = min(r.w, r.h), max(r.w, r.h)
        if not (WALL_THIN[0] <= thin <= WALL_THIN[1] and long >= WALL_MIN_LONG):
            continue
        axis = (1.0, 0.0) if r.w >= r.h else (0.0, 1.0)
        others = [o for j, o in enumerate(rects) if j != i]
        for sign in (1.0, -1.0):
            normal = mul(perp(axis), sign)
            c = r.center
            front_mid = add(c, mul(normal, thin / 2))
            back_mid = sub(c, mul(normal, thin / 2))
            holder = sub(back_mid, mul(normal, HOLDER_OFF))
            front = add(front_mid, mul(normal, FRONT_OFF))
            successor = sub(holder, mul(normal, 12.0))
            guard = sub(holder, mul(normal, 45.0))
            corridor_start = add(front_mid, mul(normal, CORRIDOR))
            ok = (free_point(holder, AGENT_RADIUS + 0.5, others, width, height)
                  and free_point(successor, AGENT_RADIUS + 0.5, others, width, height)
                  and free_point(add(front_mid, mul(normal, PREDATOR_RADIUS + 0.5)), PREDATOR_RADIUS, others, width, height)
                  and _corridor_clear(add(front_mid, mul(normal, PREDATOR_RADIUS + 0.5)), normal, CORRIDOR, others, width, height, band=20.0))
            if not ok:
                continue
            run_in_clear = _corridor_clear(add(front_mid, mul(normal, CORRIDOR)), normal, 250.0, others, width, height, band=20.0)
            # the predator must be able to slide a little along the front face without hitting neighbours
            tangent = axis
            lateral = long / 2 - 20.0
            slide_ok = all(free_point(add(add(front_mid, mul(normal, PREDATOR_RADIUS + 0.5)), mul(tangent, s)),
                                      PREDATOR_RADIUS, others, width, height) for s in (-lateral, lateral))
            if not slide_ok:
                continue
            guard_ok = free_point(guard, AGENT_RADIUS + 0.5, others, width, height)
            sites.append(Site(kind='wall', key=f'wall{i}{"+" if sign > 0 else "-"}', rect=r, rects=(r,), axis=axis,
                              normal=normal, front_mid=front_mid, holder=holder, front=front,
                              corridor_start=corridor_start, successor=successor, guard=guard if guard_ok else None,
                              thickness=thin, length=long, lateral=lateral, score=0.0 if run_in_clear else 80.0,
                              extra=dict(run_in_clear=run_in_clear)))
    return sites


def find_gap_sites(rects, width, height, depth=None, min_overlap=GAP_MIN_OVERLAP):
    """Narrow passages between two rectangles. One site per usable mouth; ``depth`` overrides the
    width-dependent bait depth. Sites carry a score (lower is better): short passages, closed far
    mouths, extreme widths and short approaches are penalised."""
    sites = []
    n = len(rects)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = rects[i], rects[j]
            for orient in ('x', 'y'):
                if orient == 'x':
                    gap = b.x - a.x2
                    lo, hi = max(a.y, b.y), min(a.y2, b.y2)
                else:
                    gap = b.y - a.y2
                    lo, hi = max(a.x, b.x), min(a.x2, b.x2)
                if not (GAP_RANGE[0] <= gap <= GAP_RANGE[1]) or hi - lo < min_overlap:
                    continue
                length = hi - lo
                d = depth if depth is not None else gap_depth(gap, length)
                if d is None:
                    continue
                center_line = (a.x2 + gap / 2) if orient == 'x' else (a.y2 + gap / 2)
                axis = (0.0, 1.0) if orient == 'x' else (1.0, 0.0)     # along the passage
                others = [o for k, o in enumerate(rects) if k not in (i, j)]
                mouths = []
                for end, sign in ((lo, -1.0), (hi, 1.0)):
                    normal = mul(axis, sign)                              # outward at this mouth
                    mouth = (center_line, end) if orient == 'x' else (end, center_line)
                    bait = sub(mouth, mul(normal, d))
                    start = add(mouth, mul(normal, PREDATOR_RADIUS + 0.5))
                    corridor = 0.0
                    for length_out in (CORRIDOR + 100.0, CORRIDOR + 50.0, CORRIDOR, GAP_CORRIDOR_MIN):
                        if _corridor_clear(start, normal, length_out, others, width, height, band=0.0):
                            corridor = length_out
                            break
                    entry_ok = (free_point(bait, AGENT_RADIUS, rects, width, height)
                                and path_clear(add(mouth, mul(normal, 40.0)), bait, AGENT_RADIUS, rects))
                    # the passage itself must be walkable for an agent from this mouth to the bait
                    mouths.append(dict(mouth=mouth, normal=normal, bait=bait, corridor=corridor, entry_ok=entry_ok))
                for m, other in ((mouths[0], mouths[1]), (mouths[1], mouths[0])):
                    if not (m['corridor'] > 0 and m['entry_ok']):
                        continue
                    normal = m['normal']
                    far_open = other['entry_ok']
                    # a guide that flies past a staffed mouth needs a clear run along the face on at
                    # least one side of the flyby point
                    fly = add(m['mouth'], mul(normal, GAP_FLYBY_OUT))
                    tng = perp(normal)
                    flyby_clear = any(free_point(add(fly, mul(tng, sgn * 60.0)), AGENT_RADIUS, rects, width, height)
                                      and path_clear(fly, add(fly, mul(tng, sgn * 60.0)), AGENT_RADIUS, rects) for sgn in (1.0, -1.0))
                    boundary = _is_boundary(a, width, height) or _is_boundary(b, width, height)
                    score = (max(0.0, 60.0 - length) + (0.0 if far_open else 40.0) + abs(gap - 14.0) * 3.0
                             + (CORRIDOR + 100.0 - m['corridor']) * 0.3 + (0.0 if flyby_clear else 60.0)
                             + (80.0 if boundary else 0.0))       # a passage along the map edge pins the predator
                    key = f'gap{i}-{j}{orient}{"+" if normal == mul(axis, 1.0) else "-"}'
                    # replacement baits stage outside the far mouth, off the axis, so the old bait
                    # can walk out past them to the exit point on the other side
                    back = mul(normal, -1.0)
                    tangent = perp(normal)
                    stage = exit_pt = None
                    for side in (tangent, mul(tangent, -1.0)):
                        cand = add(add(other['mouth'], mul(back, 22.0)), mul(side, 16.0))
                        cand_exit = add(add(other['mouth'], mul(back, 45.0)), mul(side, -16.0))
                        if free_point(cand, AGENT_RADIUS + 1.0, rects, width, height) and \
                                free_point(cand_exit, AGENT_RADIUS + 1.0, rects, width, height):
                            stage, exit_pt = cand, cand_exit
                            break
                    if stage is None:
                        stage = add(other['mouth'], mul(back, 22.0))
                        exit_pt = add(other['mouth'], mul(back, 60.0))
                    sites.append(Site(kind='gap', key=key, rect=None, rects=(a, b), axis=mul(normal, -1.0),
                                      normal=normal, front_mid=m['mouth'], holder=m['bait'],
                                      front=add(m['mouth'], mul(normal, GAP_FLYBY_OUT)),
                                      corridor_start=add(m['mouth'], mul(normal, m['corridor'])),
                                      successor=stage, guard=None,
                                      thickness=gap, length=length, lateral=gap / 2 + PREDATOR_RADIUS,
                                      far_mouth=other['mouth'], far_mouth_open=far_open, score=score,
                                      extra=dict(depth=d, corridor=m['corridor'], exit=exit_pt, flyby_clear=flyby_clear)))
    # de-duplicate the (i,j)/(j,i) symmetric hits
    seen = {}
    for s in sites:
        k = (round(s.holder[0], 1), round(s.holder[1], 1))
        seen.setdefault(k, s)
    return list(seen.values())


def find_sites(rects, width, height, kinds=('wall', 'gap')):
    out = []
    if 'wall' in kinds:
        out += find_wall_sites(rects, width, height)
    if 'gap' in kinds:
        out += find_gap_sites(rects, width, height)
    return out
