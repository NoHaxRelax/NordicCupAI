"""Plain-tuple vector helpers and axis-aligned rectangles (engine convention: y grows down)."""
from __future__ import annotations

import math
from dataclasses import dataclass

TAU = 2 * math.pi


def wrap(a: float) -> float:
    return (a + math.pi) % TAU - math.pi


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, s):
    return (a[0] * s, a[1] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def norm(a):
    return math.hypot(a[0], a[1])


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n) if n > 1e-12 else (1.0, 0.0)


def perp(a):
    """Rotate 90 degrees counter-clockwise in engine coordinates."""
    return (-a[1], a[0])


def rot(a, theta):
    c, s = math.cos(theta), math.sin(theta)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)


def polar(theta, r=1.0):
    return (r * math.cos(theta), r * math.sin(theta))


def heading_of(a):
    return math.atan2(a[1], a[0])


def point_segment(p, a, b):
    v = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(p, a), v) / max(dot(v, v), 1e-12)))
    return dist(p, add(a, mul(v, t)))


def segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def edges(self):
        """The four edges in the engine's own orientation (min corner -> max corner)."""
        return [((self.x, self.y), (self.x2, self.y)),
                ((self.x2, self.y), (self.x2, self.y2)),
                ((self.x, self.y2), (self.x2, self.y2)),
                ((self.x, self.y), (self.x, self.y2))]

    def contains(self, p, radius=0.0):
        """Engine ``_in_obstacle`` test: strict inequalities on the inflated box."""
        return (self.x - radius < p[0] < self.x2 + radius) and (self.y - radius < p[1] < self.y2 + radius)

    def distance(self, p):
        dx = max(self.x - p[0], 0.0, p[0] - self.x2)
        dy = max(self.y - p[1], 0.0, p[1] - self.y2)
        return math.hypot(dx, dy)

    def segment_hits(self, a, b, radius=0.0):
        """True when the segment a-b passes through the box inflated by ``radius``."""
        if self.contains(a, radius) or self.contains(b, radius):
            return True
        x1, y1, x2, y2 = self.x - radius, self.y - radius, self.x2 + radius, self.y2 + radius
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i in range(4):
            if segments_cross(a, b, corners[i], corners[(i + 1) % 4]):
                return True
        return False

    def inflated(self, r):
        return Rect(self.x - r, self.y - r, self.w + 2 * r, self.h + 2 * r)


def any_contains(rects, p, radius=0.0):
    return any(r.contains(p, radius) for r in rects)


def free_point(p, radius, rects, width, height):
    """A creature of ``radius`` can stand at p: inside the arena and outside every box."""
    if not (radius <= p[0] <= width - radius and radius <= p[1] <= height - radius):
        return False
    return not any_contains(rects, p, radius)


def los_clear(a, b, rects):
    """Line of sight between two points: no rectangle intersects the segment."""
    return not any(r.segment_hits(a, b) for r in rects)


def path_clear(a, b, radius, rects):
    """A creature of ``radius`` can walk straight from a to b without touching a box."""
    return not any(r.segment_hits(a, b, radius) for r in rects)
