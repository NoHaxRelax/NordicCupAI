"""Grid A* around rectangles with clearance, plus line-of-sight shortcutting.

The engine slides creatures along obstacles instead of stopping them, so exact
paths are not critical; what matters is not walking into dead ends and keeping
enough clearance for a predator (radius 10) that follows a guide (radius 5).
"""
from __future__ import annotations

import heapq
import math

from .geometry import Rect, dist, path_clear, free_point

CELL = 10.0


class Grid:
    def __init__(self, rects, width, height, radius, cell=CELL):
        self.rects = rects
        self.width, self.height, self.radius, self.cell = width, height, radius, cell
        self.nx, self.ny = int(width // cell) + 1, int(height // cell) + 1
        inflated = [r.inflated(radius) for r in rects]
        self.blocked = bytearray(self.nx * self.ny)
        for iy in range(self.ny):
            y = iy * cell
            for ix in range(self.nx):
                x = ix * cell
                if not (radius <= x <= width - radius and radius <= y <= height - radius):
                    self.blocked[iy * self.nx + ix] = 1
                    continue
                for r in inflated:
                    if r.x < x < r.x2 and r.y < y < r.y2:
                        self.blocked[iy * self.nx + ix] = 1
                        break

    def cell_of(self, p):
        return (int(round(p[0] / self.cell)), int(round(p[1] / self.cell)))

    def point_of(self, c):
        return (c[0] * self.cell, c[1] * self.cell)

    def is_blocked(self, c):
        if not (0 <= c[0] < self.nx and 0 <= c[1] < self.ny):
            return True
        return bool(self.blocked[c[1] * self.nx + c[0]])

    def nearest_free(self, c, limit=6):
        if not self.is_blocked(c):
            return c
        best = None
        for r in range(1, limit + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    for cand in ((c[0] + dx, c[1] + dy), (c[0] + dy, c[1] + dx)):
                        if not self.is_blocked(cand):
                            d = dx * dx + dy * dy
                            if best is None or d < best[0]:
                                best = (d, cand)
            if best:
                return best[1]
        return None


_grids: dict = {}


def grid_for(rects, width, height, radius):
    key = (id(rects), len(rects), width, height, round(radius, 2))
    g = _grids.get(key)
    if g is None:
        if len(_grids) > 16:
            _grids.clear()
        g = _grids[key] = Grid(rects, width, height, radius)
    return g


def astar(grid: Grid, start, goal, avoid=(), max_nodes=40000):
    """Cells to avoid: list of (center, radius) circles (soft-blocked, high cost)."""
    s = grid.nearest_free(grid.cell_of(start))
    g = grid.nearest_free(grid.cell_of(goal))
    if s is None or g is None:
        return None
    avoid_cells = []
    for (c, r) in avoid:
        avoid_cells.append((c, r))

    def penalty(cell):
        if not avoid_cells:
            return 0.0
        p = grid.point_of(cell)
        pen = 0.0
        for c, r in avoid_cells:
            d = dist(p, c)
            if d < r:
                pen += 50.0 * (1 - d / r)
        return pen

    def h(c):
        return math.hypot(c[0] - g[0], c[1] - g[1]) * grid.cell
    open_heap = [(h(s), 0.0, s)]
    came = {s: None}
    cost = {s: 0.0}
    nodes = 0
    while open_heap:
        _, c0, cur = heapq.heappop(open_heap)
        if cur == g:
            break
        if c0 > cost.get(cur, 1e18):
            continue
        nodes += 1
        if nodes > max_nodes:
            return None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not dx and not dy:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if grid.is_blocked(nb):
                    continue
                if dx and dy and (grid.is_blocked((cur[0] + dx, cur[1])) or grid.is_blocked((cur[0], cur[1] + dy))):
                    continue   # no corner cutting through blocked cells
                step = grid.cell * (1.41421356 if dx and dy else 1.0)
                nc = c0 + step + penalty(nb)
                if nc < cost.get(nb, 1e18):
                    cost[nb] = nc
                    came[nb] = cur
                    heapq.heappush(open_heap, (nc + h(nb), nc, nb))
    if g not in came:
        return None
    cells = []
    c = g
    while c is not None:
        cells.append(c)
        c = came[c]
    cells.reverse()
    pts = [start] + [grid.point_of(c) for c in cells[1:-1]] + [goal]
    return pts


def shortcut(pts, rects, radius, avoid=()):
    """Greedy line-of-sight smoothing that respects clearance and avoid circles."""
    if not pts or len(pts) < 3:
        return pts
    def ok(a, b):
        if not path_clear(a, b, radius, rects):
            return False
        for c, r in avoid:
            # sample the segment against the circle
            n = max(2, int(dist(a, b) // 10))
            for k in range(n + 1):
                t = k / n
                p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                if dist(p, c) < r:
                    return False
        return True
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not ok(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan(rects, width, height, start, goal, radius=6.0, avoid=()):
    """Waypoints from start to goal (both included). Straight line when clear."""
    if path_clear(start, goal, radius, rects) and not any(
            dist(p, c) < r for c, r in avoid for p in (start, goal)) and not avoid:
        return [start, goal]
    grid = grid_for(rects, width, height, radius)
    pts = astar(grid, start, goal, avoid)
    if pts is None:
        return None
    return shortcut(pts, rects, radius, avoid)


def path_length(pts):
    return sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) if pts else 0.0
