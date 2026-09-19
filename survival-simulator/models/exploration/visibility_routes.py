"""Agent-width fallback routes around observed wall-buffer corners.

No grid spacing is added to clearance. Only supplied observed geometry is used.
"""
import heapq
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


class VisibilityRoutes:
    def __init__(self, edges, clearance, inside):
        self.walls = unary_union([LineString(e).buffer(clearance, cap_style=3, join_style=2)
                                  for e in edges if np.linalg.norm(e[1]-e[0]) > 1e-8])
        # Vertices lie just outside collision padding, so tangent paths are safe.
        envelope = self.walls.buffer(.02, join_style=2)
        points = set()
        for polygon in shapely.get_parts(envelope):
            if polygon.geom_type != 'Polygon': continue
            for ring in [polygon.exterior, *polygon.interiors]:
                points.update(tuple(p) for p in ring.coords[:-1] if inside(np.array(p)))
        self.points = np.array(sorted(points), dtype=float).reshape((-1, 2))
        self.links = {}
        shapely.prepare(self.walls)

    def visible(self, point):
        if not len(self.points): return np.array([], dtype=int)
        segments = np.stack((np.broadcast_to(point, self.points.shape), self.points), axis=1)
        return np.flatnonzero(~shapely.intersects(self.walls, shapely.linestrings(segments)))

    def plan(self, start, end, clear):
        if not len(self.points): return []
        starts = self.visible(start)
        # Permit escaping a small localization-induced clearance violation,
        # using the existing exact wall-crossing/clearance check.
        if self.walls.intersects(Point(start)):
            starts = [i for i, point in enumerate(self.points) if clear(start, point)]
        goals = self.visible(end)
        if self.walls.intersects(Point(end)):
            goals = [i for i, point in enumerate(self.points) if clear(point, end)]
        terminal = {int(i): float(np.linalg.norm(self.points[i]-end)) for i in goals}
        costs = {int(i): float(np.linalg.norm(self.points[i]-start)) for i in starts}
        previous = {i: None for i in costs}
        queue = [(g+float(np.linalg.norm(self.points[i]-end)), g, i) for i, g in costs.items()]
        heapq.heapify(queue)
        best, winner = math.inf, None
        while queue:
            f, g, i = heapq.heappop(queue)
            if g != costs[i]: continue
            if f >= best: break
            if i in terminal and g+terminal[i] < best:
                best, winner = g+terminal[i], i
            if i not in self.links:
                self.links[i] = [(int(j), float(np.linalg.norm(self.points[j]-self.points[i])))
                                 for j in self.visible(self.points[i]) if j != i]
            for j, distance in self.links[i]:
                candidate = g+distance
                if candidate < costs.get(j, math.inf):
                    costs[j], previous[j] = candidate, i
                    heapq.heappush(queue, (candidate+float(np.linalg.norm(self.points[j]-end)), candidate, j))
        if winner is None: return []
        path = [np.asarray(end).copy()]
        while winner is not None:
            path.append(self.points[winner].copy())
            winner = previous[winner]
        path.reverse()
        # Validate against the controller's wall rule before accepting a route.
        if not all(clear(a, b) for a, b in zip([start]+path, path)): return []
        return path
