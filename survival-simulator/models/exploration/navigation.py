"""Persistent routes around observed walls, shared by harvest and survey goals.

Unknown terrain is traversable. The bounded grid is only a route planner; the
ordinary controller remains responsible for immediate collision avoidance.
"""

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from models.exploration.exploration import _path_clear
from models.exploration.visibility_routes import VisibilityRoutes


@dataclass
class NavigationResult:
    waypoint: np.ndarray | None
    blocked: bool
    remaining: float
    status: str


@dataclass
class Route:
    destination: np.ndarray
    points: list[np.ndarray] = field(default_factory=list)
    best_remaining: float = math.inf
    progress_at: float = 0.
    retries: int = 0
    revision: int = -1
    status: str = "new"
    remaining: float = math.inf


class Navigator:
    def __init__(self, cell_size=16., max_cells=20000, clearance=6., stall_seconds=5.):
        self.cell_size = cell_size
        self.max_cells = max_cells
        self.clearance = clearance
        self.stall_seconds = stall_seconds
        self.reset()

    def reset(self):
        self.frame = None
        self.edge_key = None
        self.edges = np.empty((0, 2, 2))
        self.edge_min = self.edge_max = np.empty((0, 2))
        self.size = self.width = self.height = None
        self.routes = {}
        self.revision = 0
        self.next_geometry = 0.
        self.blocked = None
        self.origin = self.extent = self.pitch = None
        self.visibility = None

    def update(self, group, now):
        size = group.world_size
        if size is None and group.known_width is not None and group.known_height is not None:
            size = (group.known_width, group.known_height)
        frame = (group.group_id, group.frame_revision, None if size is None else tuple(size))
        if frame != self.frame:
            self.reset()
            self.frame = frame
        self.size, self.width, self.height = size, group.known_width, group.known_height
        key = (getattr(group, "_edge_revision", 0), len(group.edges))
        if key == self.edge_key or now < self.next_geometry:
            return
        self.edge_key = key
        self.next_geometry = now + .5
        self.edges = np.array([[e.start, e.end] for e in group.edges], dtype=float).reshape((-1, 2, 2))
        self.edge_min, self.edge_max = self.edges.min(axis=1), self.edges.max(axis=1)
        self.revision += 1
        self.blocked = None
        self.visibility = None

    def prune(self, living_ids):
        living = set(living_ids)
        self.routes = {key: route for key, route in self.routes.items() if key in living}

    def release(self, agent_id):
        """Forget a completed/rejected goal so a later retry starts fresh."""
        self.routes.pop(agent_id, None)

    def _inside(self, point):
        if self.size is not None:
            return bool(np.all(point >= 0) and np.all(point <= np.asarray(self.size)))
        return ((self.width is None or 0 <= point[0] <= self.width)
                and (self.height is None or 0 <= point[1] <= self.height))

    def _clear(self, start, end):
        if not self._inside(end):
            return False
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1e-8 or not len(self.edges):
            return True
        low, high = np.minimum(start, end) - self.clearance, np.maximum(start, end) + self.clearance
        nearby = np.all(self.edge_max >= low, axis=1) & np.all(self.edge_min <= high, axis=1)
        # _path_clear permits moving out of an existing clearance violation,
        # but never crossing a wall. Pose noise must not trap agents in padding.
        return _path_clear(math.atan2(delta[1], delta[0]), distance,
                           self.edges[nearby] - start, self.clearance)

    def _grid(self, start, end):
        if self.blocked is not None:
            if np.all(np.minimum(start, end) >= self.origin) and np.all(np.maximum(start, end) < self.origin + self.extent):
                return
        if self.size is not None:
            low, high = np.zeros(2), np.asarray(self.size, dtype=float)
        else:
            # An observed envelope also permits useful routes before both map
            # dimensions are known, without inventing unseen world boundaries.
            points = np.concatenate((np.array([start, end]), self.edges.reshape((-1, 2))))
            low, high = points.min(axis=0) - 100., points.max(axis=0) + 100.
            if self.width is not None:
                low[0], high[0] = 0., self.width
            if self.height is not None:
                low[1], high[1] = 0., self.height
        extent = np.maximum(1., high - low)
        spacing = self.cell_size
        shape = np.maximum(1, np.ceil(extent / spacing).astype(int))
        while int(np.prod(shape)) > self.max_cells:
            spacing *= 1.2
            shape = np.maximum(1, np.ceil(extent / spacing).astype(int))
        self.origin, self.extent, self.pitch = low, extent, extent / shape
        self.blocked = np.zeros((shape[1], shape[0]), dtype=bool)
        # Reserve enough clearance that adjacent free cell centers cannot cut
        # across a thin wall. Segment checks smooth the resulting coarse path.
        padding = self.clearance + float(np.linalg.norm(self.pitch)) / 2
        for edge in self.edges:
            a, b = edge
            lower = np.maximum(0, np.floor((edge.min(axis=0) - padding - low) / self.pitch).astype(int))
            upper = np.minimum(shape - 1, np.floor((edge.max(axis=0) + padding - low) / self.pitch).astype(int))
            if np.any(lower > upper):
                continue
            xx, yy = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
            centers = low + (np.stack((xx, yy), axis=-1) + .5) * self.pitch
            direction = b - a
            length = float(direction @ direction)
            t = np.zeros(xx.shape) if length == 0 else np.clip(((centers - a) @ direction) / length, 0, 1)
            occupied = np.linalg.norm(centers - a - t[..., None] * direction, axis=-1) <= padding
            self.blocked[yy, xx] |= occupied

    def _center(self, cell):
        y, x = cell
        return self.origin + (np.array([x, y]) + .5) * self.pitch

    def _endpoint(self, point, *, escaping=False):
        xy = np.floor((point - self.origin) / self.pitch).astype(int)
        height, width = self.blocked.shape
        # A point close to a wall may occupy an inflated grid cell. Attach it
        # to a nearby free cell on the same side, never through the obstacle.
        candidates = []
        for y in range(max(0, xy[1] - 3), min(height, xy[1] + 4)):
            for x in range(max(0, xy[0] - 3), min(width, xy[0] + 4)):
                if not self.blocked[y, x]:
                    center = self._center((y, x))
                    distance = float(np.linalg.norm(center - point))
                    candidates.append((distance, y, x))
        for _, y, x in sorted(candidates):
            center = self._center((y, x))
            if self._clear(point, center) if escaping else self._clear(center, point):
                return y, x
        return None

    def _plan(self, start, end):
        route = self._grid_plan(start, end)
        if route or not self._inside(end): return route
        if self.visibility is None:
            self.visibility = VisibilityRoutes(self.edges, self.clearance, self._inside)
        return self.visibility.plan(start, end, self._clear)

    def _grid_plan(self, start, end):
        if self._clear(start, end):
            return [end.copy()]
        if not self._inside(end):
            return []
        self._grid(start, end)
        first, last = self._endpoint(start, escaping=True), self._endpoint(end)
        if first is None or last is None:
            return []
        height, width = self.blocked.shape
        frontier = [(0., 0., first)]
        costs, previous = {first: 0.}, {}
        moves = [(dy, dx, math.hypot(dx * self.pitch[0], dy * self.pitch[1]))
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
        while frontier:
            _, cost, cell = heapq.heappop(frontier)
            if cost > costs[cell]:
                continue
            if cell == last:
                cells = [cell]
                while cells[-1] != first:
                    cells.append(previous[cells[-1]])
                path = [self._center(c) for c in reversed(cells)] + [end.copy()]
                # Retain only the farthest visible waypoint at each turn.
                smooth, point, index = [], start, 0
                while index < len(path):
                    furthest = index
                    for candidate in range(index + 1, len(path)):
                        if self._clear(point, path[candidate]):
                            furthest = candidate
                    smooth.append(path[furthest])
                    point, index = path[furthest], furthest + 1
                return smooth
            y, x = cell
            for dy, dx, step in moves:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width) or self.blocked[ny, nx]:
                    continue
                if dx and dy and (self.blocked[y, nx] or self.blocked[ny, x]):
                    continue
                neighbor, proposed = (ny, nx), cost + step
                if proposed >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor], previous[neighbor] = proposed, cell
                heuristic = math.hypot((last[1] - nx) * self.pitch[0], (last[0] - ny) * self.pitch[1])
                heapq.heappush(frontier, (proposed + heuristic, proposed, neighbor))
        return []

    @staticmethod
    def _remaining(position, points):
        if not points:
            return math.inf
        path = np.vstack((position, points))
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def steer(self, agent_id, position, destination, now, waiting=False):
        position, destination = np.asarray(position, dtype=float), np.asarray(destination, dtype=float)
        route = self.routes.get(agent_id)
        if route is None or np.linalg.norm(destination - route.destination) > 5.:
            route = self.routes[agent_id] = Route(destination.copy(), progress_at=now)
        if waiting:
            route.progress_at = now
            route.status = "waiting"
            route.remaining = float(np.linalg.norm(destination - position))
            return NavigationResult(None, False, route.remaining, route.status)
        if np.linalg.norm(route.destination - position) <= 4.:
            route.status, route.remaining, route.progress_at = "arrived", 0., now
            return NavigationResult(route.destination.copy(), False, 0., route.status)
        changed = route.revision != self.revision
        if route.status == "blocked":
            # Candidates may continue gathering after a failed bait-route probe.
            # Do not permanently cache failure from their old location/map.
            if not changed and now-route.progress_at < 3.:
                return NavigationResult(None, True, route.remaining, route.status)
            route.points = []
            route.retries = 0
            route.progress_at = now
            route.status = "retrying"
        if changed and route.points:
            path = [position] + route.points
            if not all(self._clear(a, b) for a, b in zip(path, path[1:])):
                route.points = []
        if not route.points:
            route.points = self._plan(position, route.destination)
            # A newly observed wall changes the route length, not whether the
            # agent has made progress. Repeated geometry updates must not keep
            # a stationary agent's stall timer alive indefinitely.
            route.best_remaining = self._remaining(position, route.points)
            if route.revision < 0:
                route.progress_at = now
        route.revision = self.revision
        if not route.points:
            route.status = "blocked"
            return NavigationResult(None, True, math.inf, route.status)
        # Skip passed waypoints only when the next segment is still clear.
        while len(route.points) > 1 and np.linalg.norm(position - route.points[0]) < 8.:
            if not self._clear(position, route.points[1]):
                break
            route.points.pop(0)
        remaining = self._remaining(position, route.points)
        if remaining < route.best_remaining - 3.:
            route.best_remaining, route.progress_at = remaining, now
        route.status = "following"
        if now - route.progress_at >= self.stall_seconds:
            if route.retries:
                route.status, route.remaining = "blocked", remaining
                return NavigationResult(None, True, remaining, route.status)
            route.retries += 1
            route.points = self._plan(position, route.destination)
            route.best_remaining = self._remaining(position, route.points)
            route.progress_at, route.status = now, "replanned"
            if not route.points:
                route.status = "blocked"
                return NavigationResult(None, True, math.inf, route.status)
            remaining = route.best_remaining
        route.remaining = remaining
        return NavigationResult(route.points[0].copy(), False, remaining, route.status)

    def snapshot(self):
        return {agent_id: {"destination": route.destination.tolist(),
                           "waypoints": [p.tolist() for p in route.points],
                           "remaining": route.remaining if math.isfinite(route.remaining) else None,
                           "status": route.status, "replans": route.retries}
                for agent_id, route in self.routes.items()}
