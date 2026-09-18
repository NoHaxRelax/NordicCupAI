"""Persistent routes around observed walls, shared by harvest and survey goals.

Unknown terrain is traversable. The bounded grid is only a route planner; the
ordinary controller remains responsible for immediate collision avoidance.
"""

from dataclasses import dataclass, field
import heapq
import math

import numpy as np

from src.utils.controllers.exploration import _path_clear


@dataclass
class NavigationResult:
    waypoint: np.ndarray | None
    blocked: bool
    remaining: float
    status: str


@dataclass
class Route:
    destination: np.ndarray
    arrival_radius: float = 0.
    points: list[np.ndarray] = field(default_factory=list)
    best_remaining: float = math.inf
    progress_at: float = 0.
    retries: int = 0
    revision: int = -1
    status: str = "new"
    remaining: float = math.inf
    paused_at: float | None = None


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
        self.connections = {}
        self.portals = None

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
        self.connections.clear()
        self.portals = None

    def prune(self, living_ids):
        living = set(living_ids)
        self.routes = {key: route for key, route in self.routes.items() if key in living}

    def release(self, agent_id):
        """Forget a completed/rejected goal so a later retry starts fresh."""
        self.routes.pop(agent_id, None)

    def pause(self, agent_id, now):
        """Exclude deliberate rest from the stall clock, retaining its history.

        Accumulated time without progress is preserved. Repeated short scouting
        windows can therefore still expose a stuck route across longer rests.
        Call on each resting tick; the next steer also excludes the final
        paused interval before resuming.
        """
        route = self.routes.get(agent_id)
        if route is None:
            return
        if route.paused_at is not None:
            route.progress_at += max(0., now - route.paused_at)
        route.paused_at = now
        if route.status != "blocked":
            route.status = "paused"

    def _inside(self, point):
        if self.size is not None:
            return bool(np.all(point >= 0) and np.all(point <= np.asarray(self.size)))
        return ((self.width is None or 0 <= point[0] <= self.width)
                and (self.height is None or 0 <= point[1] <= self.height))

    def _clear(self, start, end, clearance=None):
        if not self._inside(end):
            return False
        margin = self.clearance if clearance is None else clearance
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1e-8 or not len(self.edges):
            return True
        low, high = np.minimum(start, end) - margin, np.maximum(start, end) + margin
        nearby = np.all(self.edge_max >= low, axis=1) & np.all(self.edge_min <= high, axis=1)
        # _path_clear permits moving out of an existing clearance violation,
        # but never crossing a wall. Pose noise must not trap agents in padding.
        return _path_clear(math.atan2(delta[1], delta[0]), distance,
                           self.edges[nearby] - start, margin)

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
        self.connections.clear()
        # Nodes need physical agent clearance. Inflating by another half-cell
        # diagonal erases usable gaps and can force enormous detours. Each
        # connection is checked exactly below, including thin-wall crossings.
        padding = self.clearance
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

    def _connection_clear(self, first, second):
        key = tuple(sorted((first, second)))
        if key not in self.connections:
            self.connections[key] = self._clear(self._center(first), self._center(second))
        return self.connections[key]

    def _short_passage(self, start, end):
        """Check a bounded set of exact two-segment routes near wall ends.

        An opening may lie between coarse grid rows. Public wall endpoints
        supply useful passage candidates without allocating a finer full map.
        Cached candidates are cheap; at most 96 are checked for each route.
        """
        if self.portals is None:
            candidates = []
            margin = self.clearance + .5
            for edge in self.edges:
                along = edge[1] - edge[0]
                length = float(np.linalg.norm(along))
                if length < 1e-8:
                    continue
                along /= length
                normal = np.array([-along[1], along[0]])
                for end_point in edge:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        candidates.append(end_point + margin * (dx * along + dy * normal))
            self.portals = (np.unique(np.round(candidates, 8), axis=0)
                            if candidates else np.empty((0, 2)))
        if not len(self.portals):
            return []
        lengths = np.linalg.norm(self.portals - start, axis=1) + np.linalg.norm(self.portals - end, axis=1)
        # Only accept genuinely short alternatives here. More complex routes
        # still use the grid, whose safe connections are cached per geometry.
        limit = 1.5 * float(np.linalg.norm(end - start))
        order = np.flatnonzero(lengths <= limit)
        order = order[np.argsort(lengths[order], kind="stable")[:96]]
        for index in order:
            point = self.portals[index]
            if self._clear(start, point) and self._clear(point, end):
                return [point.copy(), end.copy()]
        return []

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
        if self._clear(start, end):
            return [end.copy()]
        if not self._inside(end):
            return []
        passage = self._short_passage(start, end)
        if passage:
            return passage
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
                if not self._connection_clear(cell, neighbor):
                    continue
                costs[neighbor], previous[neighbor] = proposed, cell
                heuristic = math.hypot((last[1] - nx) * self.pitch[0], (last[0] - ny) * self.pitch[1])
                heapq.heappush(frontier, (proposed + heuristic, proposed, neighbor))
        return []

    def _plan_arrival(self, start, end, radius):
        """Reach a safe point near a collectible, without entering its center.

        Fruit pickup is based on contact distance. Its observed center can be
        inside wall padding even when the fruit is reachable from open ground.
        Every approach keeps normal agent clearance; the final sightline to
        the fruit may enter padding but must never cross an observed wall.
        """
        if radius <= 0:
            return self._plan(start, end)
        angle = math.atan2(start[1] - end[1], start[0] - end[0])
        candidates = []
        for offset in (0., math.pi / 4, -math.pi / 4, math.pi / 2,
                       -math.pi / 2, 3 * math.pi / 4, -3 * math.pi / 4, math.pi):
            point = end + radius * np.array([math.cos(angle + offset), math.sin(angle + offset)])
            if self._inside(point) and self._clear(point, end, clearance=0.):
                candidates.append(point)
                if self._clear(start, point):
                    return [point]
        # Most pickups use the direct approach above. Wall detours use the
        # existing bounded planner and the first reachable nearby endpoint.
        for point in candidates:
            route = self._plan(start, point)
            if route:
                return route
        return []

    @staticmethod
    def _remaining(position, points):
        if not points:
            return math.inf
        path = np.vstack((position, points))
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def steer(self, agent_id, position, destination, now, waiting=False, arrival_radius=0.):
        position, destination = np.asarray(position, dtype=float), np.asarray(destination, dtype=float)
        arrival_radius = float(arrival_radius)
        if not math.isfinite(arrival_radius) or arrival_radius < 0:
            raise ValueError("arrival_radius must be finite and nonnegative")
        route = self.routes.get(agent_id)
        destination_changed = False
        if route is None or np.linalg.norm(destination - route.destination) > 5.:
            route = self.routes[agent_id] = Route(destination.copy(), arrival_radius=arrival_radius, progress_at=now)
        elif (not math.isclose(route.arrival_radius, arrival_radius)
              or (arrival_radius > 0 and np.linalg.norm(destination - route.destination) > 1e-8)):
            # Pose uncertainty changes the permitted pickup radius. Refine
            # the existing approach without treating radius noise as a new
            # task, or stationary agents could reset stall recovery forever.
            before = self._remaining(position, route.points)
            if route.points:
                offset = route.points[-1] - route.destination
                if np.linalg.norm(offset) < 1e-8:
                    offset = position - destination
                length = float(np.linalg.norm(offset))
                route.points[-1] = (destination + offset * arrival_radius / length
                                    if length > 1e-8 else destination.copy())
            route.destination = destination.copy()
            route.arrival_radius = arrival_radius
            after = self._remaining(position, route.points)
            if math.isfinite(before) and math.isfinite(after):
                # Only agent motion should count as progress. A larger pickup
                # circle shortens the path without moving the agent at all.
                route.best_remaining += after - before
            destination_changed = True
        if route.paused_at is not None:
            route.progress_at += max(0., now - route.paused_at)
            route.paused_at = None
        if waiting:
            route.progress_at = now
            route.status = "waiting"
            route.remaining = float(np.linalg.norm(destination - position))
            return NavigationResult(None, False, route.remaining, route.status)
        if (arrival_radius > 0 and np.linalg.norm(route.destination - position) <= arrival_radius + 1e-9
                and self._clear(position, route.destination, clearance=0.)):
            route.status, route.remaining, route.progress_at = "arrived", 0., now
            return NavigationResult(None, False, 0., route.status)
        if arrival_radius <= 0 and np.linalg.norm(route.destination - position) <= 4.:
            route.status, route.remaining, route.progress_at = "arrived", 0., now
            return NavigationResult(route.destination.copy(), False, 0., route.status)
        if route.status == "blocked":
            return NavigationResult(None, True, route.remaining, route.status)
        changed = route.revision != self.revision or destination_changed
        if changed and route.points:
            path = [position] + route.points
            if (not all(self._clear(a, b) for a, b in zip(path, path[1:]))
                    or (arrival_radius > 0 and not self._clear(route.points[-1], route.destination, clearance=0.))):
                route.points = []
        if not route.points:
            route.points = self._plan_arrival(position, route.destination, arrival_radius)
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
            route.points = self._plan_arrival(position, route.destination, arrival_radius)
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
