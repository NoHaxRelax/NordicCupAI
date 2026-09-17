"""Persistent scouting and local patrol duties in observation-derived frames."""

from dataclasses import dataclass, field
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.utils.controllers.policy_inputs import ExplorationHint, observed_edges
from src.utils.controllers.world_estimator import rotate, wrap


class RapidMappingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    only_until_anchored: bool = Field(default=False, strict=True)
    straight_before_anchor: bool = Field(default=True, strict=True)
    frontier_lookahead_cells: int = Field(default=3, ge=1, le=8, strict=True)
    frontier_target_distance: float = Field(default=240.0, gt=0)
    frontier_heading_weight: float = Field(default=0.75, ge=0, le=2)
    scan_enabled: bool = Field(default=True, strict=True)
    scan_amplitude: float = Field(default=math.pi / 3, ge=0, le=math.pi / 2)
    scan_period_seconds: float = Field(default=2.0, gt=0)
    sprint_enabled: bool = Field(default=False, strict=True)
    sprint_speed_fraction: float = Field(default=1.0, ge=0, le=1)
    sprint_min_energy: float = Field(default=160.0, ge=0)
    sprint_min_energy_fraction: float = Field(default=0.30, ge=0, le=1)
    sprint_minimum_reserve: float = Field(default=100.0, ge=0)
    sprint_burst_seconds: float = Field(default=0.4, gt=0)
    sprint_cooldown_seconds: float = Field(default=2.0, ge=0)
    sprint_stop_coverage: float = Field(default=0.80, ge=0, le=1)
    sprint_progress_timeout_seconds: float = Field(default=0.4, gt=0)
    sprint_min_progress: float = Field(default=1.0, ge=0)


class ExplorationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    trait_roles_enabled: bool = Field(default=False, strict=True)
    faster_mapping_enabled: bool = Field(default=False, strict=True)
    directed_foraging_enabled: bool = Field(default=False, strict=True)
    scout_forage_energy_threshold: float = Field(default=130.0, ge=0)
    scout_food_pickup_distance: float = Field(default=25.0, ge=0)
    boundary_seek_enabled: bool = Field(default=False, strict=True)
    boundary_min_length: float = Field(default=200.0, gt=0)
    # Must exceed the map format's boundary thickness (normally 30 units).
    boundary_corner_clearance: float = Field(default=40.0, gt=0)
    population_scout_fraction: float = Field(default=0.20, ge=0, le=1)
    balanced_resident_homes_enabled: bool = Field(default=False, strict=True)
    resident_home_patch_radius: float = Field(default=100.0, gt=0)
    resident_home_relocation_radius: float = Field(default=300.0, gt=0)
    resident_home_load_penalty: float = Field(default=130.0, ge=0)
    resident_home_min_energy: float = Field(default=150.0, ge=0)
    rapid_mapping: RapidMappingConfig = Field(default_factory=RapidMappingConfig)
    early_scout_fraction: float = Field(default=0.60, ge=0, le=1)
    sparse_coverage_threshold: float = Field(default=0.20, ge=0, le=1)
    dispersal_distance: float = Field(default=250.0, gt=0)
    frontier_replan_seconds: float = Field(default=2.0, gt=0)
    duty_interval_seconds: float = Field(default=30.0, gt=0)
    scouts_per_agents: int = Field(default=3, ge=2, strict=True)
    scout_energy_fraction: float = Field(default=0.25, ge=0, le=1)
    low_energy_fraction: float = Field(default=0.18, ge=0, le=1)
    resident_radius: float = Field(default=65.0, gt=0)
    resident_orbit_seconds: float = Field(default=35.0, gt=0)
    scout_speed_fraction: float = Field(default=0.85, ge=0, le=1)
    resident_speed_fraction: float = Field(default=0.60, ge=0, le=1)
    obstacle_margin: float = Field(default=4.0, ge=0)
    frontier_cell_size: float = Field(default=120.0, gt=0)
    max_frontier_cells: int = Field(default=4096, ge=4, strict=True)
    frontier_stale_seconds: float = Field(default=120.0, gt=0)
    target_tolerance: float = Field(default=25.0, gt=0)
    max_target_age_seconds: float = Field(default=60.0, gt=0)


@dataclass
class Duty:
    agent_id: int
    group_id: int
    frame_revision: int
    role: str
    home: np.ndarray
    heading: float
    assigned_at: float
    target: np.ndarray | None = None
    target_since: float = 0.0
    objective: str = "connect clusters"
    failed_targets: dict[tuple[float, float], float] = field(default_factory=dict)
    origin: np.ndarray | None = None
    dispersing: bool = False
    dispersed: bool = False
    food_home_set: bool = False
    next_target_at: float = 0.0
    target_plan_time: float = -math.inf
    last_position: np.ndarray | None = None
    last_position_time: float = -math.inf
    progress_at: float = -math.inf
    sprint_until: float = -math.inf
    sprint_ready_at: float = -math.inf
    sprint_requested: bool = False
    boundary_goal: np.ndarray | None = None


@dataclass
class BoundaryTask:
    agent_id: int
    frame_revision: int
    target: np.ndarray


@dataclass
class FrontierPlan:
    frame_revision: int
    world_size: tuple | None
    updated_at: float
    cell_size: float
    targets: np.ndarray
    cells: np.ndarray
    coverage: float | None
    edges: np.ndarray
    edge_min: np.ndarray
    edge_max: np.ndarray
    rapid_mode: bool = False


def _point_segment_distance(point, start, end):
    # These are individual 2D points, not arrays of points. Scalar arithmetic
    # avoids allocating several NumPy arrays for every candidate and edge.
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    dx, dy = float(end[0]) - sx, float(end[1]) - sy
    length_squared = dx * dx + dy * dy
    fraction = (0.0 if length_squared == 0 else
                max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_squared)))
    return math.hypot(px - (sx + fraction * dx), py - (sy + fraction * dy))


def _path_clear(direction, distance, edges, margin):
    start = (0.0, 0.0)
    end = (distance * math.cos(direction), distance * math.sin(direction))
    for first, second in edges:
        initial = _point_segment_distance(start, first, second)
        final = _point_segment_distance(end, first, second)
        minimum = min(initial, final,
                      _point_segment_distance(first, start, end),
                      _point_segment_distance(second, start, end))
        # The endpoint distances alone miss a crossing between both interiors.
        wall_x, wall_y = second[0] - first[0], second[1] - first[1]
        cross = end[0] * wall_y - end[1] * wall_x
        if abs(cross) > 1e-9:
            travel = (first[0] * wall_y - first[1] * wall_x) / cross
            along = (first[0] * end[1] - first[1] * end[0]) / cross
            if 0 < travel <= 1 and 0 <= along <= 1:
                return False
        # When already beside a wall, allow motion parallel to or away from it.
        if minimum < min(initial, margin) - 1e-6 or (initial < margin and final < initial - 1e-6):
            return False
    return True


def avoid_edges(direction, distance, edges, margin, agent_id):
    """Choose a nearby clear heading, or stop if enclosed; deterministic ties."""
    if not edges or distance <= 0:
        return wrap(direction), distance
    lookahead = max(distance + margin, 20.0)
    side = 1 if agent_id % 2 else -1
    offsets = [0.0]
    for step in range(1, 7):
        offsets.extend((side * step * math.pi / 6, -side * step * math.pi / 6))
    for offset in offsets:
        candidate = wrap(direction + offset)
        if _path_clear(candidate, lookahead, edges, margin):
            return candidate, distance
    # A narrow opening can be safe for one short step even if lookahead fails.
    for offset in offsets:
        candidate = wrap(direction + offset)
        if _path_clear(candidate, distance, edges, margin):
            return candidate, distance
    return wrap(direction + side * math.pi / 2), 0.0


class ExplorationCoordinator:
    def __init__(self, config, max_position_uncertainty):
        self.config = config
        self.max_position_uncertainty = max_position_uncertainty
        self.reset()

    def reset(self):
        self.duties: dict[int, Duty] = {}
        self.group_duties = {}
        self.frontiers: dict[int, FrontierPlan] = {}
        self.boundary_tasks: dict[int, BoundaryTask] = {}

    def rapid_active(self, group):
        rapid = self.config.rapid_mapping
        return rapid.enabled and not (rapid.only_until_anchored and group.anchored)

    def _resident_home(self, group, pose, state, population_phase):
        positions = np.asarray([tree.position for tree in group.trees])
        distances = np.linalg.norm(positions - pose.position, axis=1)
        nearest = int(np.argmin(distances))
        if (not self.config.balanced_resident_homes_enabled or not population_phase or not group.anchored
                or state["energy"] < max(self.config.resident_home_min_energy,
                                         state["max_energy"] * self.config.scout_energy_fraction)):
            return positions[nearest].copy()
        homes = np.asarray([duty.home for duty in self.duties.values()
                            if duty.group_id == group.group_id and duty.frame_revision == group.frame_revision
                            and duty.role == "resident" and duty.food_home_set]).reshape((-1, 2))
        # Adjacent trees share local fruit supplies: count nearby resident homes,
        # rather than treating each tree as an independent empty territory.
        loads = np.sum(np.linalg.norm(positions[:, None, :] - homes, axis=2)
                       <= self.config.resident_home_patch_radius, axis=1)
        scores = distances + self.config.resident_home_load_penalty * loads
        candidates = np.flatnonzero(distances <= self.config.resident_home_relocation_radius)
        for index in sorted(candidates, key=lambda index: (scores[index], distances[index], int(index))):
            if index == nearest:
                return positions[index].copy()
            # Relocation only follows a direct route through the observed map.
            vector = positions[index] - pose.position
            edges = [(edge.start - pose.position, edge.end - pose.position) for edge in group.edges]
            if _path_clear(math.atan2(vector[1], vector[0]), float(distances[index]),
                           edges, self.config.obstacle_margin):
                return positions[index].copy()
        return positions[nearest].copy()

    def _assign(self, group, poses, states, now, trait_ratings=None, population_phase=False):
        trait_ratings = trait_ratings or {}
        members = tuple(p.agent_id for p in poses)
        frame = group.frame_revision
        for pose in poses:
            duty = self.duties.get(pose.agent_id)
            if duty is None or duty.group_id != group.group_id or duty.frame_revision != frame:
                # Homes and targets belong to one coordinate frame. A transform
                # invalidates them; never reuse an old local vector as absolute.
                previous_role = duty.role if duty is not None and self.config.faster_mapping_enabled else "resident"
                dispersed = duty.dispersed if duty is not None else False
                dispersing = duty.dispersing if duty is not None else False
                self.duties[pose.agent_id] = Duty(
                    pose.agent_id, group.group_id, frame, previous_role, pose.position.copy(),
                    pose.heading, now,
                )
                self.duties[pose.agent_id].dispersed = dispersed
                self.duties[pose.agent_id].dispersing = dispersing
            origin = getattr(pose, "origin", None)
            duty = self.duties[pose.agent_id]
            if origin is not None:
                duty.origin = origin.copy()
            elif duty.origin is None:
                duty.origin = pose.position.copy()
        eligible = [p for p in poses if states[p.agent_id]["energy"] >=
                    states[p.agent_id]["max_energy"] * self.config.scout_energy_fraction]
        count = min(len(eligible), max(1, math.ceil(len(poses) / self.config.scouts_per_agents)))
        if self.config.faster_mapping_enabled and len(poses) > 1 and not population_phase:
            frontier = self.frontiers.get(group.group_id)
            coverage = None if frontier is None else frontier.coverage
            if not group.anchored or coverage is None or coverage < self.config.sparse_coverage_threshold:
                count = min(len(eligible), len(poses) - 1,
                            max(count, math.ceil(len(poses) * self.config.early_scout_fraction)))
        if self.config.trait_roles_enabled and len(poses) > 1 and not population_phase:
            low_ranked = sum(bool(getattr(trait_ratings.get(p.agent_id), "low_rank", False)
                                  or self.duties[p.agent_id].dispersing) for p in eligible)
            count = min(len(poses) - 1, max(count, low_ranked))
        if population_phase:
            # Keep breeders near their food once a shared frame exists. A
            # healthy nonelite always takes a scouting slot before an elite;
            # fewer scouts are preferable to filling the quota with breeders.
            alternatives = [p for p in eligible if not getattr(trait_ratings.get(p.agent_id), "elite", False)]
            if alternatives:
                eligible = alternatives
            count = min(len(eligible), math.ceil(len(poses) * self.config.population_scout_fraction))
        elif len(poses) == 1:
            eligible, count = poses, 1
        previous = self.group_duties.get(group.group_id)
        scouts = {p.agent_id for p in poses if self.duties[p.agent_id].role == "scout"}
        eligible_ids = {p.agent_id for p in eligible}
        if (previous is None or previous[0] != (members, frame)
                or now >= previous[1] or len(scouts) != count or not scouts <= eligible_ids):
            stable_ids = set()
            if self.config.faster_mapping_enabled and previous is not None and now < previous[1]:
                # A newborn joining the map does not cancel a healthy scout's
                # route. Fill new vacancies without reranking existing duties.
                stable_ids = scouts & eligible_ids
            eligible.sort(key=lambda p: (
                0 if p.agent_id in stable_ids else 1,
                (0 if (getattr(trait_ratings.get(p.agent_id), "low_rank", False)
                       or self.duties[p.agent_id].dispersing) else
                 2 if getattr(trait_ratings.get(p.agent_id), "elite", False) else 1)
                if self.config.trait_roles_enabled else 0,
                -states[p.agent_id]["energy"] / max(1.0, states[p.agent_id]["max_energy"]),
                p.agent_id,
            ))
            chosen = {p.agent_id for p in eligible[:count]}
            for pose in poses:
                duty = self.duties[pose.agent_id]
                role = "scout" if pose.agent_id in chosen else "resident"
                if duty.role != role:
                    duty.role, duty.assigned_at = role, now
                    duty.home, duty.target = pose.position.copy(), None
                    duty.food_home_set = False
                    # Existing scouts retain their direction across rebalances.
                    offset = (pose.agent_id * 2.399963229728653 if self.config.faster_mapping_enabled
                              else len(chosen - {pose.agent_id}) * (pose.agent_id % 3) * math.tau / 3)
                    if self.rapid_active(group) and len(poses) == 1:
                        offset = 0.0
                    duty.heading = wrap(pose.heading + offset)
            deadline = (previous[1] if previous is not None and now < previous[1]
                        else now + self.config.duty_interval_seconds)
            self.group_duties[group.group_id] = ((members, frame), deadline)

    def _boundary_task(self, group, poses, states):
        """Direct one existing scout toward the nearer endpoint of a long wall.

        All geometry remains in the cluster's observed coordinate frame. The
        side containing the scout determines the inward normal; the endpoint
        and directed tangent give a corner approach without world dimensions.
        """
        for pose in poses:
            self.duties[pose.agent_id].boundary_goal = None
        if not self.config.boundary_seek_enabled or group.anchored:
            self.boundary_tasks.pop(group.group_id, None)
            return None
        eligible = [pose for pose in poses if self.duties[pose.agent_id].role == "scout"
                    and pose.uncertainty <= self.max_position_uncertainty
                    and states[pose.agent_id]["energy"] >= states[pose.agent_id]["max_energy"] * max(
                        self.config.scout_energy_fraction, self.config.low_energy_fraction)]
        previous = self.boundary_tasks.get(group.group_id)
        if (previous is not None and previous.frame_revision == group.frame_revision
                and any(pose.agent_id == previous.agent_id for pose in eligible)):
            self.duties[previous.agent_id].boundary_goal = previous.target.copy()
            return previous
        candidates = []
        clearance = self.config.boundary_corner_clearance
        for edge in group.edges:
            delta = edge.end - edge.start
            length = float(np.linalg.norm(delta))
            if not math.isfinite(length) or length < max(self.config.boundary_min_length, 2 * clearance):
                continue
            tangent = delta / length
            normal = np.array((-tangent[1], tangent[0]))
            for pose in eligible:
                side = float((pose.position - edge.start) @ normal)
                if abs(side) < 1e-6:
                    continue
                inward = normal if side > 0 else -normal
                targets = (edge.start + tangent * clearance + inward * clearance,
                           edge.end - tangent * clearance + inward * clearance)
                for target in targets:
                    distance = float(np.linalg.norm(target - pose.position))
                    candidates.append((distance, pose.agent_id, float(target[0]), float(target[1])))
        if not candidates:
            self.boundary_tasks.pop(group.group_id, None)
            return None
        _, agent_id, x, y = min(candidates)
        task = BoundaryTask(agent_id, group.frame_revision, np.array((x, y)))
        self.boundary_tasks[group.group_id] = task
        duty = self.duties[agent_id]
        duty.boundary_goal = task.target.copy()
        # Release an ordinary frontier reservation while this agent anchors
        # the map for the rest of its cluster.
        duty.target = None
        return task

    def _frontier_plan(self, group, poses, now):
        """Cache one bounded frontier grid per map, not per agent or tick."""
        plan = self.frontiers.get(group.group_id)
        world_size = tuple(group.world_size) if group.anchored and group.world_size is not None else None
        if (plan is not None and plan.frame_revision == group.frame_revision and plan.world_size == world_size
                and plan.rapid_mode == self.rapid_active(group)
                and now - plan.updated_at < self.config.frontier_replan_seconds):
            return plan
        cell = self.config.frontier_cell_size
        if world_size is not None:
            width, height = world_size
            while math.ceil(width / cell) * math.ceil(height / cell) > self.config.max_frontier_cells:
                cell *= 2
            nx, ny = max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))
        else:
            nx = ny = None
        # Keep explored cells explored even when their food observations age.
        # Pose cells keep the frontier useful when memory is still empty.
        positions = [position for position, _, _ in group.visited.values()]
        positions.extend(p.position for p in poses if p.uncertainty <= self.max_position_uncertainty)
        reached = {tuple(np.floor(position / cell).astype(int)) for position in positions}
        if world_size is not None:
            reached = {(x, y) for x, y in reached if 0 <= x < nx and 0 <= y < ny}
        lookahead = self.config.rapid_mapping.frontier_lookahead_cells if self.rapid_active(group) else 1
        unknown = {(x + dx * step, y + dy * step) for x, y in reached
                   for dx, dy in ((-1, 0), (0, -1), (0, 1), (1, 0))
                   for step in range(1, lookahead + 1)}
        unknown -= reached
        if world_size is not None:
            unknown = {(x, y) for x, y in unknown if 0 <= x < nx and 0 <= y < ny}
        cells = np.asarray(sorted(unknown)[:self.config.max_frontier_cells], dtype=int).reshape((-1, 2))
        targets = (cells.astype(float) + 0.5) * cell
        if world_size is not None and len(targets):
            # Final partial cells use their actual center, inside world bounds.
            targets = (cells * cell + np.minimum((cells + 1) * cell, world_size)) / 2
        edges = np.asarray([(edge.start, edge.end) for edge in group.edges], dtype=float).reshape((-1, 2, 2))
        edge_min = edges.min(axis=1) if len(edges) else np.empty((0, 2))
        edge_max = edges.max(axis=1) if len(edges) else np.empty((0, 2))
        coverage = None if world_size is None else len(reached) / (nx * ny)
        plan = FrontierPlan(group.frame_revision, world_size, now, cell, targets, cells,
                            coverage, edges, edge_min, edge_max, rapid_mode=self.rapid_active(group))
        self.frontiers[group.group_id] = plan
        return plan

    def _clear_route(self, plan, position, target):
        if not len(plan.edges):
            return True
        margin = self.config.obstacle_margin
        lower, upper = np.minimum(position, target) - margin, np.maximum(position, target) + margin
        relevant = np.all(plan.edge_min <= upper, axis=1) & np.all(plan.edge_max >= lower, axis=1)
        edges = (plan.edges[relevant] - position).tolist()
        vector = target - position
        return _path_clear(math.atan2(vector[1], vector[0]), float(np.linalg.norm(vector)), edges, margin)

    def _nearby_frontier(self, group, pose, reserved, now, outward=False):
        plan = self.frontiers[group.group_id]
        targets = plan.targets
        if not len(targets):
            return None
        duty = self.duties[pose.agent_id]
        vectors = targets - pose.position
        distances = np.linalg.norm(vectors, axis=1)
        score = distances - 0.35 * (vectors @ rotate((1, 0), duty.heading))
        rapid = self.config.rapid_mapping
        if self.rapid_active(group):
            projection = vectors @ rotate((1, 0), duty.heading)
            alignment = projection / np.maximum(distances, 1e-9)
            score = (np.abs(distances - rapid.frontier_target_distance)
                     + rapid.frontier_heading_weight * rapid.frontier_target_distance * (1 - alignment))
        valid = distances > self.config.target_tolerance
        # Cached candidates that were reached since the last plan are no
        # longer frontiers for this agent.
        valid &= np.any(plan.cells != np.floor(pose.position / plan.cell_size).astype(int), axis=1)
        for target in reserved:
            valid &= np.linalg.norm(targets - target, axis=1) >= plan.cell_size * 0.65
        for target in duty.failed_targets:
            valid &= np.linalg.norm(targets - target, axis=1) > 1e-6
        if outward and duty.origin is not None:
            origin_distances = np.linalg.norm(targets - duty.origin, axis=1)
            current_distance = float(np.linalg.norm(pose.position - duty.origin))
            # Prefer progress away from the birth location. Some sideways
            # routes remain possible when an observed stone blocks the direct way.
            progress = np.minimum(np.maximum(0.0, origin_distances - current_distance),
                                  max(0.0, self.config.dispersal_distance - current_distance))
            score -= 0.45 * progress
            valid &= origin_distances >= current_distance - plan.cell_size * 0.25
        indices = np.flatnonzero(valid)
        for index in indices[np.argsort(score[indices], kind="stable")][:16]:
            if self._clear_route(plan, pose.position, targets[index]):
                if self.rapid_active(group):
                    duty.heading = math.atan2(vectors[index, 1], vectors[index, 0])
                return targets[index].copy()
        return None

    def _frontier(self, group, pose, reserved, now):
        if self.config.faster_mapping_enabled:
            return self._nearby_frontier(group, pose, reserved, now)
        width, height = group.world_size
        cell = self.config.frontier_cell_size
        while math.ceil(width / cell) * math.ceil(height / cell) > self.config.max_frontier_cells:
            cell *= 2
        nx, ny = max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))
        visited = {(int(position[0] // cell), int(position[1] // cell))
                   for position, _, last_seen in group.visited.values()
                   if now - last_seen <= self.config.frontier_stale_seconds}
        candidates = []
        failed_targets = self.duties[pose.agent_id].failed_targets
        for x in range(nx):
            for y in range(ny):
                if (x, y) in visited:
                    continue
                target = np.array(((x * cell + min((x + 1) * cell, width)) / 2,
                                   (y * cell + min((y + 1) * cell, height)) / 2))
                if tuple(target) in failed_targets:
                    continue
                if any(np.linalg.norm(target - other) < cell * 0.5 for other in reserved):
                    continue
                vector = target - pose.position
                # Prefer nearby unexplored cells with a small directional bias.
                heading = rotate((1, 0), self.duties[pose.agent_id].heading)
                score = float(np.linalg.norm(vector) - 0.15 * (vector @ heading))
                candidates.append((score, x, y, target))
        return min(candidates, key=lambda item: item[:3])[3] if candidates else None

    def _scout_target(self, group, pose, reserved, now, outward=False):
        duty = self.duties[pose.agent_id]
        duty.failed_targets = {point: time for point, time in duty.failed_targets.items()
                               if now - time < self.config.frontier_stale_seconds}
        reached = duty.target is not None and np.linalg.norm(duty.target - pose.position) <= self.config.target_tolerance
        expired = duty.target is not None and now - duty.target_since >= self.config.max_target_age_seconds
        occupied = duty.target is not None and any(
            np.linalg.norm(duty.target - other) < self.config.target_tolerance for other in reserved)
        already_explored = False
        newly_blocked = False
        if self.config.faster_mapping_enabled:
            plan = self.frontiers[group.group_id]
            if duty.target is not None and plan.updated_at > duty.target_plan_time:
                if self.rapid_active(group):
                    # Keep straight goals through cell-entry, but react when
                    # newly observed stone geometry invalidates the route.
                    newly_blocked = not self._clear_route(plan, pose.position, duty.target)
                else:
                    already_explored = not np.any(np.linalg.norm(plan.targets - duty.target, axis=1) < 1e-6)
                duty.target_plan_time = plan.updated_at
        if (expired or newly_blocked) and not reached:
            duty.failed_targets[tuple(duty.target)] = now
            duty.failed_targets = dict(list(duty.failed_targets.items())[-64:])
        if reached or expired or occupied or already_explored or newly_blocked:
            duty.target = None
            duty.next_target_at = now
        if duty.target is None and now >= duty.next_target_at:
            duty.target = (self._nearby_frontier(group, pose, reserved, now, outward)
                           if self.config.faster_mapping_enabled else self._frontier(group, pose, reserved, now))
            duty.target_since = now
            duty.next_target_at = now + self.config.frontier_replan_seconds
            if self.config.faster_mapping_enabled:
                duty.target_plan_time = self.frontiers[group.group_id].updated_at
        if duty.target is not None:
            reserved.append(duty.target)
        return duty.target

    def _sprint_speed(self, state, pose, duty, group, now, direction, goal, edges):
        """Request short scouting bursts only on a visible, progressing route."""
        rapid = self.config.rapid_mapping
        duty.sprint_requested = False
        if not self.rapid_active(group) or not rapid.sprint_enabled:
            return None
        movement = rotate((1.0, 0.0), direction)
        if duty.last_position is not None and now > duty.last_position_time:
            progress = float((pose.position - duty.last_position) @ movement)
            if progress >= rapid.sprint_min_progress:
                duty.progress_at = now
        if now > duty.last_position_time:
            duty.last_position = pose.position.copy()
            duty.last_position_time = now
        speed = state["sprint_speed"] * rapid.sprint_speed_fraction
        walk = min(state["speed"], state["sprint_speed"])
        frontier = self.frontiers.get(group.group_id)
        coverage = None if frontier is None else frontier.coverage
        if (duty.role != "scout" or speed <= walk
                or pose.uncertainty > self.max_position_uncertainty
                or state["energy"] < max(rapid.sprint_min_energy,
                                          state["max_energy"] * rapid.sprint_min_energy_fraction,
                                          rapid.sprint_minimum_reserve)
                or now - duty.progress_at > rapid.sprint_progress_timeout_seconds
                or (coverage is not None and coverage >= rapid.sprint_stop_coverage)
                or (goal is not None and np.linalg.norm(goal - pose.position) < speed)):
            return None
        local_direction = wrap(direction - pose.heading)
        # Speeding sideways while the camera is looking away spends energy
        # against obstacles that have not yet been surveyed. Wait for the
        # scanning cone to include the travel direction again.
        if (abs(local_direction) > max(0.0, state["vision_angle"] / 2)
                or state["vision_range"] < speed + self.config.obstacle_margin
                or not _path_clear(local_direction, speed + self.config.obstacle_margin,
                                   edges, self.config.obstacle_margin)):
            return None
        if now >= duty.sprint_until:
            if now < duty.sprint_ready_at:
                return None
            duty.sprint_until = now + rapid.sprint_burst_seconds
            duty.sprint_ready_at = duty.sprint_until + rapid.sprint_cooldown_seconds
        duty.sprint_requested = True
        return speed

    def instructions(self, states, estimator, now, trait_ratings=None, population_phase=False, managed_groups=()):
        if not self.config.enabled:
            return {}
        trait_ratings = trait_ratings or {}
        states = {state["agent_id"]: state for state in states}
        self.duties = {key: duty for key, duty in self.duties.items() if key in states
                       and estimator.poses[key].group_id not in managed_groups}
        self.group_duties = {key: value for key, value in self.group_duties.items() if key in estimator.groups}
        self.frontiers = {key: value for key, value in self.frontiers.items() if key in estimator.groups}
        self.boundary_tasks = {key: value for key, value in self.boundary_tasks.items() if key in estimator.groups}
        hints = {}
        for group_id, group in sorted(estimator.groups.items()):
            if group_id in managed_groups:
                continue
            poses = sorted((p for p in estimator.poses.values() if p.group_id == group_id),
                           key=lambda p: p.agent_id)
            if self.config.faster_mapping_enabled or self.config.directed_foraging_enabled:
                self._frontier_plan(group, poses, now)
            self._assign(group, poses, states, now, trait_ratings, population_phase)
            boundary = self._boundary_task(group, poses, states)
            claimed = {pose.agent_id: self.duties[pose.agent_id].target for pose in poses
                       if self.duties[pose.agent_id].target is not None}
            for pose in poses:
                duty, state = self.duties[pose.agent_id], states[pose.agent_id]
                # Existing scouts own their routes before newborns request a
                # destination, independent of the order of agent IDs.
                reserved = [target for owner, target in claimed.items() if owner != pose.agent_id]
                reliable = pose.uncertainty <= self.max_position_uncertainty
                hungry = state["energy"] < state["max_energy"] * self.config.low_energy_fraction
                healthy = state["energy"] >= state["max_energy"] * self.config.scout_energy_fraction
                rating = trait_ratings.get(pose.agent_id)
                low_rank = self.config.trait_roles_enabled and bool(getattr(rating, "low_rank", False))
                elite = self.config.trait_roles_enabled and bool(getattr(rating, "elite", False))
                if population_phase and duty.role != "scout":
                    duty.dispersing = False
                if duty.origin is not None and reliable:
                    if np.linalg.norm(pose.position - duty.origin) >= self.config.dispersal_distance:
                        duty.dispersed = True
                        duty.dispersing = False
                disperse = (self.config.trait_roles_enabled and healthy and not duty.dispersed
                            and (low_rank or duty.dispersing) and (not population_phase or duty.role == "scout"))
                if disperse and not duty.dispersing:
                    duty.dispersing = True
                    duty.target = None
                    duty.next_target_at = now
                speed = (self.config.scout_speed_fraction if duty.role == "scout"
                         else self.config.resident_speed_fraction)
                direction = duty.heading
                goal = None
                if hungry and group.trees and reliable:
                    goal = min(group.trees, key=lambda tree: np.linalg.norm(tree.position - pose.position)).position
                    if np.linalg.norm(goal - pose.position) <= self.config.target_tolerance:
                        # Fruit can be outside the current vision cone. Keep
                        # surveying the tree instead of parking at its center.
                        phase = math.tau * now / self.config.resident_orbit_seconds + pose.agent_id * 2.4
                        goal = goal + rotate((self.config.target_tolerance, 0), phase)
                    duty.objective = "find food near remembered tree"
                elif boundary is not None and boundary.agent_id == pose.agent_id:
                    goal = boundary.target
                    duty.objective = "find perpendicular boundary at nearer corner"
                elif disperse and reliable:
                    straight = (self.rapid_active(group) and self.config.rapid_mapping.straight_before_anchor
                                and not group.anchored)
                    if self.config.faster_mapping_enabled and not straight:
                        goal = self._scout_target(group, pose, reserved, now, outward=True)
                    if goal is None and not straight:
                        outward = pose.position - duty.origin
                        heading = math.atan2(outward[1], outward[0]) if np.linalg.norm(outward) > 20 else duty.heading
                        goal = duty.origin + rotate((self.config.dispersal_distance, 0), heading)
                    duty.objective = "disperse from birth area"
                elif (duty.role == "scout" and self.rapid_active(group)
                      and self.config.rapid_mapping.straight_before_anchor and not group.anchored):
                    duty.target = None
                    duty.objective = "connect clusters and find boundaries"
                elif duty.role == "resident" and reliable:
                    # Patrol a bounded home area; return directly after a food or
                    # predator detour carries the resident beyond its radius.
                    if (elite or population_phase) and not duty.food_home_set and group.trees:
                        duty.home = self._resident_home(group, pose, state, population_phase)
                        # Young residents recover at the nearest tree first;
                        # assign their lasting patch once they can travel safely.
                        duty.food_home_set = (
                            not (self.config.balanced_resident_homes_enabled and population_phase and group.anchored)
                            or state["energy"] >= max(self.config.resident_home_min_energy,
                                                      state["max_energy"] * self.config.scout_energy_fraction)
                        )
                    offset = pose.position - duty.home
                    phase = math.tau * now / self.config.resident_orbit_seconds + pose.agent_id * 2.4
                    goal = (duty.home if np.linalg.norm(offset) > self.config.resident_radius
                            else duty.home + rotate((self.config.resident_radius * 0.65, 0), phase))
                    duty.objective = "forage near breeding area" if elite or population_phase else "patrol home"
                elif reliable and (self.config.faster_mapping_enabled or (group.anchored and group.world_size is not None)):
                    goal = self._scout_target(group, pose, reserved, now)
                    duty.objective = ("explore unknown cells" if group.anchored else "expand cluster map") if goal is not None else "seek unexplored routes"
                else:
                    duty.objective = "connect clusters and find boundaries" if reliable else "seek landmarks"
                if goal is not None:
                    vector = goal - pose.position
                    if np.linalg.norm(vector) > 1e-6:
                        direction = math.atan2(vector[1], vector[0])
                local_direction = wrap(direction - pose.heading)
                edges = observed_edges(state["observations"])
                requested_speed = self._sprint_speed(state, pose, duty, group, now, direction, goal, edges)
                distance = (requested_speed if requested_speed is not None else
                            min(state["speed"], state["sprint_speed"]) * speed)
                if goal is not None:
                    distance = min(distance, float(np.linalg.norm(goal - pose.position)))
                safe_direction, distance = avoid_edges(
                    local_direction, distance, edges,
                    self.config.obstacle_margin, pose.agent_id,
                )
                if duty.role == "scout" and goal is None:
                    duty.heading = wrap(pose.heading + safe_direction)
                rapid = self.config.rapid_mapping
                look_direction = None
                if self.rapid_active(group) and rapid.scan_enabled and duty.role == "scout":
                    phase = math.tau * now / rapid.scan_period_seconds + pose.agent_id * 2.399963229728653
                    offset = rapid.scan_amplitude * math.sin(phase) if requested_speed is None else 0.0
                    look_direction = wrap(safe_direction + offset)
                food_distance_limit = None
                frontier = self.frontiers.get(group.group_id)
                coverage = None if frontier is None else frontier.coverage
                if (self.config.directed_foraging_enabled and duty.role == "scout" and reliable and not hungry
                        and state["energy"] >= max(self.config.scout_forage_energy_threshold,
                                                   state["max_energy"] * self.config.scout_energy_fraction)
                        and duty.objective != "find food near remembered tree"
                        and (not group.anchored or (coverage is not None and coverage < 0.9))):
                    food_distance_limit = self.config.scout_food_pickup_distance
                hints[pose.agent_id] = ExplorationHint(
                    tuple(rotate((distance, 0), safe_direction)), duty.role, duty.objective,
                    self.config.obstacle_margin, requested_speed,
                    rapid.sprint_minimum_reserve if requested_speed is not None else 75.0,
                    look_direction, food_distance_limit=food_distance_limit,
                )
                if duty.target is None:
                    claimed.pop(pose.agent_id, None)
                else:
                    claimed[pose.agent_id] = duty.target
        return hints

    def snapshot(self):
        return {agent_id: {
            "role": duty.role, "home": duty.home.tolist(), "scout_heading": duty.heading,
            "target": (duty.boundary_goal.tolist() if duty.boundary_goal is not None
                       else None if duty.target is None else duty.target.tolist()),
            "boundary_target": None if duty.boundary_goal is None else duty.boundary_goal.tolist(),
            "objective": duty.objective, "role_since": duty.assigned_at,
            "origin": None if duty.origin is None else duty.origin.tolist(),
            "dispersing": duty.dispersing, "dispersed": duty.dispersed,
            "sprint_requested": duty.sprint_requested,
        } for agent_id, duty in sorted(self.duties.items())}
