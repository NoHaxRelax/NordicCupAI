"""Estimate shared local coordinate frames from API observations and actions.

No simulator objects, world coordinates, map dimensions, or hidden tree data are
accepted. Disconnected groups deliberately keep separate coordinate frames.
"""

from dataclasses import dataclass, field
import math
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.utils.controllers.boundary_anchor import (
    infer_boundary_frame, infer_directed_boundary_axes, infer_single_boundary_frame,
)


Nonnegative = Annotated[float, Field(ge=0)]


class EstimatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    odometry_error_per_unit: float = Field(ge=0)
    max_position_uncertainty: float = Field(gt=0)
    landmark_match_radius: float = Field(gt=0)
    landmark_consensus_tolerance: float = Field(gt=0)
    minimum_landmark_matches: int = Field(ge=2)
    sighting_uncertainty: float = Field(gt=0)
    tree_memory_seconds: float = Field(gt=0)
    visited_cell_size: float = Field(gt=0)
    max_trees_per_group: int = Field(ge=2)
    max_visited_cells_per_group: int = Field(ge=1)
    biome_movement_factors: dict[str, Nonnegative]
    biome_food_potential: dict[str, Nonnegative]
    unknown_biome_movement_factor: float = Field(ge=0)
    unknown_biome_food_potential: float = Field(ge=0)
    max_edges_per_group: int = Field(default=1024, ge=2)
    max_sighting_links: int = Field(default=4096, ge=1)
    boundary_minimum_length: float = Field(default=200.0, gt=0)
    boundary_wall_thickness: float = Field(default=30.0, ge=0)
    anchor_boundaries: bool = True
    anchor_single_boundary: bool = False
    detect_stale_observations: bool = False
    shared_landmark_merges: bool = False
    shared_landmark_interval_seconds: float = Field(default=1.0, gt=0)
    shared_landmark_max_candidates: int = Field(default=128, ge=2)
    shared_landmark_max_pairs: int = Field(default=32, ge=1)
    # Experimental, conservative recovery from exact stone shapes. A single
    # segment cannot identify a collision perpendicular to its direction.
    relocalize_stones: bool = False


def wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


def rotate(point, angle):
    x, y = point
    return np.array((x * math.cos(angle) - y * math.sin(angle),
                     x * math.sin(angle) + y * math.cos(angle)), dtype=float)


def relative_vector(observation):
    return rotate((float(observation["distance"]), 0), float(observation["angle"]))


def tree_sightings(observations):
    return sorted((obj for obj in observations if obj["type"] == "Tree"),
                  key=lambda obj: (obj["distance"], obj["angle"]))


def edge_sightings(observations):
    """The ray caster can return the same whole segment more than once."""
    edges, exact = {}, set()
    for obj in observations:
        if obj.get("type") != "Edge":
            continue
        raw = obj.get("coords", ())
        try:
            if len(raw) == 2 and len(raw[0]) == 2 and len(raw[1]) == 2:
                # Rays commonly repeat precisely the same segment dozens of times.
                # Avoid repeated array allocation, validation and rounding for them.
                raw_key = (raw[0][0], raw[0][1], raw[1][0], raw[1][1])
                if raw_key in exact:
                    continue
                exact.add(raw_key)
        except (TypeError, IndexError):
            pass
        coords = np.asarray(raw, dtype=float)
        if coords.shape != (2, 2) or not np.isfinite(coords).all():
            continue
        if np.linalg.norm(coords[1] - coords[0]) < 1e-6:
            continue
        key = tuple(np.round(coords.ravel(), 6))
        edges[key] = coords
    return [edges[key] for key in sorted(edges)]


class ObservationGeometry:
    """Parsed once per observer, with offsets reused until its heading changes."""

    def __init__(self, observations):
        self.edges = edge_sightings(observations)
        self.trees = tree_sightings(observations)
        self.agents = sorted((obj for obj in observations
                              if obj["type"] == "Agent" and "id" in obj and "rel_dir" in obj),
                             key=lambda obj: obj["id"])
        self._heading = None
        self._edges = self._trees = None
        self._stone_key = None
        self._stone_candidates = None

    def _set_heading(self, heading):
        if self._heading != heading:
            self._heading = heading
            self._edges = self._trees = None

    def edge_offsets(self, heading):
        self._set_heading(heading)
        if self._edges is None:
            self._edges = np.array([[rotate(point, heading) for point in coords]
                                   for coords in self.edges]).reshape((-1, 2, 2))
        return self._edges

    def tree_offsets(self, heading):
        self._set_heading(heading)
        if self._trees is None:
            self._trees = [rotate(relative_vector(observation), heading) for observation in self.trees]
        return self._trees


def observation_geometry(observations):
    return observations if isinstance(observations, ObservationGeometry) else ObservationGeometry(observations)


@dataclass
class EstimatedPose:
    agent_id: int
    group_id: int
    position: np.ndarray
    heading: float = 0.0
    uncertainty: float = 0.0
    biome: str = ""
    age: float = 0.0
    origin: np.ndarray | None = None

    def __post_init__(self):
        self.origin = np.asarray(self.position if self.origin is None else self.origin, dtype=float).copy()


@dataclass
class TreeLandmark:
    position: np.ndarray
    last_seen: float


@dataclass
class EdgeLandmark:
    start: np.ndarray
    end: np.ndarray
    last_seen: float
    sightings: int = 1


@dataclass
class BiomeSample:
    position: np.ndarray
    biome: str
    last_seen: float
    uncertainty: float


@dataclass
class SightingLink:
    observer_id: int
    target_id: int
    first_seen: float
    last_seen: float
    count: int
    distance: float
    angle: float
    relative_heading: float | None


@dataclass
class MapGroup:
    group_id: int
    trees: list[TreeLandmark] = field(default_factory=list)
    # Keyed by local grid cell; values contain position, observed biome potential, time.
    visited: dict = field(default_factory=dict)
    revision: int = 0
    edges: list[EdgeLandmark] = field(default_factory=list)
    biomes: dict[tuple[int, int], BiomeSample] = field(default_factory=dict)
    anchored: bool = False
    world_size: tuple[float, float] | None = None
    known_width: float | None = None
    known_height: float | None = None
    frame_revision: int = 0
    _edge_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _tree_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _boundary_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _boundary_key: tuple | None = field(default=None, init=False, repr=False)
    _geometry_revision: int = field(default=0, init=False, repr=False)
    _edge_revision: int = field(default=0, init=False, repr=False)
    _anchor_attempt: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _axes_attempt: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _observed_axes: object | None = field(default=None, init=False, repr=False)
    _shared_shape_key: tuple | None = field(default=None, init=False, repr=False)
    _shared_shapes: dict = field(default_factory=dict, init=False, repr=False)

    def edge_positions(self):
        if self._edge_positions is None or len(self._edge_positions) > len(self.edges):
            self._edge_positions = np.array([[edge.start, edge.end] for edge in self.edges]).reshape((-1, 2, 2))
        elif len(self._edge_positions) < len(self.edges):
            added = np.array([[edge.start, edge.end] for edge in self.edges[len(self._edge_positions):]])
            self._edge_positions = np.concatenate((self._edge_positions, added))
        return self._edge_positions

    def tree_positions(self):
        if self._tree_positions is None or len(self._tree_positions) > len(self.trees):
            self._tree_positions = np.array([tree.position for tree in self.trees]).reshape((-1, 2))
        elif len(self._tree_positions) < len(self.trees):
            added = np.array([tree.position for tree in self.trees[len(self._tree_positions):]])
            self._tree_positions = np.concatenate((self._tree_positions, added))
        return self._tree_positions

    def boundary_positions(self, thickness):
        if self.world_size is None:
            return np.empty((0, 2, 2))
        key = (*self.world_size, thickness)
        if self._boundary_key != key:
            width, height = self.world_size
            self._boundary_positions = np.array(
                [((0, y), (width, y)) for y in (0, thickness, height - thickness, height)]
                + [((x, 0), (x, height)) for x in (0, thickness, width - thickness, width)]
            )
            self._boundary_key = key
        return self._boundary_positions


class WorldEstimator:
    def __init__(self, config: EstimatorConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.poses: dict[int, EstimatedPose] = {}
        self.groups: dict[int, MapGroup] = {}
        self.last_actions = {}
        self.last_time = None
        self.links: dict[tuple[int, int], SightingLink] = {}
        self.known_width = self.known_height = None
        self._dimension_conflict = False
        self._shared_merge_time = None
        self._shared_pair_attempts = {}

    def remember_actions(self, actions):
        self.last_actions = {action.agent_id: action for action in actions}

    def _cell(self, position):
        return tuple(np.floor(position / self.config.visited_cell_size).astype(int))

    def _correct_from_trees(self, pose, observations):
        observations = observation_geometry(observations)
        if not observations.trees:
            return
        group = self.groups[pose.group_id]
        trees = group.trees
        if not trees:
            return
        positions = group.tree_positions()
        candidates, used = [], set()
        for offset in observations.tree_offsets(pose.heading):
            distances = np.linalg.norm(positions - (pose.position + offset), axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= self.config.landmark_match_radius and index not in used:
                used.add(index)
                candidates.append(positions[index] - offset)
        if len(candidates) < self.config.minimum_landmark_matches:
            return
        candidates = np.array(candidates)
        center = np.median(candidates, axis=0)
        residuals = np.linalg.norm(candidates - center, axis=1)
        consistent = candidates[residuals <= self.config.landmark_consensus_tolerance]
        if len(consistent) >= self.config.minimum_landmark_matches:
            pose.position = np.mean(consistent, axis=0)
            pose.uncertainty = self.config.sighting_uncertainty

    def _correct_from_edges(self, pose, observations):
        """Use both endpoints of a static segment to correct collision drift.

        A segment carries more information than a distance to an infinite wall.
        Multiple matched segments must agree on the same translation.
        """
        group = self.groups[pose.group_id]
        observations = observation_geometry(observations)
        if not observations.edges:
            return False
        self._correct_from_boundary_endpoints(pose, observations)
        if self.config.relocalize_stones:
            self._relocalize_from_stones(pose, observations)
        edges = group.edges
        if not edges:
            return False
        stored = group.edge_positions()
        candidates, used = [], set()
        tolerance = self.config.landmark_consensus_tolerance
        for offsets in observations.edge_offsets(pose.heading):
            errors = np.linalg.norm(stored - (pose.position + offsets), axis=2).max(axis=1)
            if self.config.relocalize_stones:
                shape_error = np.linalg.norm((stored[:, 1] - stored[:, 0]) - (offsets[1] - offsets[0]), axis=1)
                errors = np.where(shape_error <= 1e-5, errors, np.inf)
            index = int(np.argmin(errors))
            radius = self.config.landmark_match_radius + (1e-8 if self.config.relocalize_stones else 0)
            if errors[index] > radius or index in used:
                continue
            # Reject wrong length/orientation even if its center is nearby.
            translations = stored[index] - offsets
            if np.linalg.norm(translations[0] - translations[1]) > tolerance:
                continue
            used.add(index)
            candidates.append(translations.mean(axis=0))
        if not candidates:
            return False
        candidates = np.array(candidates)
        center = np.median(candidates, axis=0)
        agreed = candidates[np.linalg.norm(candidates - center, axis=1) <= tolerance]
        if len(agreed) < len(candidates) / 2 or not len(agreed):
            return False
        pose.position = agreed.mean(axis=0)
        pose.uncertainty = self.config.sighting_uncertainty
        return True

    def _correct_from_boundary_endpoints(self, pose, observations):
        """A full vertical wall fixes y; a full horizontal wall fixes x.

        These constraints do not depend on which side of the world is visible,
        or on being within the ordinary landmark matching radius. They allow
        recovery after a collision without seeding a shifted duplicate wall.
        """
        group = self.groups[pose.group_id]
        if not group.anchored:
            return
        width, height = self._dimensions(group)
        xs, ys = [], []
        for start, end in observation_geometry(observations).edge_offsets(pose.heading):
            vector = end - start
            # The protocol supplies exact geometry; do not equate similar stones.
            if width is not None and np.linalg.norm(vector - (width, 0)) <= 1e-5:
                xs.append(-start[0])
            if height is not None and np.linalg.norm(vector - (0, height)) <= 1e-5:
                ys.append(-start[1])
        for axis, samples in enumerate((xs, ys)):
            if samples and max(samples) - min(samples) <= self.config.landmark_consensus_tolerance:
                pose.position[axis] = float(np.mean(samples))

    def _boundary_residual(self, group, start, end):
        if not group.anchored or np.linalg.norm(end - start) < self.config.boundary_minimum_length:
            return 0.0
        expected = group.boundary_positions(self.config.boundary_wall_thickness)
        if not len(expected):
            # The origin can be known before the far bounds. Validate the known
            # coordinate constraints without inventing an unknown dimension.
            width, height = self._dimensions(group)
            thickness = self.config.boundary_wall_thickness
            length = float(np.linalg.norm(end - start))
            ys = [0, thickness] + ([height - thickness, height] if height is not None else
                                  [max(float(start[1]), 2 * thickness)])
            xs = [0, thickness] + ([width - thickness, width] if width is not None else
                                  [max(float(start[0]), 2 * thickness)])
            expected = np.array([((0, y), (width if width is not None else length, y)) for y in ys]
                                + [((x, 0), (x, height if height is not None else length)) for x in xs])
        return float(np.linalg.norm(expected - (start, end), axis=2).max(axis=1).min())

    @staticmethod
    def _dimensions(group):
        return group.world_size if group.world_size is not None else (group.known_width, group.known_height)

    def _compatible_absolute_groups(self, first, second):
        combined = []
        for a, b in zip(self._dimensions(first), self._dimensions(second)):
            if a is not None and b is not None and abs(a - b) > self.config.landmark_consensus_tolerance:
                return False
            combined.append(a if a is not None else b)
        if all(value is not None for value in combined):
            proposed = MapGroup(-1, anchored=True, world_size=tuple(combined))
            if any(self._boundary_residual(proposed, edge.start, edge.end) > self.config.landmark_consensus_tolerance
                   for edge in first.edges + second.edges):
                return False
        return True

    def _transform_group(self, group_id, angle, offset):
        """Change the frame of ALL retained geometry together."""
        group = self.groups[group_id]
        for pose in self.poses.values():
            if pose.group_id == group_id:
                pose.position = rotate(pose.position, angle) + offset
                pose.origin = rotate(pose.origin, angle) + offset
                pose.heading = wrap(pose.heading + angle)
        for tree in group.trees:
            tree.position = rotate(tree.position, angle) + offset
        for edge in group.edges:
            edge.start = rotate(edge.start, angle) + offset
            edge.end = rotate(edge.end, angle) + offset
        group._edge_positions = group._tree_positions = None
        group._geometry_revision += 1
        group._edge_revision += 1
        visited = {}
        for point, potential, time in group.visited.values():
            point = rotate(point, angle) + offset
            cell = self._cell(point)
            if cell not in visited or time > visited[cell][2]:
                visited[cell] = (point, potential, time)
        group.visited = visited
        biomes = {}
        for sample in group.biomes.values():
            sample.position = rotate(sample.position, angle) + offset
            cell = self._cell(sample.position)
            if cell not in biomes or sample.last_seen > biomes[cell].last_seen:
                biomes[cell] = sample
        group.biomes = biomes
        group.frame_revision += 1
        group.revision += 1

    def _merge(self, moving_pose, reference_pose, target_position, target_heading):
        old_id, new_id = moving_pose.group_id, reference_pose.group_id
        old, new = self.groups[old_id], self.groups[new_id]
        if old.anchored and new.anchored:
            if not self._compatible_absolute_groups(old, new):
                return False
            # Independently anchored frames already use the same coordinates.
            # A noisy agent sighting must not rotate or shift an absolute map.
            target_position, target_heading = moving_pose.position.copy(), moving_pose.heading
        angle = wrap(target_heading - moving_pose.heading)
        offset = target_position - rotate(moving_pose.position, angle)
        self._transform_group(old_id, angle, offset)
        if old.anchored and new.anchored:
            dimensions = [a if a is not None else b
                          for a, b in zip(self._dimensions(new), self._dimensions(old))]
            new.known_width, new.known_height = dimensions
            if all(value is not None for value in dimensions):
                new.world_size = tuple(dimensions)
        for pose in self.poses.values():
            if pose.group_id == old_id:
                if not (old.anchored and new.anchored):
                    pose.uncertainty += reference_pose.uncertainty + self.config.sighting_uncertainty
                pose.group_id = new_id
        for tree in old.trees:
            self._observe_tree(new, tree.position, tree.last_seen)
        for position, potential, time in old.visited.values():
            cell = self._cell(position)
            if cell not in new.visited or time > new.visited[cell][2]:
                new.visited[cell] = (position, potential, time)
        for edge in old.edges:
            self._observe_edge(new, edge.start, edge.end, edge.last_seen, edge.sightings)
        for sample in old.biomes.values():
            cell = self._cell(sample.position)
            if cell not in new.biomes or sample.last_seen > new.biomes[cell].last_seen:
                new.biomes[cell] = sample
        # Receiving another map changes its contents, not the reference frame.
        # Existing residents/scouts can keep targets and origins in this frame.
        new.revision += 1
        del self.groups[old_id]
        return True

    def _observe_edge(self, group, start, end, time, sightings=1, *, validated=False):
        if not validated and self._boundary_residual(group, start, end) > self.config.landmark_consensus_tolerance:
            return
        if group.edges:
            stored = group.edge_positions()
            distances = np.linalg.norm(stored - np.array((start, end)), axis=2).max(axis=1)
            if self.config.relocalize_stones:
                shape_error = np.linalg.norm((stored[:, 1] - stored[:, 0]) - (end - start), axis=1)
                distances = np.where(shape_error <= 1e-5, distances, np.inf)
            index = int(np.argmin(distances))
            if distances[index] <= self.config.landmark_consensus_tolerance:
                edge = group.edges[index]
                edge.last_seen = max(edge.last_seen, time)
                edge.sightings += sightings
                return
        group.edges.append(EdgeLandmark(start.copy(), end.copy(), time, sightings))
        group._geometry_revision += 1
        group._edge_revision += 1

    def _pool_boundary_dimensions(self):
        """Share translation-invariant observed lengths across disconnected maps."""
        widths = [] if self.known_width is None else [self.known_width]
        heights = [] if self.known_height is None else [self.known_height]
        for group in self.groups.values():
            if group.anchored and group.world_size is not None:
                widths.append(group.world_size[0])
                heights.append(group.world_size[1])
                continue
            attempt = (group._edge_revision, len(group.edges))
            if group._axes_attempt != attempt:
                group._axes_attempt = attempt
                group._observed_axes = infer_directed_boundary_axes(
                    group.edges, self.config.boundary_minimum_length, self.config.boundary_wall_thickness)
            axes = group._observed_axes
            if axes is not None:
                if axes.known_width is not None:
                    widths.append(axes.known_width)
                if axes.known_height is not None:
                    heights.append(axes.known_height)
            if group.anchored:
                width, height = self._dimensions(group)
                if width is not None:
                    widths.append(width)
                if height is not None:
                    heights.append(height)
        if any(values and max(values) - min(values) > self.config.landmark_consensus_tolerance
               for values in (widths, heights)):
            self._dimension_conflict = True
            return
        if not self._dimension_conflict:
            self.known_width = widths[0] if widths else None
            self.known_height = heights[0] if heights else None

    def _anchor_groups(self, observations=None):
        if not self.config.anchor_boundaries:
            return
        if self.config.anchor_single_boundary:
            self._pool_boundary_dimensions()
        for group_id, group in list(self.groups.items()):
            if group.anchored:
                if self.config.anchor_single_boundary and not self._dimension_conflict:
                    width, height = self._dimensions(group)
                    group.known_width = width if width is not None else self.known_width
                    group.known_height = height if height is not None else self.known_height
                    if group.world_size is None and group.known_width is not None and group.known_height is not None:
                        proposed = MapGroup(-1, anchored=True, world_size=(group.known_width, group.known_height))
                        if all(self._boundary_residual(proposed, edge.start, edge.end) <= self.config.landmark_consensus_tolerance
                               for edge in group.edges):
                            group.world_size = proposed.world_size
                            group.revision += 1
                continue
            attempt = (group._edge_revision, len(group.edges))
            frame = None
            if group._anchor_attempt != attempt:
                group._anchor_attempt = attempt
                frame = infer_boundary_frame(
                    group.edges, minimum_length=self.config.boundary_minimum_length,
                    wall_thickness=self.config.boundary_wall_thickness,
                    tolerance=self.config.landmark_consensus_tolerance,
                )
            if (frame is None and self.config.anchor_single_boundary and not self._dimension_conflict
                    and observations is not None and group._observed_axes is not None):
                candidates = []
                for pose in sorted(self.poses.values(), key=lambda pose: pose.agent_id):
                    if pose.group_id != group_id or pose.uncertainty > self.config.max_position_uncertainty:
                        continue
                    observed = observations.get(pose.agent_id)
                    if observed is None:
                        continue
                    offsets = observed.edge_offsets(pose.heading)
                    if not len(offsets) or not np.any(np.linalg.norm(offsets[:, 1] - offsets[:, 0], axis=1)
                                                      >= self.config.boundary_minimum_length):
                        continue
                    edges = [EdgeLandmark(pose.position + start, pose.position + end, self.last_time or 0)
                             for start, end in offsets]
                    candidate = infer_single_boundary_frame(
                        edges, pose.position, orientation_edges=group.edges,
                        known_width=self.known_width, known_height=self.known_height,
                        minimum_length=self.config.boundary_minimum_length,
                        wall_thickness=self.config.boundary_wall_thickness,
                        tolerance=self.config.landmark_consensus_tolerance,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                if candidates and all(abs(wrap(candidate.angle - candidates[0].angle)) <= 1e-7 and
                                      np.linalg.norm(candidate.offset - candidates[0].offset) <= self.config.landmark_consensus_tolerance
                                      for candidate in candidates):
                    frame = candidates[0]
            if frame is None:
                continue
            self._transform_group(group_id, frame.angle, frame.offset)
            group.anchored, group.world_size = True, frame.world_size
            group.known_width, group.known_height = (frame.world_size if frame.world_size is not None else
                                                   (frame.known_width, frame.known_height))
        # Independently anchored components now have an observed common frame;
        # they can merge even if their agents have never met.
        anchored = sorted(key for key, group in self.groups.items() if group.anchored)
        if len(anchored) < 2:
            return
        reference_id = anchored[0]
        reference = min((p for p in self.poses.values() if p.group_id == reference_id),
                        key=lambda p: p.agent_id)
        for group_id in anchored[1:]:
            if not self._compatible_absolute_groups(self.groups[group_id], self.groups[reference_id]):
                continue
            moving = min((p for p in self.poses.values() if p.group_id == group_id),
                         key=lambda p: p.agent_id)
            self._merge(moving, reference, moving.position.copy(), moving.heading)

    def _observe_tree(self, group, position, time):
        # A tight gate avoids treating two nearby trees as the same landmark.
        if group.trees:
            distances = np.linalg.norm(group.tree_positions() - position, axis=1)
            index = int(np.argmin(distances))
            if distances[index] <= self.config.landmark_consensus_tolerance:
                group.trees[index].last_seen = max(group.trees[index].last_seen, time)
                return
        group.trees.append(TreeLandmark(position.copy(), time))
        group._geometry_revision += 1

    def _shared_shape_index(self, group):
        """Index exact directed stone lengths, independent of recency ordering.

        Full boundary faces and their thickness-sized caps are repetitive map
        features. Boundary anchoring handles them separately; they must never
        serve as an identity for an otherwise unconnected cluster.
        """
        key = (group._edge_revision, len(group.edges), self.config.boundary_minimum_length,
               self.config.boundary_wall_thickness)
        if group._shared_shape_key == key:
            return group._shared_shapes
        tolerance = 1e-5
        shapes = []
        for edge in group.edges:
            start, end = tuple(map(float, edge.start)), tuple(map(float, edge.end))
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if (not all(math.isfinite(value) for value in (*start, *end))
                    or length <= tolerance or length >= self.config.boundary_minimum_length
                    or abs(length - self.config.boundary_wall_thickness) <= tolerance):
                continue
            shapes.append((length, start, end, math.atan2(dy, dx), (dx / length, dy / length)))
        index = {}
        for identifier, shape in enumerate(sorted(shapes)):
            length, start, end, angle, direction = shape
            index.setdefault(math.floor(length / tolerance), []).append(
                (identifier, length, start, end, angle, direction)
            )
        group._shared_shape_key, group._shared_shapes = key, index
        return index

    def _known_stone_candidates(self, pose, observations):
        """Possible translations for exact directed shapes, cached per view."""
        group = self.groups[pose.group_id]
        key = (pose.group_id, group._edge_revision, len(group.edges), pose.heading)
        if observations._stone_key == key:
            return observations._stone_candidates
        index = self._shared_shape_index(group)
        candidates = []
        for offsets in observations.edge_offsets(pose.heading):
            vector = offsets[1] - offsets[0]
            length = math.hypot(*vector)
            if length <= 1e-5 or length >= self.config.boundary_minimum_length:
                continue
            signature = math.floor(length / 1e-5)
            positions = []
            for neighbor in (signature - 1, signature, signature + 1):
                for _, _, start, end, _, _ in index.get(neighbor, ()):
                    if math.hypot(end[0] - start[0] - vector[0], end[1] - start[1] - vector[1]) <= 1e-5:
                        positions.append(np.asarray(start) - offsets[0])
            if positions:
                candidates.append(((vector[0] / length, vector[1] / length), np.array(positions)))
        observations._stone_key, observations._stone_candidates = key, candidates
        return candidates

    def _stone_residual(self, pose, observations):
        return max((float(np.linalg.norm(positions - pose.position, axis=1).min())
                    for _, positions in self._known_stone_candidates(pose, observations)), default=0.0)

    def _stone_along_residual(self, pose, observations):
        """Only displacement along a known segment contradicts its opposite face.

        The protocol directs both horizontal rectangle faces along +x and both
        vertical faces along +y. An unseen opposite face therefore has identical
        along-segment endpoints, but a different normal offset. Do not confuse
        that legitimate new face with a bad pose. Repeated shapes are compatible
        if ANY retained placement agrees along the observed segment.

        A single normal collision remains indistinguishable from an unseen face;
        full recovery still needs the nonparallel consensus below.
        """
        return max((float(np.abs((positions - pose.position) @ direction).min())
                    for direction, positions in self._known_stone_candidates(pose, observations)), default=0.0)

    def _relocalize_from_stones(self, pose, observations):
        """Recover collision displacement without trusting a shifted duplicate.

        Beyond the ordinary proximity gate, at least two nonparallel observed
        segments must imply one placement agreeing with ALL known observed
        shapes. A single side cannot distinguish opposite faces of a stone.
        """
        candidates = self._known_stone_candidates(pose, observations)
        if len(candidates) < 2 or sum(len(positions) for _, positions in candidates) > self.config.shared_landmark_max_candidates:
            return False
        if self._stone_residual(pose, observations) <= self.config.landmark_match_radius + 1e-8:
            return False
        if not any(abs(a[0] * b[1] - a[1] * b[0]) > 0.25
                   for index, (a, _) in enumerate(candidates) for b, _ in candidates[index + 1:]):
            return False
        tolerance = self.config.landmark_consensus_tolerance
        group, valid = self.groups[pose.group_id], []
        for _, positions in candidates:
            for seed in positions:
                agreeing = []
                for _, options in candidates:
                    distances = np.linalg.norm(options - seed, axis=1)
                    nearest = int(np.argmin(distances))
                    if distances[nearest] > tolerance + 1e-8:
                        break
                    agreeing.append(options[nearest])
                if len(agreeing) != len(candidates):
                    continue
                position = np.mean(agreeing, axis=0)
                if any(self._boundary_residual(group, position + start, position + end) > tolerance
                       for start, end in observations.edge_offsets(pose.heading)):
                    continue
                if any(np.linalg.norm(position - other) > tolerance for _, other in valid):
                    return False
                residual = float(np.square(np.array(agreeing) - position).sum())
                valid.append((residual, position))
        if not valid:
            return False
        _, pose.position = min(valid, key=lambda item: (item[0], tuple(item[1])))
        return True

    def _shared_landmark_transform(self, moving, reference):
        """Return one unambiguous transform supported by nonparallel stones.

        Random stone dimensions act as observed geometric fingerprints. Exact
        directed lengths generate candidates; BOTH endpoints of at least two
        distinct nonparallel segments must then agree. Repeated shapes that
        support multiple placements are rejected, even if one has more votes.
        Work is bounded; overflowing the candidate budget rejects the pair.
        """
        shape_tolerance, angle_tolerance = 1e-5, 1e-7
        moving_index = self._shared_shape_index(moving)
        reference_index = self._shared_shape_index(reference)
        matches = []
        for signature, sources in moving_index.items():
            for neighbor in (signature - 1, signature, signature + 1):
                for target in reference_index.get(neighbor, ()):
                    for source in sources:
                        if abs(source[1] - target[1]) > shape_tolerance:
                            continue
                        angle = wrap(target[4] - source[4])
                        cosine, sine = math.cos(angle), math.sin(angle)
                        x, y = source[2]
                        offset = (target[2][0] - (x * cosine - y * sine),
                                  target[2][1] - (x * sine + y * cosine))
                        matches.append((source, target, angle, offset, cosine, sine))
                        if len(matches) > self.config.shared_landmark_max_candidates:
                            return None
        if len(matches) < 2:
            return None
        tolerance = self.config.landmark_consensus_tolerance
        tested, valid = [], []
        for _, _, angle, offset, cosine, sine in matches:
            # Identical hypotheses arise from each agreeing corner/segment.
            if any(abs(wrap(angle - other_angle)) <= angle_tolerance
                   and math.dist(offset, other_offset) <= shape_tolerance
                   for other_angle, other_offset in tested):
                continue
            tested.append((angle, offset))
            supporting, error_sum = [], 0.0
            for source, target, other_angle, _, _, _ in matches:
                if abs(wrap(angle - other_angle)) > angle_tolerance:
                    continue
                error = max(math.hypot(x * cosine - y * sine + offset[0] - rx,
                                       x * sine + y * cosine + offset[1] - ry)
                            for (x, y), (rx, ry) in zip(source[2:4], target[2:4]))
                if error <= tolerance:
                    supporting.append((source[0], target[0], target[5]))
                    error_sum += error * error
            independent = any(
                source_id != other_source and target_id != other_target
                and abs(direction[0] * other_direction[1] - direction[1] * other_direction[0]) > 0.25
                for index, (source_id, target_id, direction) in enumerate(supporting)
                for other_source, other_target, other_direction in supporting[index + 1:]
            )
            if not independent:
                continue
            if reference.anchored and any(
                self._boundary_residual(reference, rotate(edge.start, angle) + offset,
                                        rotate(edge.end, angle) + offset) > tolerance
                for edge in moving.edges
            ):
                continue
            if any(abs(wrap(angle - other[0])) > angle_tolerance
                   or math.dist(offset, other[1]) > tolerance for other in valid):
                return None
            valid.append((angle, offset, len(supporting), error_sum))
        if not valid:
            return None
        angle, offset, _, _ = min(valid, key=lambda item: (-item[2], item[3], item[0], item[1]))
        return angle, np.array(offset)

    def _merge_shared_landmarks(self, sim_time):
        if not self.config.shared_landmark_merges or len(self.groups) < 2:
            return
        if (self._shared_merge_time is not None
                and sim_time - self._shared_merge_time < self.config.shared_landmark_interval_seconds):
            return
        self._shared_merge_time = sim_time
        self._shared_pair_attempts = {pair: revision for pair, revision in self._shared_pair_attempts.items()
                                      if all(group_id in self.groups for group_id in pair)}
        group_ids = sorted(self.groups)
        attempted = 0
        for index, first_id in enumerate(group_ids):
            for second_id in group_ids[index + 1:]:
                if first_id not in self.groups or second_id not in self.groups:
                    continue
                first, second = self.groups[first_id], self.groups[second_id]
                if first.anchored and second.anchored:
                    continue  # Independent canonical frames are merged by boundary anchoring.
                pair = (first_id, second_id)
                revision = (first._edge_revision, len(first.edges), first.anchored,
                            second._edge_revision, len(second.edges), second.anchored)
                if self._shared_pair_attempts.get(pair) == revision:
                    continue
                if attempted >= self.config.shared_landmark_max_pairs:
                    return
                attempted += 1
                self._shared_pair_attempts[pair] = revision
                reference, moving = (second, first) if second.anchored else (first, second)
                transform = self._shared_landmark_transform(moving, reference)
                if transform is None:
                    continue
                angle, offset = transform
                reference_pose = min((pose for pose in self.poses.values() if pose.group_id == reference.group_id),
                                     key=lambda pose: (pose.uncertainty, pose.agent_id))
                moving_pose = min((pose for pose in self.poses.values() if pose.group_id == moving.group_id),
                                  key=lambda pose: pose.agent_id)
                self._merge(moving_pose, reference_pose, rotate(moving_pose.position, angle) + offset,
                            wrap(moving_pose.heading + angle))

    def _correction_state(self, pose):
        return (pose.group_id, self.groups[pose.group_id]._geometry_revision,
                pose.heading, pose.position[0], pose.position[1], pose.uncertainty)

    @staticmethod
    def _retain_landmarks(group, field_name, cache_name, limit, expiry=None):
        """Keep baseline recency order without rebuilding unchanged geometry."""
        landmarks = getattr(group, field_name)
        order = sorted((index for index, landmark in enumerate(landmarks)
                        if expiry is None or expiry[0] - landmark.last_seen <= expiry[1]),
                       key=lambda index: landmarks[index].last_seen)[-limit:]
        if len(order) == len(landmarks) and all(index == value for index, value in enumerate(order)):
            return
        cached = getattr(group, cache_name)
        if cached is not None and len(cached) == len(landmarks):
            setattr(group, cache_name, cached[order])
        else:
            setattr(group, cache_name, None)
        setattr(group, field_name, [landmarks[index] for index in order])
        if field_name == "edges" and len(order) != len(landmarks):
            group._edge_revision += 1

    def update(self, agent_states, sim_time, *, observe=True):
        if not agent_states:
            self.reset()
            return
        if self.last_time is not None and sim_time < self.last_time:
            self.reset()
        if any(state["agent_id"] in self.poses and state["age"] < self.poses[state["agent_id"]].age
               for state in agent_states):
            self.reset()
        # Retries must not integrate movement or count the same sighting twice.
        if self.last_time is not None and sim_time == self.last_time:
            return
        # Prediction still runs every action tick when shared-map sensing is
        # throttled. New agents always need a real observation/alignment pass.
        observe = observe or any(state["agent_id"] not in self.poses for state in agent_states)
        agent_states = sorted(agent_states, key=lambda state: state["agent_id"])
        # The supplied simulator can skip an observation/age update when a
        # preceding agent dies while its list is being traversed. Its action
        # still ran. Integrate that motion, but never reuse the cached bearings
        # as if they were measured at the new pose. Generic callers can disable
        # this protocol-specific freshness signal.
        stale_ids = {state["agent_id"] for state in agent_states
                     if self.config.detect_stale_observations
                     and state["agent_id"] in self.poses
                     and state["age"] == self.poses[state["agent_id"]].age}
        agent_states = [dict(state, observations=[]) if state["agent_id"] in stale_ids else state
                        for state in agent_states]
        geometry = ({state["agent_id"]: ObservationGeometry(state["observations"]) for state in agent_states}
                    if observe else {})
        unchanged_corrections = {}
        living = {state["agent_id"] for state in agent_states}
        self.poses = {key: pose for key, pose in self.poses.items() if key in living}
        for state in agent_states:
            agent_id = state["agent_id"]
            pose = self.poses.get(agent_id)
            if pose is None:
                # IDs are unique within an episode; each unaligned agent gets a
                # fresh gauge, never an invented position in another group's map.
                pose = EstimatedPose(agent_id, agent_id, np.zeros(2))
                self.poses[agent_id] = pose
                self.groups[agent_id] = MapGroup(agent_id)
            elif agent_id in self.last_actions:
                action = self.last_actions[agent_id]
                factor = self.config.biome_movement_factors.get(
                    pose.biome, self.config.unknown_biome_movement_factor
                )
                displacement = action.move_distance * factor
                pose.position += rotate((displacement, 0), pose.heading + action.move_direction)
                pose.heading = wrap(pose.heading + action.turn_angle)
                pose.uncertainty += abs(displacement) * self.config.odometry_error_per_unit
            pose.biome, pose.age = state["biome"], state["age"]
            if not observe:
                continue
            old_x, old_y = pose.position
            if not self._correct_from_edges(pose, geometry[agent_id]):
                self._correct_from_trees(pose, geometry[agent_id])
            if pose.position[0] == old_x and pose.position[1] == old_y:
                unchanged_corrections[agent_id] = self._correction_state(pose)

        if not observe:
            live_groups = {pose.group_id for pose in self.poses.values()}
            self.groups = {key: group for key, group in self.groups.items() if key in live_groups}
            self.last_time = sim_time
            return

        # Agent observations provide both an identified relative position and
        # relative heading: heading_j = heading_i + bearing_ij + pi - rel_dir.
        for state in agent_states:
            observer = self.poses[state["agent_id"]]
            for observation in geometry[observer.agent_id].agents:
                target = self.poses.get(observation["id"])
                if target is None or target is observer:
                    continue
                coincident = observation["distance"] <= 1e-6
                relative_heading = (None if coincident else
                                    wrap(observation["angle"] + math.pi - observation["rel_dir"]))
                key = (observer.agent_id, target.agent_id)
                previous = self.links.get(key)
                self.links[key] = SightingLink(
                    *key, previous.first_seen if previous else sim_time, sim_time,
                    previous.count + 1 if previous else 1, float(observation["distance"]),
                    float(observation["angle"]), relative_heading,
                )
                # At zero separation there is no line joining the two agents:
                # the two atan2 bearings are not opposites. Newborns often
                # overlap a parent; wait for separation before aligning frames.
                if coincident:
                    continue
                target_heading = wrap(observer.heading + relative_heading)
                target_position = observer.position + rotate(relative_vector(observation), observer.heading)
                if target.group_id != observer.group_id:
                    observer_group, target_group = self.groups[observer.group_id], self.groups[target.group_id]
                    # An established absolute frame must survive a new sighting,
                    # even when an unanchored component has a smaller ID.
                    prefer_observer = (observer_group.anchored and not target_group.anchored) or (
                        observer_group.anchored == target_group.anchored
                        and observer.group_id < target.group_id
                    )
                    if prefer_observer:
                        self._merge(target, observer, target_position, target_heading)
                    else:
                        heading = wrap(target.heading - relative_heading)
                        position = target.position - rotate(relative_vector(observation), heading)
                        self._merge(observer, target, position, heading)
                elif target.uncertainty > observer.uncertainty + self.config.sighting_uncertainty:
                    target.position = target_position
                    target.heading = target_heading
                    target.uncertainty = observer.uncertainty + self.config.sighting_uncertainty

        for state in agent_states:
            pose = self.poses[state["agent_id"]]
            group = self.groups[pose.group_id]
            observed = geometry[pose.agent_id]
            # The second pass still runs after any pose/frame/geometry change.
            # Exact no-op first passes can be reused without changing matching.
            if unchanged_corrections.get(pose.agent_id) != self._correction_state(pose):
                if not self._correct_from_edges(pose, observed):
                    self._correct_from_trees(pose, observed)
            observed_segments = pose.position + observed.edge_offsets(pose.heading)
            residual = max((self._boundary_residual(group, start, end)
                            for start, end in observed_segments), default=0)
            if self.config.relocalize_stones:
                stone_residual = self._stone_along_residual(pose, observed)
                if stone_residual > self.config.landmark_match_radius + 1e-8:
                    residual = max(residual, stone_residual)
            if residual > self.config.landmark_consensus_tolerance:
                # A contradiction is evidence of a bad pose, not a new wall.
                # Suppress ALL new landmarks and biome samples from that pose.
                pose.uncertainty = max(pose.uncertainty, self.config.max_position_uncertainty + residual)
            if pose.uncertainty > self.config.max_position_uncertainty or pose.agent_id in stale_ids:
                continue
            potential = self.config.biome_food_potential.get(
                pose.biome, self.config.unknown_biome_food_potential
            )
            cell = self._cell(pose.position)
            group.visited[cell] = (pose.position.copy(), potential, sim_time)
            group.biomes[cell] = BiomeSample(pose.position.copy(), pose.biome, sim_time, pose.uncertainty)
            for start, end in observed_segments:
                self._observe_edge(group, start, end, sim_time, validated=True)
            for offset in observed.tree_offsets(pose.heading):
                position = pose.position + offset
                self._observe_tree(group, position, sim_time)
        live_groups = {pose.group_id for pose in self.poses.values()}
        self.groups = {key: group for key, group in self.groups.items() if key in live_groups}
        self._merge_shared_landmarks(sim_time)
        self._anchor_groups(geometry)
        for group in self.groups.values():
            self._retain_landmarks(group, "trees", "_tree_positions", self.config.max_trees_per_group,
                                   expiry=(sim_time, self.config.tree_memory_seconds))
            group.visited = dict(sorted(group.visited.items(), key=lambda pair: pair[1][2])[
                -self.config.max_visited_cells_per_group:
            ])
            group.biomes = dict(sorted(group.biomes.items(), key=lambda pair: pair[1].last_seen)[
                -self.config.max_visited_cells_per_group:
            ])
            self._retain_landmarks(group, "edges", "_edge_positions", self.config.max_edges_per_group)
        self.links = dict(sorted(self.links.items(), key=lambda pair: pair[1].last_seen)[
            -self.config.max_sighting_links:
        ])
        self.last_time = sim_time

    def snapshot(self):
        """Public, JSON-serializable map; no ground-truth alignment is applied."""
        return {
            "sim_time": self.last_time,
            "biome_cell_size": self.config.visited_cell_size,
            "groups": [{
                "group_id": group_id, "anchored": group.anchored,
                "world_size": list(group.world_size) if group.world_size else None,
                "known_width": self._dimensions(group)[0], "known_height": self._dimensions(group)[1],
                "frame_revision": group.frame_revision,
                "trees": [{"position": tree.position.tolist(), "last_seen": tree.last_seen}
                          for tree in group.trees],
                "edges": [{"start": edge.start.tolist(), "end": edge.end.tolist(),
                           "last_seen": edge.last_seen, "sightings": edge.sightings}
                          for edge in group.edges],
                "biomes": [{"position": sample.position.tolist(), "biome": sample.biome,
                            "last_seen": sample.last_seen, "uncertainty": sample.uncertainty}
                           for _, sample in sorted(group.biomes.items())],
            } for group_id, group in sorted(self.groups.items())],
            "agents": [{"agent_id": pose.agent_id, "group_id": pose.group_id,
                        "position": pose.position.tolist(), "heading": pose.heading,
                        "origin": pose.origin.tolist(), "uncertainty": pose.uncertainty}
                       for _, pose in sorted(self.poses.items())],
            "links": [{"observer_id": link.observer_id, "target_id": link.target_id,
                       "first_seen": link.first_seen, "last_seen": link.last_seen,
                       "count": link.count, "distance": link.distance,
                       "angle": link.angle, "relative_heading": link.relative_heading}
                      for _, link in sorted(self.links.items())],
        }
