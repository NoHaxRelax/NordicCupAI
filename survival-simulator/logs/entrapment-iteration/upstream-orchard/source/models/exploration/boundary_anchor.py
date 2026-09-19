"""Recover a world frame from the simulator's observed boundary segments.

This deliberately uses a property of the observation protocol: ``spawn_obstacle``
orders both horizontal edges toward world +x and both vertical edges toward
world +y, and observations retain the complete original endpoints. Together,
two perpendicular long edges identify the axes, origin, and map dimensions.
Arbitrarily ordered or clipped segments would not provide this information.

No environment object, true agent position, or hidden map dimensions are read.
The minimum boundary length and known wall thickness describe the map format;
ordinary generated stones are shorter than the default length threshold.
"""

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterable, Protocol

import numpy as np


class BoundaryEdge(Protocol):
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class BoundaryFrame:
    """Transform local points by ``rotate(point, angle) + offset``."""

    angle: float
    offset: np.ndarray
    world_size: tuple[float, float] | None
    known_width: float | None = None
    known_height: float | None = None


@dataclass(frozen=True)
class DirectedBoundaryAxes:
    """Observed world-axis orientation and any observed full wall lengths."""

    angle: float
    known_width: float | None
    known_height: float | None


def infer_directed_boundary_axes(edges, minimum_length=200.0, wall_thickness=30.0):
    """Identify +x/+y from two nonparallel, positively directed segments.

    This stricter optional protocol path assumes exact, complete rectangle
    segments. Translation does not affect a wall's observed width or height.
    All available directions and repeated dimensions must agree.
    """
    vectors = []
    for edge in edges:
        segment = np.asarray((edge.start, edge.end), dtype=float)
        if segment.shape != (2, 2) or not np.isfinite(segment).all():
            continue
        vector = segment[1] - segment[0]
        length = float(np.linalg.norm(vector))
        if length > 1e-6:
            vectors.append((vector, length))
    if len(vectors) < 2:
        return None
    first, first_length = vectors[0]
    first_direction = first / first_length
    perpendicular = next((vector / length for vector, length in vectors[1:]
                          if abs(float(np.dot(first_direction, vector / length))) < 1e-7), None)
    if perpendicular is None:
        return None
    cross = first_direction[0] * perpendicular[1] - first_direction[1] * perpendicular[0]
    horizontal = first_direction if cross > 0 else perpendicular
    angle = -math.atan2(horizontal[1], horizontal[0])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    dimensions = [[], []]
    for vector, length in vectors:
        transformed = rotation @ vector
        errors = [np.linalg.norm(transformed - (length, 0)), np.linalg.norm(transformed - (0, length))]
        axis = int(np.argmin(errors))
        if errors[axis] > 1e-5:
            return None
        if length >= minimum_length:
            dimensions[axis].append(length)
    result = []
    for samples in dimensions:
        if samples and (max(samples) - min(samples) > 1e-5 or min(samples) <= 2 * wall_thickness):
            return None
        result.append(float(np.mean(samples)) if samples else None)
    return DirectedBoundaryAxes(angle, *result)


def infer_single_boundary_frame(
    edges, observer_position, *, orientation_edges=(), known_width=None, known_height=None,
    minimum_length=200.0, wall_thickness=30.0, tolerance=3.0,
):
    """Anchor a fresh inner-wall sighting using its observed side and axis basis.

    A top/left wall fixes the origin directly. A bottom/right wall additionally
    needs the perpendicular world dimension, learned from other full walls.
    Both positive axis directions must already be observed. Unknown dimensions
    remain None; absolute coordinates do not require knowing the far bounds.

    Rays return the nearest wall. The observer must be farther than one wall
    thickness from the observed face, excluding outer-face sightings by an
    agent inside a boundary wall (possible after a collision/clamp). Close or
    on-wall observations are intentionally insufficient for this optional path.
    """
    edges, orientation_edges = list(edges), list(orientation_edges)
    all_edges = orientation_edges + edges
    axes = infer_directed_boundary_axes(all_edges, minimum_length, wall_thickness)
    observer_position = np.asarray(observer_position, dtype=float)
    if axes is None or observer_position.shape != (2,) or not np.isfinite(observer_position).all():
        return None
    dimensions = [known_width, known_height]
    for axis, measured in enumerate((axes.known_width, axes.known_height)):
        if measured is not None:
            if dimensions[axis] is not None and abs(dimensions[axis] - measured) > tolerance:
                return None
            dimensions[axis] = measured if dimensions[axis] is None else dimensions[axis]
    if any(value is not None and (not math.isfinite(value) or value <= 2 * wall_thickness)
           for value in dimensions):
        return None
    cosine, sine = math.cos(axes.angle), math.sin(axes.angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    observer = rotation @ observer_position
    frames = []
    for edge in edges:
        segment = np.asarray((edge.start, edge.end), dtype=float)
        if segment.shape != (2, 2) or not np.isfinite(segment).all():
            continue
        segment = segment @ rotation.T
        vector = segment[1] - segment[0]
        if np.linalg.norm(vector) < minimum_length:
            continue
        axis = int(np.argmax(vector))
        normal = 1 - axis
        side = observer[normal] - segment[0, normal]
        if abs(side) <= wall_thickness + tolerance:
            continue
        if side > 0:
            face_coordinate = wall_thickness
        elif dimensions[normal] is not None:
            face_coordinate = dimensions[normal] - wall_thickness
        else:
            continue
        offset = -segment[0].copy()
        offset[normal] += face_coordinate
        absolute_observer = observer + offset
        if any(absolute_observer[i] < wall_thickness - tolerance or
               (dimensions[i] is not None and absolute_observer[i] > dimensions[i] - wall_thickness + tolerance)
               for i in range(2)):
            continue
        # Validate ALL retained long geometry in the proposed frame. For an
        # unknown far bound, at most one other parallel inner face is possible.
        unknown_far = {}
        valid = True
        for remembered in all_edges:
            stored = np.asarray((remembered.start, remembered.end), dtype=float)
            if stored.shape != (2, 2) or not np.isfinite(stored).all():
                continue
            stored = stored @ rotation.T + offset
            delta = stored[1] - stored[0]
            if np.linalg.norm(delta) < minimum_length:
                continue
            along = int(np.argmax(delta))
            across = 1 - along
            if (dimensions[along] is None or abs(stored[0, along]) > tolerance or
                    abs(stored[1, along] - dimensions[along]) > tolerance or
                    abs(delta[across]) > tolerance):
                valid = False
                break
            face = float(stored[0, across])
            allowed = [wall_thickness]
            if dimensions[across] is not None:
                allowed.append(dimensions[across] - wall_thickness)
            elif face > 2 * wall_thickness + tolerance:
                if across not in unknown_far:
                    unknown_far[across] = face
                allowed.append(unknown_far[across])
            if min(abs(face - expected) for expected in allowed) > tolerance:
                valid = False
                break
        if valid:
            if frames and np.linalg.norm(offset - frames[0]) > tolerance:
                return None
            frames.append(offset)
    if not frames:
        return None
    size = tuple(dimensions) if all(value is not None for value in dimensions) else None
    return BoundaryFrame(axes.angle, frames[0], size, *dimensions)


def infer_boundary_frame(
    edges: Iterable[BoundaryEdge],
    minimum_length: float = 200.0,
    wall_thickness: float = 30.0,
    tolerance: float = 3.0,
) -> BoundaryFrame | None:
    """Infer an absolute frame only when all usable long edges agree.

    ``start`` and ``end`` must be in the same local coordinate frame, with
    their observed ordering preserved. Short, degenerate, and nonfinite edges
    are ignored. A single wall, including its parallel faces, is insufficient.
    ``tolerance`` is the maximum endpoint error in world distance units. It
    also bounds the perpendicularity error over each pair's shorter segment.

    Both inner and outer boundary faces are accepted. Unrelated long segments
    or disagreement between remembered observations prevent an anchor. This
    conservative check assumes the configured length threshold separates the
    boundary from ordinary obstacles; it cannot recognize arbitrary large
    custom obstacles that exactly imitate the boundary geometry.
    """
    if not math.isfinite(minimum_length) or minimum_length <= 0:
        raise ValueError("minimum_length must be positive and finite")
    if not math.isfinite(wall_thickness) or wall_thickness < 0:
        raise ValueError("wall_thickness must be nonnegative and finite")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative and finite")

    segments = []
    for edge in edges:
        start, end = np.asarray(edge.start, dtype=float), np.asarray(edge.end, dtype=float)
        if start.shape != (2,) or end.shape != (2,):
            continue
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        vector = end - start
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length < minimum_length:
            continue
        segments.append((start, end, vector / length, length))

    best_frame, best_error = None, math.inf
    for first, second in combinations(segments, 2):
        perpendicular_error = abs(float(np.dot(first[2], second[2]))) * min(first[3], second[3])
        if perpendicular_error > tolerance + 1e-8:
            continue
        cross = first[2][0] * second[2][1] - first[2][1] * second[2][0]
        if abs(cross) < 0.5:
            continue
        horizontal, vertical = (first, second) if cross > 0 else (second, first)
        width, height = horizontal[3], vertical[3]
        if min(width, height) <= 2 * wall_thickness:
            continue
        angle = -math.atan2(horizontal[2][1], horizontal[2][0])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        horizontal_start = rotation @ horizontal[0]
        vertical_start = rotation @ vertical[0]
        offset = -np.array((horizontal_start[0], vertical_start[1]))

        # Directed endpoints identify the origin. Checking full segments also
        # rejects perpendicular stones and walls from incompatible local maps.
        expected = np.array(
            [((0, y), (width, y)) for y in (0, wall_thickness, height - wall_thickness, height)]
            + [((x, 0), (x, height)) for x in (0, wall_thickness, width - wall_thickness, width)]
        )
        errors = []
        for start, end, _, _ in segments:
            transformed = np.array((rotation @ start + offset, rotation @ end + offset))
            error = float(np.min(np.max(np.linalg.norm(expected - transformed, axis=2), axis=1)))
            if error > tolerance + 1e-8:
                break
            errors.append(error)
        if len(errors) != len(segments):
            continue
        total_error = sum(error * error for error in errors)
        if total_error < best_error:
            best_frame = BoundaryFrame(angle=angle, offset=offset, world_size=(width, height))
            best_error = total_error
    return best_frame
