"""Potential predator traps inferred from a group's observed rock faces.

The simulator exposes whole rectangle faces, but no obstacle identities. Two
perpendicular faces meeting at a corner determine a candidate rectangle. Extra
matching faces strengthen that evidence. These estimates never constitute a
guarantee that an unseen obstacle will not block a bait position or its approach.
"""

from dataclasses import dataclass, field
import copy
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.utils.trap_sites import AGENT_RADIUS, PREDATOR_RADIUS, find_trap_sites


class TrapInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    refresh_interval_seconds: float = Field(default=10., gt=0)
    face_tolerance: float = Field(default=.5, gt=0, le=2)
    safety: Literal["measured", "safe", "paranoid"] = "safe"
    max_sites: int = Field(default=64, ge=1, le=512)
    max_scan_cells: int = Field(default=2400000, ge=10000)


def observed_rectangles(edges, tolerance=.5, maximum_side=200.):
    """Recover rectangles only from adjoining complete horizontal/vertical faces.

    Parallel faces alone are ambiguous: they might belong to different rocks.
    Large boundary faces are handled separately from known world bounds. No
    assumption about an unseen opposite face's position comes from world truth.
    """
    horizontal, vertical = [], []
    for pair in np.asarray(edges, dtype=float).reshape((-1, 2, 2)):
        if not np.isfinite(pair).all():
            continue
        low, high = pair.min(axis=0), pair.max(axis=0)
        dx, dy = high - low
        if 2 * tolerance < dx < maximum_side and dy <= tolerance:
            horizontal.append((float(low[0]), float(high[0]), float(pair[:, 1].mean())))
        elif 2 * tolerance < dy < maximum_side and dx <= tolerance:
            vertical.append((float(low[1]), float(high[1]), float(pair[:, 0].mean())))
    if not horizontal or not vertical:
        return []
    horizontal = np.array(sorted(set(horizontal)))
    vertical = np.array(sorted(set(vertical)))
    candidates = []
    for x0, x1, y in horizontal:
        adjacent = ((np.minimum(abs(vertical[:, 2] - x0), abs(vertical[:, 2] - x1)) <= tolerance)
                    & (np.minimum(abs(vertical[:, 0] - y), abs(vertical[:, 1] - y)) <= tolerance))
        for y0, y1, x in vertical[adjacent]:
            h_spans = np.maximum(abs(horizontal[:, 0] - x0), abs(horizontal[:, 1] - x1)) <= tolerance
            v_spans = np.maximum(abs(vertical[:, 0] - y0), abs(vertical[:, 1] - y1)) <= tolerance
            faces = [bool(np.any(h_spans & (abs(horizontal[:, 2] - side) <= tolerance))) for side in (y0, y1)]
            faces += [bool(np.any(v_spans & (abs(vertical[:, 2] - side) <= tolerance))) for side in (x0, x1)]
            bounds = tuple(round(float(value), 5) for value in (x0, y0, x1, y1))
            candidates.append((sum(faces), bounds))
    # A rectangle seen from several corners is one obstacle, even when small
    # independent pose errors shift the reconstructed copies slightly.
    result, retained = [], []
    for faces, bounds in sorted(set(candidates), key=lambda row: (-row[0], row[1])):
        if any(max(abs(a - b) for a, b in zip(bounds, other)) <= tolerance for other in retained):
            continue
        retained.append(bounds)
        x0, y0, x1, y1 = bounds
        result.append(dict(bounds=list(bounds), observed_faces=faces,
                           rectangle=[x0, y0, x1 - x0, y1 - y0]))
    return sorted(result, key=lambda item: item["bounds"])


def _point_clear(point, edges, radius):
    if not len(edges):
        return True
    starts, vectors = edges[:, 0], edges[:, 1] - edges[:, 0]
    lengths = np.sum(vectors * vectors, axis=1)
    projection = np.clip(np.sum((np.asarray(point) - starts) * vectors, axis=1)
                         / np.maximum(lengths, 1e-12), 0, 1)
    distances = np.linalg.norm(starts + projection[:, None] * vectors - point, axis=1)
    return bool(np.all(distances >= radius - 1e-6))


@dataclass
class TrapLayer:
    frame: tuple
    status: str = "waiting for absolute coordinates"
    edge_key: tuple | None = None
    fingerprint: tuple | None = None
    next_check_at: float = 0.
    updated_at: float | None = None
    scan_count: int = 0
    stale: bool = False
    rectangles: list = field(default_factory=list)
    raw_sites: list = field(default_factory=list)
    sites: list = field(default_factory=list)


class TrapMapper:
    def __init__(self, config, boundary_thickness=30., maximum_rock_side=200.):
        self.config = config
        self.boundary_thickness = boundary_thickness
        self.maximum_rock_side = maximum_rock_side
        self.reset()

    def reset(self):
        self.layers = {}
        self.last_time = None

    def update(self, groups, now):
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        if not self.config.enabled:
            self.layers.clear()
            return
        self.layers = {key: layer for key, layer in self.layers.items() if key in groups}
        for key, group in sorted(groups.items()):
            size = group.world_size
            if size is None and group.known_width is not None and group.known_height is not None:
                size = (group.known_width, group.known_height)
            frame = (group.frame_revision, group.anchored, None if size is None else tuple(size))
            layer = self.layers.get(key)
            if layer is None or layer.frame != frame:
                layer = self.layers[key] = TrapLayer(frame)
            if not group.anchored:
                continue
            if size is None:
                layer.status = "waiting for observed world bounds"
                continue
            if (not all(math.isfinite(v) and v >= 1 for v in size)
                    or math.ceil(size[0]) * math.ceil(size[1]) > self.config.max_scan_cells):
                layer.status = "world bounds exceed scan budget"
                continue
            edge_key = (group._edge_revision, len(group.edges))
            if edge_key == layer.edge_key:
                continue
            # Old positions remain visible as stale estimates during the bounded
            # refresh interval. They must not be advertised as current targets.
            layer.stale = bool(layer.sites)
            layer.status = "geometry changed; refresh pending"
            if now < layer.next_check_at:
                continue
            edges = np.array([[edge.start, edge.end] for edge in group.edges], dtype=float).reshape((-1, 2, 2))
            rectangles = observed_rectangles(edges, self.config.face_tolerance, self.maximum_rock_side)
            width, height = map(lambda value: int(round(value)), size)
            rectangles = [item for item in rectangles if all(
                0 <= item["bounds"][axis] < item["bounds"][axis + 2] <= size[axis]
                for axis in (0, 1))]
            rocks = [tuple(item["rectangle"]) for item in rectangles]
            fingerprint = tuple(rocks)
            if fingerprint != layer.fingerprint:
                thickness = self.boundary_thickness
                # Bounds and wall thickness are public map-format constraints.
                # They can support edge slots once both dimensions are observed.
                boundaries = [(0., 0., width, thickness), (0., height - thickness, width, thickness),
                              (0., 0., thickness, height), (width - thickness, 0., thickness, height)]
                layer.raw_sites = (find_trap_sites(rocks + boundaries, width, height,
                    safety=self.config.safety, kinds=("wall", "slot", "shelter")) if rocks else [])
                layer.scan_count += int(bool(rocks))
                layer.fingerprint = fingerprint
            sites = []
            for site in layer.raw_sites:
                # Even an incomplete face can disqualify a previously open bait
                # or predator position. It need not form a whole rectangle first.
                if not _point_clear(site.bait, edges, AGENT_RADIUS):
                    continue
                if site.kind != "shelter" and not _point_clear(site.predator_side, edges, PREDATOR_RADIUS):
                    continue
                row = site.as_dict()
                row.update(geometry_guaranteed=row["guaranteed"], guaranteed=False,
                           status="candidate", source="observed edges", approach_verified=False)
                sites.append(row)
            layer.sites = sites[:self.config.max_sites]
            layer.rectangles = rectangles
            layer.edge_key = edge_key
            layer.stale = False
            layer.updated_at = now
            layer.next_check_at = now + self.config.refresh_interval_seconds
            layer.status = "ready" if rocks else "waiting for matching rock faces"

    def snapshot(self, group_id):
        layer = self.layers.get(group_id)
        if layer is None:
            return None
        return copy.deepcopy(dict(
            status=layer.status, stale=layer.stale, updated_at=layer.updated_at,
            next_check_at=layer.next_check_at, scan_count=layer.scan_count,
            frame_revision=layer.frame[0], safety=self.config.safety,
            observed_rectangles=layer.rectangles, sites=layer.sites))
