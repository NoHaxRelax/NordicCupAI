"""Find tree-sized unsurveyed patches, excluding known walls and tiny slivers."""

import math

import numpy as np
from scipy.ndimage import binary_propagation, distance_transform_edt, label
from scipy.spatial import cKDTree


def sensed_points(points, view):
    _, origin, heading, hearing, vision, cone, edges, _ = view
    offset = points - origin
    c, s = math.cos(heading), math.sin(heading)
    local = np.column_stack((offset[:, 0] * c + offset[:, 1] * s,
                             -offset[:, 0] * s + offset[:, 1] * c))
    distance = np.linalg.norm(local, axis=1)
    heard = distance < max(0., hearing - 4.)
    visible = (distance < vision - 4.) & (np.abs(np.arctan2(local[:, 1], local[:, 0])) < cone / 2 - .03)
    candidates = np.flatnonzero(visible & ~heard)
    rays = local[candidates]
    for start, end in edges:
        dx, dy = end[0] - start[0], end[1] - start[1]
        cross = rays[:, 0] * dy - rays[:, 1] * dx
        valid = np.abs(cross) > 1e-9
        t = np.divide(start[0] * dy - start[1] * dx, cross, out=np.full(len(rays), np.inf), where=valid)
        u = np.divide(start[0] * rays[:, 1] - start[1] * rays[:, 0], cross,
                      out=np.full(len(rays), np.inf), where=valid)
        visible[candidates[(t > 0) & (t < 1) & (u >= 0) & (u <= 1)]] = False
    return heard | visible


class SurveyGaps:
    def __init__(self, size, cell_size=10., maximum=24000, minimum_width=25.):
        self.origin = np.array([40., 40.])
        self.origin[:] = min(40., min(size) / 4)
        extent = np.asarray(size) - 2 * self.origin
        self.spacing = float(cell_size)
        while np.prod(np.ceil(extent / self.spacing)) > maximum:
            self.spacing *= 1.2
        self.nx, self.ny = np.maximum(1, np.ceil(extent / self.spacing).astype(int))
        self.pitch = extent / (self.nx, self.ny)
        xx, yy = np.meshgrid(self.origin[0] + (np.arange(self.nx) + .5) * self.pitch[0],
                             self.origin[1] + (np.arange(self.ny) + .5) * self.pitch[1])
        self.points = np.column_stack((xx.ravel(), yy.ravel()))
        self.seen = np.zeros((self.ny, self.nx), dtype=bool)
        self.minimum_width = minimum_width
        self.patches = []
        self.reachable = np.ones_like(self.seen)

    def observe(self, views):
        for view in views.values():
            # Only evaluate the local sensing disk, not every world pixel.
            nearby = np.flatnonzero(np.sum((self.points - view[1]) ** 2, axis=1) <= max(view[3], view[4]) ** 2)
            self.seen.ravel()[nearby[sensed_points(self.points[nearby], view)]] = True

    def find(self, group, poses):
        walls = np.zeros_like(self.seen)
        for edge in group.edges:
            start, end = np.asarray(edge.start), np.asarray(edge.end)
            steps = max(2, math.ceil(np.linalg.norm(end - start) / min(self.pitch) * 2))
            points = np.linspace(start, end, steps)
            indices = np.floor((points - self.origin) / self.pitch).astype(int)
            valid = (indices[:, 0] >= 0) & (indices[:, 0] < self.nx) & (indices[:, 1] >= 0) & (indices[:, 1] < self.ny)
            walls[indices[valid, 1], indices[valid, 0]] = True
        seeds = np.zeros_like(walls)
        for pose in poses:
            x, y = np.floor((pose.position - self.origin) / self.pitch).astype(int)
            if 0 <= x < self.nx and 0 <= y < self.ny and not walls[y, x]:
                seeds[y, x] = True
        reachable = binary_propagation(seeds, mask=~walls)
        self.reachable = reachable
        unknown = ~self.seen & reachable
        # Padding makes world edges real boundaries of a candidate patch.
        clearance = distance_transform_edt(np.pad(unknown, 1), sampling=self.pitch[::-1])[1:-1, 1:-1]
        clearance -= max(self.pitch) / 2  # A lone raster sample is not a tree-sized patch.
        components, _ = label(unknown)
        areas = np.bincount(components.ravel()) * float(np.prod(self.pitch))
        valid = unknown & (clearance >= self.minimum_width / 2)
        indices = np.flatnonzero(valid)
        self.patches = []
        if not len(indices) or not poses:
            return self.patches
        distance, _ = cKDTree([p.position for p in poses]).query(self.points[indices])
        score = np.minimum(clearance.ravel()[indices], self.minimum_width) - .1 * distance
        chosen = set()
        for index in np.argsort(-score, kind="stable"):
            flat = indices[index]
            component = int(components.ravel()[flat])
            if component in chosen or areas[component] < self.minimum_width ** 2:
                continue
            chosen.add(component)
            self.patches.append(dict(position=self.points[flat].tolist(), area=float(areas[component]),
                                     clearance=float(clearance.ravel()[flat])))
            if len(self.patches) >= 32:
                break
        return self.patches
