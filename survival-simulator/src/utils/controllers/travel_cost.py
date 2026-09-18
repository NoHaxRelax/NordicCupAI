"""Integrate travel effort over the observation-derived biome raster.

Movement is charged before biome slowing. A physical distance in river,
for example, therefore needs distance / .3 requested walking units. This
helper reads only the planner's public-observation estimate, never the world.
"""

import math

import numpy as np


MOVEMENT_FACTORS = dict(forest=1., grassland=1., swamp=.5, desert=.8, river=.3)


class PublicBiomeTravel:
    """Cache a biome layer, then price arbitrary remaining route polylines."""

    def __init__(self, movement_factors=None):
        self.movement_factors = dict(MOVEMENT_FACTORS if movement_factors is None else movement_factors)
        if any(not math.isfinite(value) or value <= 0 for value in self.movement_factors.values()):
            raise ValueError("Movement factors must be positive and finite")
        self._layer = None
        self._key = None
        self.bounds = self.pitch = self.inverse = self.confidence = None
        self.x_lines = self.y_lines = np.empty(0)

    def update(self, layer):
        """Prepare a newly fitted layer once; missing/malformed layers fall back.

        The biome estimator replaces layers on refits and supplies a fingerprint
        and timestamp. Ordinary calls reuse its unchanged cached arrays.
        """
        key = None if layer is None else (id(layer), layer.get("fingerprint"), layer.get("updated_at"),
                                         id(layer.get("labels")), id(layer.get("confidence")))
        if key == self._key:
            return
        self._key, self._layer = key, layer
        self.bounds = self.pitch = self.inverse = self.confidence = None
        self.x_lines = self.y_lines = np.empty(0)
        if layer is None:
            return
        try:
            bounds = np.asarray(layer["bounds"], dtype=float)
            labels = np.asarray(layer["labels"], dtype=int)
            confidence = np.asarray(layer["confidence"], dtype=float)
            palette = layer["palette"]
            if (bounds.shape != (4,) or not np.all(np.isfinite(bounds))
                    or np.any(bounds[2:] <= bounds[:2]) or labels.ndim != 2 or not labels.size
                    or confidence.shape != labels.shape):
                return
            inverse = np.zeros(labels.shape, dtype=float)
            known = np.zeros(labels.shape, dtype=bool)
            for index, name in enumerate(palette):
                factor = self.movement_factors.get(name)
                if factor is not None:
                    mask = labels == index
                    inverse[mask], known[mask] = 1. / factor, True
            confidence = np.where(known & np.isfinite(confidence), np.clip(confidence, 0., 1.), 0.)
            height, width = labels.shape
            pitch = (bounds[2:] - bounds[:2]) / (width, height)
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        self.bounds, self.pitch = bounds, pitch
        self.inverse, self.confidence = inverse, confidence
        self.x_lines = np.linspace(bounds[0], bounds[2], width + 1)
        self.y_lines = np.linspace(bounds[1], bounds[3], height + 1)

    def requested_distance(self, points, fallback_factor):
        """Return requested walking units for the exact supplied route length.

        Split segments at raster boundaries and integrate the confidence-weighted
        inverse speed factor in each crossed cell. Unknown cells and portions
        outside the estimate use the agent's currently observed biome factor.
        Nonfinite routes return infinity so callers reject them as unaffordable.
        """
        fallback_factor = float(fallback_factor)
        if not math.isfinite(fallback_factor) or fallback_factor <= 0:
            raise ValueError("fallback_factor must be positive and finite")
        points = np.asarray(points, dtype=float)
        if not points.size:
            return 0.
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must contain two-dimensional coordinates")
        if not np.all(np.isfinite(points)):
            return math.inf
        fallback_inverse = 1. / fallback_factor
        total = 0.
        for start, end in zip(points, points[1:]):
            delta = end - start
            length = math.hypot(*delta)
            if not math.isfinite(length):
                return math.inf
            if length == 0:
                continue
            if self.bounds is None:
                total += length * fallback_inverse
                continue
            cuts = [np.array([0., 1.])]
            for axis, lines in enumerate((self.x_lines, self.y_lines)):
                if delta[axis] != 0:
                    crossings = (lines - start[axis]) / delta[axis]
                    cuts.append(crossings[(crossings > 0.) & (crossings < 1.)])
            fractions = np.unique(np.concatenate(cuts))
            midpoints = start + ((fractions[:-1] + fractions[1:]) / 2.)[:, None] * delta
            inside = np.all((midpoints >= self.bounds[:2]) & (midpoints < self.bounds[2:]), axis=1)
            inverse = np.full(len(midpoints), fallback_inverse)
            if np.any(inside):
                cells = np.floor((midpoints[inside] - self.bounds[:2]) / self.pitch).astype(int)
                x, y = cells[:, 0], cells[:, 1]
                confidence = self.confidence[y, x]
                inverse[inside] = confidence * self.inverse[y, x] + (1. - confidence) * fallback_inverse
            total += length * float(np.diff(fractions) @ inverse)
        return total
