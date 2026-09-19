"""Inverse Voronoi estimates from labelled, estimated agent positions only.

The map format supplies the prior (at most ten land generators), not their
locations or labels. Several generators may share a label. Rivers are a later
overlay, so they are excluded from the land fit and interpolated locally.
"""

import copy
import hashlib
import math
from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay, QhullError
from scipy.special import expit


BIOMES = ("forest", "grassland", "swamp", "desert", "river")


class BiomeInferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    max_sites: int = Field(default=10, ge=4, le=32)
    refit_interval_seconds: float = Field(default=2.0, gt=0)
    warmup_seconds: float = Field(default=30.0, ge=0)
    refit_interval_increment_seconds: float = Field(default=2.0, gt=0)
    max_refit_interval_seconds: float = Field(default=30.0, gt=0)
    fit_compute_budget_fraction: float = Field(default=0.10, gt=0, le=1)
    # Disable for reproducible benchmarks: CPU contention must not alter actions.
    adapt_to_wall_clock: bool = Field(default=True, strict=True)
    minimum_validation_agreement: float = Field(default=0.90, ge=0, le=1)
    minimum_fit_agreement: float = Field(default=0.95, ge=0, le=1)
    stable_grid_change_fraction: float = Field(default=0.05, ge=0, le=1)
    validation_min_samples: int = Field(default=12, ge=1)
    min_samples: int = Field(default=12, ge=3)
    max_samples: int = Field(default=768, ge=12)
    max_sample_uncertainty: float = Field(default=8.0, gt=0)
    max_iterations: int = Field(default=80, ge=1, le=500)
    max_site_additions: int = Field(default=2, ge=1, le=10)
    misclassification_tolerance: float = Field(default=0.02, ge=0, lt=1)
    minimum_site_separation: float = Field(default=30.0, gt=0)
    grid_cell_size: float = Field(default=24.0, gt=0)
    max_grid_cells: int = Field(default=6000, ge=4)
    support_distance: float = Field(default=240.0, gt=0)
    river_support_distance: float = Field(default=100.0, gt=0)

    @model_validator(mode="after")
    def valid_schedule(self):
        if self.max_refit_interval_seconds < self.refit_interval_seconds:
            raise ValueError("max_refit_interval_seconds must be at least refit_interval_seconds")
        return self


def _distances(points, sites):
    return np.sum((points[:, None, :] - sites[None, :, :]) ** 2, axis=2)


def _initial_sites(points, labels, maximum):
    """Seed disconnected same-label patches separately before optimizing.

    Local Delaunay neighbours supply topology, not final borders. Long hull
    edges must not join two forests across an intervening desert or data gap.
    Collinear/duplicate samples safely fall back to one center per label.
    """
    parent = np.arange(len(points))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    try:
        triangles = Delaunay(points).simplices
        spacing = cKDTree(points).query(points, k=2)[0][:, 1]
        positive = spacing[spacing > 1e-8]
        limit = 3 * np.median(positive) if len(positive) else 0
        for triangle in triangles:
            for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                if labels[first] == labels[second] and np.linalg.norm(points[first] - points[second]) <= limit:
                    parent[root(first)] = root(second)
    except QhullError:
        return ([points[labels == label].mean(axis=0) for label in sorted(set(labels))], sorted(set(labels)))
    centers, classes, extras = [], [], []
    roots = np.array([root(index) for index in range(len(points))])
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        components = [members[roots[members] == key] for key in sorted(set(roots[members]))]
        components.sort(key=lambda indices: (-len(indices), int(indices[0])))
        centers.append(points[components[0]].mean(axis=0))
        classes.append(label)
        for component in components[1:]:
            if len(component) >= max(3, .02 * len(members)):
                extras.append((len(component), label, int(component[0]), points[component].mean(axis=0)))
    for _, label, _, center in sorted(extras, key=lambda item: (-item[0], item[1], item[2]))[:maximum - len(centers)]:
        centers.append(center)
        classes.append(label)
    return centers, classes


def _optimize(points, labels, weights, sites, site_labels, upper, iterations):
    """Fit ordinary Euclidean Voronoi sites, using label separation constraints.

    Each observation should be nearer a generator of its own label than any
    other label. A smooth classification loss penalizes coincident differently
    labelled sites, which can fool a squared distance hinge by shrinking all
    margins to zero. Weak regularization stabilizes equivalent site sets.
    """
    if len(set(labels)) < 2:
        return sites
    same = labels[:, None] == site_labels[None, :]
    anchors = sites.copy()
    rows = np.arange(len(points))

    def objective(flat):
        centers = flat.reshape((-1, 2))
        distances = _distances(points, centers)
        own = np.argmin(np.where(same, distances, np.inf), axis=1)
        other = np.argmin(np.where(~same, distances, np.inf), axis=1)
        temperature = .005
        difference = (distances[rows, own] - distances[rows, other]) / temperature
        gradient = np.zeros_like(centers)
        factors = 2 * weights * expit(difference) / temperature
        np.add.at(gradient, own, factors[:, None] * (centers[own] - points))
        np.add.at(gradient, other, -factors[:, None] * (centers[other] - points))
        regularization = 1e-5
        gradient += 2 * regularization * (centers - anchors)
        loss = np.sum(weights * np.logaddexp(0, difference)) + regularization * np.sum((centers - anchors) ** 2)
        return float(loss), gradient.ravel()

    result = minimize(objective, sites.ravel(), jac=True, method="L-BFGS-B",
                      bounds=[(0, float(upper[axis])) for _ in sites for axis in (0, 1)],
                      options={"maxiter": iterations, "ftol": 1e-14, "gtol": 1e-9})
    return result.x.reshape((-1, 2)) if np.isfinite(result.x).all() else sites


def fit_sites(points, labels, uncertainty, bounds, config, previous=None):
    scale = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1)
    origin = np.array(bounds[:2])
    normalized = (points - origin) / scale
    upper = (np.array(bounds[2:]) - origin) / scale
    present = sorted(set(labels.tolist()))
    if previous is None:
        centers, classes = _initial_sites(normalized, labels, config.max_sites)
    else:
        centers = [((np.array(site["position"]) - origin) / scale).tolist()
                   for site in previous if site["biome"] in present]
        classes = [site["biome"] for site in previous if site["biome"] in present]
        if len(centers) + len(set(present) - set(classes)) > config.max_sites:
            # A newly discovered label must have a site even if an earlier,
            # incomplete fit already used the generator budget.
            centers, classes = _initial_sites(normalized, labels, config.max_sites)
    for label in present:
        if label not in classes:
            centers.append(normalized[labels == label].mean(axis=0).tolist())
            classes.append(label)
    centers = np.clip(np.array(centers), 0, upper)
    classes = np.array(classes)
    weights = 1 / (1 + uncertainty / 4) ** 2
    weights /= np.array([math.sqrt(np.sum(labels == label)) for label in labels])
    weights /= weights.sum()
    for addition in range(config.max_site_additions + 1):
        centers = _optimize(normalized, labels, weights, centers, classes, upper, config.max_iterations)
        distances = _distances(normalized, centers)
        predicted = classes[np.argmin(distances, axis=1)]
        wrong = predicted != labels
        if (wrong.mean() <= config.misclassification_tolerance or len(centers) >= config.max_sites
                or addition == config.max_site_additions):
            break
        own_distance = np.min(np.where(labels[:, None] == classes, distances, np.inf), axis=1)
        candidates = wrong & (own_distance > (config.minimum_site_separation / scale) ** 2)
        if not candidates.any():
            break
        # A second region of the same biome needs a second generator, rather
        # than bending a single convex cell around an intervening biome.
        error = own_distance - distances.min(axis=1)
        index = int(np.argmax(np.where(candidates, error * weights, -np.inf)))
        centers = np.vstack((centers, normalized[index]))
        classes = np.append(classes, labels[index])
    return centers * scale + origin, classes, float(np.mean(predicted == labels))


def voronoi_borders(sites, labels, bounds):
    """Clip pairwise bisectors by every other site's nearest-site half-plane."""
    segments = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            normal = sites[j] - sites[i]
            length = np.linalg.norm(normal)
            if labels[i] == labels[j] or length < 1e-8:
                continue
            midpoint = (sites[i] + sites[j]) / 2
            tangent = np.array([-normal[1], normal[0]]) / length
            constraints = [(np.array([-1., 0.]), -bounds[0]), (np.array([0., -1.]), -bounds[1]),
                           (np.array([1., 0.]), bounds[2]), (np.array([0., 1.]), bounds[3])]
            constraints += [(2 * (site - sites[i]), float(site @ site - sites[i] @ sites[i]))
                            for k, site in enumerate(sites) if k not in (i, j)]
            low, high = -np.inf, np.inf
            for axis, limit in constraints:
                slope = float(axis @ tangent)
                room = float(limit - axis @ midpoint)
                if abs(slope) < 1e-9:
                    if room < -1e-6:
                        high = -np.inf
                        break
                elif slope > 0:
                    high = min(high, room / slope)
                else:
                    low = max(low, room / slope)
            if high - low > 1e-6 and np.isfinite([low, high]).all():
                segments.append((midpoint + low * tangent, midpoint + high * tangent, labels[i], labels[j]))
    return segments


def _build_layer(samples, bounds, config, previous, now):
    points = np.array([sample.position for sample in samples])
    labels = np.array([sample.biome for sample in samples])
    uncertainty = np.array([sample.uncertainty for sample in samples])
    land = labels != "river"
    sites, classes, agreement = (np.empty((0, 2)), np.array([], dtype=str), None)
    if land.any():
        sites, classes, agreement = fit_sites(points[land], labels[land], uncertainty[land], bounds, config, previous)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    cell_size = config.grid_cell_size
    while True:
        nx, ny = max(1, math.ceil(width / cell_size)), max(1, math.ceil(height / cell_size))
        if nx * ny <= config.max_grid_cells:
            break
        cell_size *= 1.2
    xs = bounds[0] + (np.arange(nx) + .5) * width / nx
    ys = bounds[1] + (np.arange(ny) + .5) * height / ny
    xx, yy = np.meshgrid(xs, ys)
    queries = np.column_stack((xx.ravel(), yy.ravel()))
    grid = np.full(len(queries), -1, dtype=int)
    confidence = np.zeros(len(queries))
    land_distance = np.full(len(queries), np.inf)
    trees = {label: cKDTree(points[labels == label]) for label in sorted(set(labels))}
    if len(sites):
        inferred = classes[np.argmin(_distances(queries, sites), axis=1)]
        land_distance, nearest = cKDTree(points[land]).query(queries)
        supported = land_distance <= config.support_distance
        grid[supported] = [BIOMES.index(label) for label in inferred[supported]]
        confidence = np.maximum(0, 1 - land_distance / config.support_distance) * agreement
        confidence[inferred != labels[land][nearest]] *= .25
    if "river" in trees:
        distance, _ = trees["river"].query(queries)
        river = (distance < land_distance) & (distance <= config.river_support_distance)
        grid[river] = BIOMES.index("river")
        confidence[river] = np.maximum(0, 1 - distance[river] / config.river_support_distance)
    borders = []
    for start, end, first, second in voronoi_borders(sites, classes, bounds):
        # Segment long bisectors so support fades locally along the border.
        count = max(1, math.ceil(np.linalg.norm(end - start) / (2 * config.grid_cell_size)))
        for index in range(count):
            a, b = start + (end - start) * index / count, start + (end - start) * (index + 1) / count
            midpoint = (a + b) / 2
            distance = max(trees[first].query(midpoint)[0], trees[second].query(midpoint)[0])
            support = max(0., 1 - distance / config.support_distance)
            if support > .05:
                borders.append(dict(start=a.tolist(), end=b.tolist(), biomes=[str(first), str(second)],
                                    kind="land", confidence=round(support, 3)))
    grid, confidence = grid.reshape(ny, nx), confidence.reshape(ny, nx)
    # Rivers are curved overlays: show local raster bank estimates, never a
    # fictitious convex river cell in the land Voronoi model.
    for axis in (0, 1):
        first, second = (grid[:-1], grid[1:]) if axis == 0 else (grid[:, :-1], grid[:, 1:])
        river_index = BIOMES.index("river")
        crossings = (first >= 0) & (second >= 0) & ((first == river_index) != (second == river_index))
        for y, x in np.argwhere(crossings):
            x0, y0 = bounds[0] + x * width / nx, bounds[1] + y * height / ny
            a, b = (([x0, y0 + height / ny], [x0 + width / nx, y0 + height / ny]) if axis == 0 else
                    ([x0 + width / nx, y0], [x0 + width / nx, y0 + height / ny]))
            adjacent = confidence[y + (axis == 0), x + (axis == 1)]
            borders.append(dict(start=a, end=b, kind="river", confidence=round(float(min(confidence[y, x], adjacent)), 3)))
    generators = [dict(position=site.tolist(), biome=str(label)) for site, label in zip(sites, classes)]
    fingerprint = hashlib.sha256(grid.tobytes() + confidence.tobytes() + np.array(bounds).tobytes()
                                 + sites.tobytes()).hexdigest()[:20]
    return dict(method="fitted_voronoi_with_local_river_overlay", updated_at=now, bounds=list(bounds),
                sample_count=len(samples), land_sample_agreement=agreement, sites=generators, borders=borders,
                palette=list(BIOMES), labels=grid.tolist(), confidence=np.round(confidence, 3).tolist(),
                fingerprint=fingerprint, confidence_is_heuristic=True)


class BiomeEstimator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.layers = {}
        self.frames = {}
        self.next_fit = 0.
        self.attempt_frame = None
        self.last_time = None
        self.aligned_at = None
        self.warmup_until = None
        self.interval = self.config.refit_interval_seconds
        self.next_validation = 0.
        self.reference_samples = {}
        self.last_fit_seconds = None
        self.average_fit_seconds = None
        self.fit_count = 0
        self.validation_agreement = None
        self.validation_sample_count = 0
        self.reason = "waiting for shared coordinates"

    def _begin_schedule(self, frame, now):
        self.attempt_frame = frame
        self.frames.pop(frame[0], None)
        self.aligned_at = now
        self.warmup_until = now + self.config.warmup_seconds
        self.interval = self.config.refit_interval_seconds
        self.next_fit = self.next_validation = now
        self.reference_samples = {}
        self.validation_agreement = None
        self.validation_sample_count = 0
        self.reason = "new absolute frame"

    def _compute_interval(self):
        # Bound average fitting work per simulated second. Round to multiples
        # of the initial cadence so small timing fluctuations do not jitter it.
        base = self.config.refit_interval_seconds
        if not self.config.adapt_to_wall_clock:
            return base
        cost = self.average_fit_seconds or 0.
        multiples = max(1, math.ceil(cost / (base * self.config.fit_compute_budget_fraction)))
        return min(self.config.max_refit_interval_seconds, base * multiples)

    def _restart_learning(self, now, reason, layer):
        self.interval = self._compute_interval()
        self.warmup_until = now + self.config.warmup_seconds
        earliest = layer["updated_at"] + self.interval
        self.next_fit = min(self.next_fit, max(now, earliest))
        self.reason = reason

    @staticmethod
    def _signature(sample):
        return (round(float(sample.position[0]), 2), round(float(sample.position[1]), 2), sample.biome)

    def _check_evidence(self, layer, samples, signatures):
        fresh = [sample for key, sample in samples.items() if signatures[key] != self.reference_samples.get(key)]
        self.validation_sample_count = len(fresh)
        self.validation_agreement = None
        if not fresh:
            return None
        known = {signature[2] for signature in self.reference_samples.values()}
        if any(sample.biome not in known for sample in fresh):
            return "new biome observed"
        # Validate the published estimate on evidence it has not yet fitted,
        # including the river overlay and regions it previously left unknown.
        grid = np.array(layer["labels"], dtype=int)
        ny, nx = grid.shape
        x0, y0, x1, y1 = layer["bounds"]
        points = np.array([sample.position for sample in fresh])
        indices = np.floor((points - (x0, y0)) / (x1 - x0, y1 - y0) * (nx, ny)).astype(int)
        inside = (indices[:, 0] >= 0) & (indices[:, 0] < nx) & (indices[:, 1] >= 0) & (indices[:, 1] < ny)
        prediction = np.full(len(fresh), -1, dtype=int)
        prediction[inside] = grid[indices[inside, 1], indices[inside, 0]]
        observed = np.array([layer["palette"].index(sample.biome) for sample in fresh])
        self.validation_agreement = float(np.mean(prediction == observed))
        if (len(fresh) >= self.config.validation_min_samples
                and self.validation_agreement < self.config.minimum_validation_agreement):
            return "new observations disagree"
        return None

    def _schedule_next(self, now, stable, reason):
        if not stable:
            self.warmup_until = now + self.config.warmup_seconds
        if stable and now >= self.warmup_until - 1e-9:
            self.interval = min(self.config.max_refit_interval_seconds,
                                max(self.interval + self.config.refit_interval_increment_seconds, self._compute_interval()))
        else:
            self.interval = self._compute_interval()
        self.next_fit = now + self.interval
        self.reason = reason

    def update(self, groups, poses, now):
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        living = {pose.group_id for pose in poses.values()}
        self.frames = {key: frame for key, frame in self.frames.items() if key in living}
        self.layers = {key: value for key, value in self.layers.items() if key in living
                       and key in groups and groups[key].anchored
                       and self.frames.get(key) == groups[key].frame_revision}
        if not self.config.enabled:
            self.layers.clear()
            return
        # Keep a prior estimate across a newborn's brief disconnection, but
        # only fit new data when the living population shares an absolute frame.
        if len(living) != 1:
            return
        key = next(iter(living))
        group = groups[key]
        frame = (key, group.frame_revision)
        if not group.anchored:
            return
        if frame != self.attempt_frame or (key not in self.layers and key in self.frames):
            self._begin_schedule(frame, now)
        if now + 1e-9 < min(self.next_fit, self.next_validation):
            return
        self.next_validation = now + self.config.refit_interval_seconds
        samples_by_cell = {cell: sample for cell, sample in sorted(group.biomes.items())
                   if sample.biome in BIOMES and np.isfinite(sample.position).all()
                   and 0 <= sample.uncertainty <= self.config.max_sample_uncertainty
                   and min(sample.position) >= 0}
        width = group.world_size[0] if group.world_size else group.known_width
        height = group.world_size[1] if group.world_size else group.known_height
        samples_by_cell = {cell: sample for cell, sample in samples_by_cell.items()
                           if (width is None or sample.position[0] < width)
                           and (height is None or sample.position[1] < height)}
        signatures = {cell: self._signature(sample) for cell, sample in samples_by_cell.items()}
        samples = list(samples_by_cell.values())
        previous_layer = self.layers.get(key)
        known_bounds = (width, height)
        if previous_layer is not None:
            problem = self._check_evidence(previous_layer, samples_by_cell, signatures)
            if previous_layer["known_bounds"] != known_bounds:
                problem = "new world bounds"
            if problem is not None:
                self._restart_learning(now, problem, previous_layer)
        if now + 1e-9 < self.next_fit:
            return
        if len(samples) < self.config.min_samples:
            self.next_fit = now + self._compute_interval()
            self.reason = "insufficient reliable samples"
            return
        if (previous_layer is not None and signatures == self.reference_samples
                and (previous_layer["land_sample_agreement"] is None
                     or previous_layer["land_sample_agreement"] >= self.config.minimum_fit_agreement)
                and previous_layer["known_bounds"] == known_bounds):
            # Identical evidence needs a cheap check, not another optimization.
            # This saves work without treating repeated samples as validation.
            self._schedule_next(now, True, "unchanged evidence; fit reused")
            return
        if len(samples) > self.config.max_samples:
            # Deterministic stratification retains rare biome labels.
            selected = []
            present = sorted({sample.biome for sample in samples})
            remaining = self.config.max_samples - len(present)
            for label in present:
                subset = [sample for sample in samples if sample.biome == label]
                quota = 1 + int(remaining * (len(subset) - 1) / (len(samples) - len(present)))
                selected.extend(subset[index] for index in np.linspace(0, len(subset) - 1, min(quota, len(subset)), dtype=int))
            samples = selected[:self.config.max_samples]
        bounds = (0., 0., float(width or max(s.position[0] for s in samples) + self.config.support_distance),
                  float(height or max(s.position[1] for s in samples) + self.config.support_distance))
        previous = previous_layer["sites"] if previous_layer is not None else None
        started = perf_counter()
        layer = _build_layer(samples, bounds, self.config, previous, now)
        self.last_fit_seconds = max(0., perf_counter() - started)
        self.average_fit_seconds = (self.last_fit_seconds if self.average_fit_seconds is None else
                                    .75 * self.average_fit_seconds + .25 * self.last_fit_seconds)
        self.fit_count += 1
        agreement = layer["land_sample_agreement"]
        stable = agreement is None or agreement >= self.config.minimum_fit_agreement
        reason = "stable estimate" if stable else "fit still uncertain"
        if previous_layer is not None:
            same_grid = previous_layer["bounds"] == layer["bounds"] and np.shape(previous_layer["labels"]) == np.shape(layer["labels"])
            change = float(np.mean(np.array(previous_layer["labels"]) != np.array(layer["labels"]))) if same_grid else 1.
            if change > self.config.stable_grid_change_fraction:
                stable, reason = False, "estimate changed substantially"
            if (self.validation_sample_count >= self.config.validation_min_samples
                    and self.validation_agreement is not None
                    and self.validation_agreement < self.config.minimum_validation_agreement):
                stable, reason = False, "new observations disagree"
        self._schedule_next(now, stable, reason)
        self.reference_samples = signatures
        layer["bounds_complete"] = width is not None and height is not None
        layer["known_bounds"] = known_bounds
        self.layers[key] = layer
        self.frames[key] = group.frame_revision

    def snapshot(self, group_id):
        layer = copy.deepcopy(self.layers.get(group_id))
        if layer is not None:
            layer["refit_schedule"] = dict(
                alignment_started_at=self.aligned_at, warmup_until=self.warmup_until,
                interval_seconds=self.interval, next_check_at=self.next_fit,
                fit_count=self.fit_count, last_fit_ms=1000 * self.last_fit_seconds,
                average_fit_ms=1000 * self.average_fit_seconds,
                validation_agreement=self.validation_agreement,
                validation_sample_count=self.validation_sample_count, reason=self.reason)
        return layer
