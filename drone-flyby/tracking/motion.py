"""Calibrate and evaluate constant-translation perspective motion.

Runtime module: no dataset paths, labels, random box errors or future frames.
Coordinates are source pixels; time is measured in caller-supplied motion ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


class CalibrationError(ValueError):
    """The observations do not support a usable shared motion calibration."""


class ProjectionError(ValueError):
    """Requested geometry reaches or passes a projective singularity."""


def finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def box_array(box):
    box = finite(box, "box")
    if box.shape != (4,) or np.any(box[2:] <= box[:2]):
        raise ValueError("Expected positive-area [x1, y1, x2, y2] box")
    return box


def corners(box):
    x1, y1, x2, y2 = box_array(box)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def project(points, matrix):
    points = finite(points, "points")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected N by 2 points")
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.all(np.isfinite(homogeneous)) or np.any(homogeneous[:, 2] <= 1e-9):
        raise ProjectionError("Point reaches or crosses the camera plane")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


@dataclass(frozen=True)
class ViewGeometry:
    """Actual delivered crop, not the crop requested for a future frame.

    Boxes use pixel edges. Feature points use OpenCV pixel centres, including
    the half-pixel offset when an image has been resized.
    """
    source_size: tuple[int, int]
    region: tuple[float, float, float, float]
    image_size: tuple[int, int]

    def __post_init__(self):
        for name in ("source_size", "image_size"):
            values = finite(getattr(self, name), name)
            if values.shape != (2,) or np.any(values <= 0) or np.any(values != np.floor(values)):
                raise ValueError(f"{name} must contain two positive integers")
            object.__setattr__(self, name, tuple(int(v) for v in values))
        region = box_array(self.region)
        if np.any(region[:2] < 0) or np.any(region[2:] > self.source_size):
            raise ValueError("Crop must lie inside the source image")
        object.__setattr__(self, "region", tuple(float(v) for v in region))

    @classmethod
    def from_request(cls, request: Mapping):
        view = request["view"]
        return cls((request["original_width"], request["original_height"]),
                   tuple(view["source_region_xyxy"]), (view["width"], view["height"]))

    @property
    def scale(self):
        return (np.array(self.region[2:]) - self.region[:2]) / self.image_size

    def points_to_source(self, points):
        return (finite(points, "points") + .5) * self.scale - .5 + self.region[:2]

    def box_to_source(self, box, *, normalized=False):
        box = box_array(box)
        if normalized:
            box = box * np.tile(self.image_size, 2)
        if np.any(box[:2] < 0) or np.any(box[2:] > self.image_size):
            raise ValueError("Detection box must lie inside the delivered view")
        return box * np.tile(self.scale, 2) + np.tile(self.region[:2], 2)

    def box_from_source(self, box, *, normalized=False):
        result = (box_array(box) - np.tile(self.region[:2], 2)) / np.tile(self.scale, 2)
        return result / np.tile(self.image_size, 2) if normalized else result


@dataclass(frozen=True)
class MotionModel:
    """H(a,b) = (I + (b-origin) A) inverse(I + (a-origin) A)."""
    matrix: np.ndarray
    source_size: tuple[int, int]
    origin_tick: float
    calibrated_until: float
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        matrix = finite(self.matrix, "motion matrix").copy()
        if matrix.shape != (3, 3) or np.linalg.matrix_rank(matrix, tol=1e-8) > 1:
            raise ValueError("Motion matrix must be 3 by 3 and rank at most one")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        size = ViewGeometry(tuple(self.source_size), (0, 0, *self.source_size), tuple(self.source_size)).source_size
        object.__setattr__(self, "source_size", size)
        origin, end = map(float, finite([self.origin_tick, self.calibrated_until], "ticks"))
        if end < origin:
            raise ValueError("Calibration cannot finish before it starts")
        object.__setattr__(self, "origin_tick", origin)
        object.__setattr__(self, "calibrated_until", end)
        self._at(end)

    def _at(self, tick):
        tick = float(finite(tick, "tick"))
        if tick < self.origin_tick:
            raise ValueError("Tick predates this calibration's time origin")
        elapsed = tick - self.origin_tick
        if 1 + elapsed * np.trace(self.matrix) <= 1e-9:
            raise ProjectionError("Requested time reaches or crosses the scene plane")
        return np.eye(3) + elapsed * self.matrix

    def mapping(self, start_tick, end_tick):
        start, end = self._at(start_tick), self._at(end_tick)
        return np.linalg.solve(start.T, end.T).T

    def points(self, points, start_tick, end_tick):
        return project(points, self.mapping(start_tick, end_tick))

    def box(self, box, start_tick, end_tick):
        points = self.points(corners(box), start_tick, end_tick)
        return np.r_[points.min(axis=0), points.max(axis=0)]

    def to_dict(self):
        return {"version": 1, "matrix": self.matrix.tolist(), "source_size": list(self.source_size),
                "origin_tick": self.origin_tick, "calibrated_until": self.calibrated_until,
                "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported motion-model version")
        return cls(data["matrix"], tuple(data["source_size"]), data["origin_tick"],
                   data["calibrated_until"], data.get("diagnostics", {}))

    @classmethod
    def from_matches(cls, first, second, *, source_size=(3840, 2160),
                     first_tick=0., second_tick=1., min_matches=12):
        """Fit from two sets of corresponding source-pixel feature positions.

        Geometry is frozen after fitting. Epipoles at infinity (parallel image
        flow, including nadir translation) are supported without dividing by z.
        """
        x, y = finite(first, "first points"), finite(second, "second points")
        if x.ndim != 2 or x.shape[1] != 2 or x.shape != y.shape or len(x) < max(3, min_matches):
            raise CalibrationError("Insufficient paired N by 2 feature coordinates")
        dt = float(finite(second_tick, "second_tick") - finite(first_tick, "first_tick"))
        if dt <= 0:
            raise CalibrationError("Calibration needs two distinct increasing motion ticks")
        source_size = ViewGeometry(tuple(source_size), (0, 0, *source_size), tuple(source_size)).source_size
        flow = y - x
        lengths = np.linalg.norm(flow, axis=1)
        moving = lengths > 1e-3
        if moving.sum() < max(3, min_matches):
            raise CalibrationError("Insufficient motion to calibrate; wait for a moving frame")
        x, y, flow, lengths = x[moving], y[moving], flow[moving], lengths[moving]
        scale = max(source_size) / 3.84
        centre = np.array(source_size) / 2
        xn, yn = (x-centre)/scale, (y-centre)/scale
        normals = np.c_[-flow[:, 1], flow[:, 0]] / lengths[:, None]
        rhs = (normals*x).sum(axis=1)
        if np.linalg.cond(normals) < 1e6:
            ep = np.linalg.lstsq(normals, rhs, rcond=None)[0]
            for _ in range(20):
                factor = lengths / np.maximum(100., np.linalg.norm(ep-y, axis=1))
                residual = (normals@ep-rhs)*factor
                weights = np.sqrt(np.minimum(1., .6/np.maximum(1e-5, np.abs(residual))))*factor
                ep = np.linalg.lstsq(normals*weights[:, None], rhs*weights, rcond=None)[0]
            e = np.r_[(ep-centre)/scale, 1.]
        else:
            # Homogeneous flow-ray intersection also represents points at infinity.
            lines = np.c_[normals, -(normals*xn).sum(axis=1)]
            _, singular, vh = np.linalg.svd(lines, full_matrices=False)
            if singular[1] < 1e-8:
                raise CalibrationError("Features do not constrain a common motion direction")
            e = vh[-1]
        radial = e[:2] - yn*e[2]
        radial_norm = np.linalg.norm(radial, axis=1)
        valid = radial_norm > 1e-8
        x, y, xn, yn, radial, radial_norm = [a[valid] for a in (x, y, xn, yn, radial, radial_norm)]
        if len(x) < max(3, min_matches):
            raise CalibrationError("Too few features away from the vanishing point")
        design = np.c_[xn, np.ones(len(xn))]
        if np.linalg.matrix_rank(design) < 3:
            raise CalibrationError("Calibration features must cover a two-dimensional area")
        amount = ((yn-xn)*radial).sum(axis=1)/(radial_norm**2*dt)
        q = np.linalg.lstsq(design, amount, rcond=None)[0]
        for _ in range(25):
            travel = dt*(design@q)
            predicted = (xn + travel[:, None]*e[:2])/(1+travel[:, None]*e[2])
            errors = np.linalg.norm(predicted-yn, axis=1)*scale
            weights = np.sqrt(np.minimum(1., 1.5/np.maximum(1e-5, errors)))*radial_norm
            q = np.linalg.lstsq(design*weights[:, None], amount*weights, rcond=None)[0]
        transform = np.array([[1/scale, 0, -centre[0]/scale],
                              [0, 1/scale, -centre[1]/scale], [0, 0, 1.]])
        matrix = np.linalg.solve(transform, np.outer(e, q)@transform)
        model = cls(matrix, source_size, first_tick, second_tick)
        errors = np.linalg.norm(model.points(x, first_tick, second_tick)-y, axis=1)
        median = float(np.median(errors))
        if not np.isfinite(median) or median > 6.:
            raise CalibrationError(f"Shared motion fit is poor: median residual {median:.2f} source pixels")
        source_e = np.linalg.solve(transform, e)
        return cls(matrix, source_size, first_tick, second_tick,
                   {"matches_used": len(x), "median_residual_source_px": median,
                    "p90_residual_source_px": float(np.percentile(errors, 90)),
                    "epipole_homogeneous": source_e.tolist(), "added_noise": False})


def calibrate_images(first_image, second_image, first_view: ViewGeometry,
                     second_view: ViewGeometry, *, first_tick=0., second_tick=1.):
    """SIFT calibration from two real images, including different zooms/crops.

    NumPy is sufficient for prediction. OpenCV is needed only for calibration.
    Images are uint8 gray or BGR arrays. Insufficient texture raises
    CalibrationError instead of silently substituting an unmeasured motion.
    """
    import cv2

    if first_view.source_size != second_view.source_size:
        raise ValueError("Calibration images must have the same source dimensions")
    points, descriptors = [], []
    sift = cv2.SIFT_create(nfeatures=6500)
    for image, view in ((first_image, first_view), (second_image, second_view)):
        if image is None or image.dtype != np.uint8 or image.shape[:2] != view.image_size[::-1]:
            raise ValueError("Expected uint8 image matching the supplied view dimensions")
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim != 2:
            raise ValueError("Expected grayscale or three-channel BGR image")
        ratio = min(1., 1920/image.shape[1], 1080/image.shape[0])
        if ratio < 1:
            image = cv2.resize(image, (round(image.shape[1]*ratio), round(image.shape[0]*ratio)),
                               interpolation=cv2.INTER_AREA)
        working_view = ViewGeometry(view.source_size, view.region, image.shape[1::-1])
        keypoints, desc = sift.detectAndCompute(image, None)
        if desc is None or len(keypoints) < 12:
            raise CalibrationError("Insufficient texture in a calibration image")
        points.append(working_view.points_to_source(np.array([k.pt for k in keypoints])))
        descriptors.append(desc)
    matcher = cv2.BFMatcher()
    forward = matcher.knnMatch(descriptors[0], descriptors[1], k=2)
    reverse = {pair[0].queryIdx: pair[0].trainIdx
               for pair in matcher.knnMatch(descriptors[1], descriptors[0], k=2)
               if len(pair) == 2 and pair[0].distance < .7*pair[1].distance}
    good = [pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < .7*pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    if len(good) < 12:
        raise CalibrationError("Too few mutual feature matches between the delivered crops")
    x = points[0][[m.queryIdx for m in good]]
    y = points[1][[m.trainIdx for m in good]]
    h, _ = cv2.findHomography(x, y, cv2.RANSAC, 6.)
    if h is None:
        raise CalibrationError("Unable to establish coherent image motion")
    # Loose gate removes mismatches while retaining measurable departures from a plane.
    homogeneous = np.c_[x, np.ones(len(x))] @ h.T
    projected = homogeneous[:, :2]/homogeneous[:, 2, None]
    keep = np.linalg.norm(projected-y, axis=1) < 40.
    model = MotionModel.from_matches(x[keep], y[keep], source_size=first_view.source_size,
                                     first_tick=first_tick, second_tick=second_tick)
    diagnostics = {**model.diagnostics, "mutual_matches": len(good),
                   "first_region": list(first_view.region), "second_region": list(second_view.region)}
    return MotionModel(model.matrix, model.source_size, model.origin_tick, model.calibrated_until, diagnostics)
