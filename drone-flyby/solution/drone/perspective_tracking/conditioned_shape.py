"""Experimental class/position/axis-conditioned, reflection-consistent profiles.

The motion calibration supplies the perspective geometry. This small residual
model supplies bounded edge-rate corrections. It is not a metric 3D model.
Unvalidated classes fall back to the shared tracker, even after fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .edge_profile import EdgeMotionProfile, edge_geometry
from .motion import MotionModel, ProjectionError, box_array, finite


def fit_residual_factors(model, observations, *, minimum_gap=2.):
    """Fit one local training window relative to the exact shared box warp.

    The first box anchors all pairs. Zero log factors produce zero correction,
    including the difference between edge midpoints and warped box corners.
    """
    anchor, box = observations[0]
    box = box_array(box)
    ep, rates = edge_geometry(model, box, anchor)
    times, targets = [], []
    for tick, later in observations[1:]:
        if tick-anchor < minimum_gap:
            continue
        elapsed = tick-anchor
        nominal = ep+(box-ep)/(1+elapsed*rates)
        baseline = model.box(box, anchor, tick)
        targets.append(box_array(later)-baseline+nominal-ep)
        times.append(elapsed)
    if len(times) < 3:
        raise ValueError('Need three future training views separated from the anchor')
    target, x = np.array(targets), np.array(times)[:, None]*rates
    offsets, factors = box-ep, []
    for edge in range(4):
        a, b, u = offsets[edge], target[:, edge], x[:, edge]
        if abs(a) < 1e-5 or np.any(a*b <= 0) or np.sum(u*u) < 1e-12:
            raise ValueError('Training does not constrain every residual edge')
        value = float(np.sum(u*(a/b-1))/np.sum(u*u))
        if value <= 0 or np.any(1+value*u <= 0):
            raise ValueError('Invalid residual factor fit')
        for _ in range(25):
            residual = a/(1+value*u)-b
            derivative = -a*u/(1+value*u)**2
            weights = np.minimum(1., 1.5/np.maximum(1e-8, np.abs(residual)))
            change = float(np.sum(weights*derivative*residual)/np.sum(weights*derivative**2))
            fraction = 1.
            while fraction > 1e-6:
                proposed = value-fraction*change
                if proposed > 0 and np.all(1+proposed*u > 1e-6):
                    break
                fraction *= .5
            if fraction <= 1e-6:
                break
            value = proposed
            if abs(fraction*change) < 1e-10:
                break
        factors.append(value)
    return factors


def axis_from_crop(image):
    """Return an undirected image-axis proxy and anisotropy, not 3D heading.

    Uses only this crop. Low spatial anisotropy or insufficient detail returns
    no axis. The score is a geometric descriptor, not calibrated confidence.
    """
    import cv2
    if image is None or min(image.shape[:2]) < 8:
        return None, 0.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    weight = np.hypot(gx, gy)
    weight[[0, -1], :] = 0; weight[:, [0, -1]] = 0
    if weight.sum() < 1e-8:
        return None, 0.
    y, x = np.indices(gray.shape)
    points = np.c_[x.ravel(), y.ravel()]
    weights = weight.ravel()/weight.sum()
    centered = points-np.sum(points*weights[:, None], axis=0)
    covariance = (centered*weights[:, None]).T@centered
    values, vectors = np.linalg.eigh(covariance)
    score = float((values[1]-values[0])/max(values.sum(), 1e-8))
    if score < .25:
        return None, score
    axis = vectors[:, 1]
    return float(np.arctan2(axis[1], axis[0]) % np.pi), score


def features(model, box, tick, orientation=None, axis_score=0.):
    box = box_array(box)
    ep, _ = edge_geometry(model, box, tick)
    centre = (box[:2]+box[2:])/2
    size = box[2:]-box[:2]
    x = (centre[0]-ep[0])/(model.source_size[0]/2)
    y = centre[1]/model.source_size[1]-.5
    aspect = np.log(size[0]/size[1])
    score = float(finite(axis_score, 'axis score'))
    if not 0 <= score <= 1:
        raise ValueError('Axis score must lie in [0, 1]')
    if orientation is None:
        c, s = 0., 0.
    else:
        angle = float(finite(orientation, 'orientation'))
        c, s = score*np.cos(2*angle), score*np.sin(2*angle)
    # Under horizontal reflection x and sin(2 angle) change sign. All even
    # features remain unchanged, including the interaction x*sin(2 angle).
    even = np.array([1., x*x, y, aspect, c, x*s])
    odd = np.array([x, s])
    return even, odd


def _design(even, odd, label, classes, counts):
    e = np.zeros(6*(len(classes)+1)); o = np.zeros(2*(len(classes)+1))
    e[:6] = even; o[:2] = odd
    if label in classes:
        index = classes.index(label)+1
        # One instance supports only a shrunk class intercept. It must not
        # teach a class-specific position/pose surface from its own trajectory.
        if counts[label] >= 3:
            e[6*index:6*(index+1)] = even
            o[2*index:2*(index+1)] = odd
        else:
            e[6*index] = 1.
    return e, o


@dataclass(frozen=True)
class ConditionedShapeModel:
    classes: tuple[str, ...]
    instance_counts: dict
    even_weights: np.ndarray
    odd_weights: np.ndarray
    reference_step_m: float
    training: dict
    max_factor: float = 1.15
    validated_classes: tuple[str, ...] = ()

    def __post_init__(self):
        if len(set(self.classes)) != len(self.classes):
            raise ValueError('Duplicate classes')
        for name, expected in [('even_weights', (6*(len(self.classes)+1), 3)),
                               ('odd_weights', (2*(len(self.classes)+1),))]:
            value = finite(getattr(self, name), name).copy()
            if value.shape != expected:
                raise ValueError(f'Wrong {name} shape')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if float(finite(self.reference_step_m, 'reference step')) <= 0:
            raise ValueError('Reference step must be positive')
        if float(finite(self.max_factor, 'factor bound')) <= 1:
            raise ValueError('Factor bound must exceed one')
        if any(label not in self.classes for label in self.validated_classes):
            raise ValueError('Unknown validated class')

    @classmethod
    def fit(cls, samples, *, reference_step_m, ridge=1., max_factor=1.15):
        """Fit offline sample records, balanced by physical training instance.

        Each record carries instance_id, sequence_id, label, even/odd features,
        and four fitted edge factors. Frame windows are not independent objects.
        Ridge values must be chosen without inspecting external test labels.
        """
        if not samples or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('Need training samples and positive regularization')
        classes = tuple(sorted({s['label'] for s in samples}))
        identities = {s['instance_id'] for s in samples}
        counts = {c: len({s['instance_id'] for s in samples if s['label'] == c}) for c in classes}
        appearances = {i: sum(s['instance_id'] == i for s in samples) for i in identities}
        es, os, ys, ds, weights = [], [], [], [], []
        coverage = {c: [] for c in classes}
        for sample in samples:
            even, odd = finite(sample['even'], 'even features'), finite(sample['odd'], 'odd features')
            if even.shape != (6,) or odd.shape != (2,):
                raise ValueError('Invalid conditioning features')
            factors = finite(sample['factors'], 'training factors')
            if factors.shape != (4,) or np.any(factors <= 0):
                raise ValueError('Invalid training profile')
            logs = np.log(factors)
            e, o = _design(even, odd, sample['label'], classes, counts)
            es.append(e); os.append(o)
            ys.append([(logs[0]+logs[2])/2, logs[1], logs[3]])
            ds.append((logs[2]-logs[0])/2)
            weights.append(1/appearances[sample['instance_id']])
            coverage[sample['label']].append(np.r_[even[1:], odd].tolist())
        weights = np.array(weights)
        def solve(x, target, global_width):
            x, target = np.array(x), np.array(target)
            penalty = np.full(x.shape[1], 10.*ridge)
            penalty[:global_width] = ridge
            weighted = x*weights[:, None]
            return np.linalg.solve(x.T@weighted+np.diag(penalty), weighted.T@target)
        return cls(classes, counts, solve(es, ys, 6), solve(os, ds, 2), reference_step_m,
                   {'instances': sorted(identities), 'sequences': sorted({s['sequence_id'] for s in samples}),
                    'sample_count': len(samples), 'ridge': ridge, 'coverage': coverage,
                    'class_geometry_min_instances': 3, 'samples_weighted_per_instance': True}, max_factor)

    def candidate_factors(self, label, model, box, tick, *, orientation=None, axis_score=0.):
        even, odd = features(model, box, tick, orientation, axis_score)
        e, o = _design(even, odd, label, self.classes, self.instance_counts)
        symmetric, top, bottom = e@self.even_weights
        antisymmetric = float(o@self.odd_weights)
        logs = np.array([symmetric-antisymmetric, top, symmetric+antisymmetric, bottom])
        bound = np.log(self.max_factor)
        return np.exp(np.clip(logs, -bound, bound))

    def predict(self, label, model, box, tick, *, distance_m, orientation=None,
                axis_score=0., class_confidence=1., diagnostic=False):
        """Use a validated correction, or explicitly return the shared fallback.

        diagnostic=True bypasses qualification/coverage for offline measurement.
        The default does not enable an unvalidated model simply because it fits.
        """
        distance = float(finite(distance_m, 'travelled distance'))
        confidence = float(finite(class_confidence, 'class confidence'))
        if distance < 0 or not 0 <= confidence <= 1:
            raise ValueError('Invalid distance or class confidence')
        baseline = model.box(box, tick, tick+distance/self.reference_step_m)
        reason = None
        if not diagnostic:
            if label not in self.validated_classes:
                reason = 'class_not_validated_on_independent_instances'
            elif confidence < .8:
                reason = 'uncertain_class'
            elif orientation is None or axis_score < .25:
                reason = 'uncertain_image_axis'
            else:
                even, odd = features(model, box, tick, orientation, axis_score)
                # Mirrored equivalents have identical even features and negate
                # both odd features. They are coverage equivalents, not new data.
                stored = np.array(self.training['coverage'][label])
                mirrored = stored.copy(); mirrored[:, -2:] *= -1
                stored = np.r_[stored, mirrored]
                value = np.r_[even[1:], odd]
                scales = np.array([.2, .2, .35, .35, .25, .3, .35])
                if np.min(np.max(np.abs(stored-value)/scales, axis=1)) > 1:
                    reason = 'outside_observed_shape_context'
        if reason:
            return {'box': baseline, 'mode': 'shared_fallback', 'reason': reason}
        factors = self.candidate_factors(label, model, box, tick, orientation=orientation, axis_score=axis_score)
        try:
            corrected_midpoints = EdgeMotionProfile(label, tuple(factors), self.reference_step_m).predict(
                model, box, tick, distance_m=distance)
            nominal_midpoints = EdgeMotionProfile(label, (1.,)*4, self.reference_step_m).predict(
                model, box, tick, distance_m=distance)
            box = baseline+(corrected_midpoints-nominal_midpoints)
            if np.any(box[2:] <= box[:2]):
                raise ProjectionError('Residual correction crosses box edges')
        except ProjectionError:
            return {'box': baseline, 'mode': 'shared_fallback', 'reason': 'invalid_profile_projection'}
        return {'box': box, 'mode': 'diagnostic_candidate' if diagnostic else 'conditioned_profile',
                'reason': None, 'factors': factors.tolist()}

    def qualify(self, records):
        """Conservative release gate, separate from fitting or model selection.

        Each record describes one independent held-out object. Tracker-generated
        pseudo-labels cannot qualify a class. Thresholds are engineering guards,
        not a statistical guarantee about unobserved data.
        """
        accepted, reasons = [], {}
        for label in self.classes:
            rows = [r for r in records if r['label'] == label]
            issues = []
            if self.instance_counts[label] < 2:
                issues.append('fewer_than_two_training_instances')
            if len({r['instance_id'] for r in rows}) < 2:
                issues.append('fewer_than_two_held_out_instances')
            if any(r['instance_id'] in self.training['instances'] or r['sequence_id'] in self.training['sequences'] for r in rows):
                issues.append('training_instance_or_flight_overlap')
            if any(r['label_source'] not in ('official', 'independent_reviewed') for r in rows):
                issues.append('evaluation_uses_tracker_pseudo_labels')
            if any(r['frames'] < 3 or r['candidate_passes'] != r['frames'] or
                   r['candidate_mean_iou'] < r['baseline_mean_iou'] or r['candidate_min_iou'] < .6 for r in rows):
                issues.append('held_out_regression_or_insufficient_margin')
            if issues:
                reasons[label] = issues
            else:
                accepted.append(label)
        return replace(self, validated_classes=tuple(accepted)), reasons

    def to_dict(self):
        return {'version': 1, 'classes': list(self.classes), 'instance_counts': self.instance_counts,
                'even_weights': self.even_weights.tolist(), 'odd_weights': self.odd_weights.tolist(),
                'reference_step_m': self.reference_step_m, 'training': self.training,
                'max_factor': self.max_factor, 'validated_classes': list(self.validated_classes)}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported conditioned-shape version')
        return cls(tuple(data['classes']), data['instance_counts'], data['even_weights'], data['odd_weights'],
                   data['reference_step_m'], data['training'], data['max_factor'], tuple(data['validated_classes']))
