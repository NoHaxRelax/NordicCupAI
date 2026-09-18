"""Learn the effective motion of box edges from repeated training views.

The four factors capture departure from the calibrated reference plane. They
are not measured physical heights. Once learned, a profile needs one new box
and travelled distance to predict. Profiles must be tested on unseen instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .motion import MotionModel, ProjectionError, box_array, finite


def edge_geometry(model: MotionModel, box, tick):
    box = box_array(box)
    x1, y1, x2, y2 = box
    cx, cy = (box[:2]+box[2:])/2
    # Midpoints approximate the physical points responsible for each extremum.
    points = np.array([[x1, cy], [cx, y1], [x2, cy], [cx, y2]])
    model.mapping(tick, tick)
    # A=e q^T. Its third row is q^T when the epipole is normalized to z=1.
    q = model.matrix[2]
    if np.linalg.norm(q) < 1e-12:
        raise ValueError('Edge profiles currently require a finite epipole')
    column = int(np.argmax(np.abs(q)))
    e = model.matrix[:, column]/q[column]
    ep = np.array([e[0], e[1], e[0], e[1]])
    rates = np.c_[points, np.ones(4)]@q/(1+(tick-model.origin_tick)*np.trace(model.matrix))
    return ep, rates


@dataclass(frozen=True)
class EdgeMotionProfile:
    """Four dimensionless edge-motion factors in x1,y1,x2,y2 order."""
    label: str
    factors: tuple[float, float, float, float]
    reference_step_m: float
    training: dict = field(default_factory=dict)

    def __post_init__(self):
        factors = finite(self.factors, 'edge factors')
        if factors.shape != (4,) or np.any(factors <= 0):
            raise ValueError('Expected four positive edge-motion factors')
        step = float(finite(self.reference_step_m, 'reference step'))
        if step <= 0:
            raise ValueError('Reference distance per motion tick must be positive')
        object.__setattr__(self, 'factors', tuple(float(v) for v in factors))
        object.__setattr__(self, 'reference_step_m', step)

    def predict(self, model: MotionModel, box, anchor_tick, *, distance_m):
        """Predict after total travelled distance, with no later object views.

        For constant speed v and elapsed seconds dt, distance_m=v*dt. With
        changing speed use integrated distance, not just the final speed.
        Camera orientation/altitude and the applicable calibration must hold.
        anchor_tick must use the calibration's distance-based motion clock, not
        a raw frame index when travelled distance per frame changes.
        """
        box = box_array(box)
        distance = float(finite(distance_m, 'travelled distance'))
        if distance < 0:
            raise ValueError('Forecast distance must be nonnegative')
        ticks = distance/self.reference_step_m
        model.mapping(anchor_tick, anchor_tick+ticks)
        ep, rates = edge_geometry(model, box, anchor_tick)
        denominator = 1+ticks*rates*np.array(self.factors)
        if np.any(denominator <= 1e-9):
            raise ProjectionError('Profile reaches its projection horizon')
        predicted = ep+(box-ep)/denominator
        if np.any(predicted[2:] <= predicted[:2]):
            raise ProjectionError('Predicted extrema cross; the profile is no longer applicable')
        return predicted

    @classmethod
    def fit(cls, model, training_tracks, *, label, reference_step_m, minimum_gap=2):
        """Fit on supplied training instances; no implicit dataset lookup.

        training_tracks maps an instance ID to [(tick, source_box), ...].
        Clipped boxes are excluded. Every fit may inspect all supplied training
        views; callers must exclude held-out objects before calling this method.
        """
        minimum_gap = float(finite(minimum_gap, 'minimum training gap'))
        if minimum_gap <= 0:
            raise ValueError('Minimum training gap must be positive')
        starts, ends, inputs, instance_counts = [], [], [], {}
        for name, observations in training_tracks.items():
            usable = [(float(f), box_array(b)) for f, b in sorted(observations)
                      if np.all(box_array(b)[:2] > 0) and
                      np.all(box_array(b)[2:] < np.array(model.source_size)-1)]
            instance_counts[name] = len(usable)
            for i, (a, box) in enumerate(usable):
                ep, rates = edge_geometry(model, box, a)
                for b, later in usable[i+1:]:
                    if b-a < minimum_gap:
                        continue
                    starts.append(box-ep); ends.append(later-ep)
                    inputs.append((b-a)*rates)
        if len(starts) < 3:
            raise ValueError('Need at least three usable training-view pairs')
        start, end, u = np.array(starts), np.array(ends), np.array(inputs)
        factors, fit_errors = [], []
        for edge in range(4):
            a, b, x = start[:, edge], end[:, edge], u[:, edge]
            valid = (np.abs(a) > 1e-5) & (np.abs(b) > 1e-5) & (a*b > 0) & (np.abs(x) > 1e-8)
            a, b, x = a[valid], b[valid], x[valid]
            if len(a) < 3:
                raise ValueError('Training does not constrain every edge')
            target = a/b-1
            value = float(np.sum(x*target)/np.sum(x*x))
            if value <= 0 or np.any(1+value*x <= 0):
                raise ValueError('No valid initial edge-motion fit')
            # Robust Gauss-Newton on source-pixel residuals, not random noise.
            for _ in range(25):
                denominator = 1+value*x
                residual = a/denominator-b
                derivative = -a*x/(denominator**2)
                weight = np.minimum(1., 1.5/np.maximum(1e-8, np.abs(residual)))
                change = float(np.sum(weight*derivative*residual)/np.sum(weight*derivative**2))
                fraction = 1.
                while fraction > 1e-6:
                    proposed = value-fraction*change
                    if proposed > 0 and np.all(1+proposed*x > 1e-6):
                        break
                    fraction *= .5
                if fraction <= 1e-6:
                    break
                value = proposed
                if abs(fraction*change) < 1e-10:
                    break
            factors.append(value)
            fit_errors.append(float(np.median(np.abs(a/(1+value*x)-b))))
        return cls(label, tuple(factors), reference_step_m,
                   {'instance_views': instance_counts, 'view_pairs': len(starts),
                    'median_training_edge_error_px': fit_errors,
                    'interpretation': 'Effective relative motion/depth factors, not heights in metres.'})

    def to_dict(self):
        return {'version': 1, 'label': self.label, 'factors': list(self.factors),
                'reference_step_m': self.reference_step_m, 'training': self.training}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported edge profile version')
        return cls(data['label'], tuple(data['factors']), data['reference_step_m'], data.get('training', {}))
