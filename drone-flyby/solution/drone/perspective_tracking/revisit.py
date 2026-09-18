"""Detector-agnostic tracking with automatic correction on reliable revisits.

Runtime only: no dataset paths or class-shape training. All boxes are stored in
source coordinates; unseen objects keep their forecasts when the camera pans.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .motion import MotionModel, ProjectionError, ViewGeometry, box_array, finite
from .tracker import OBJECT_CLASSES


def overlap(a, b):
    intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return float(intersection/union) if union > 0 else 0.


@dataclass(frozen=True)
class Detection:
    """A detector box in delivered-image pixels, with a competition class name.

    complete=False excludes a known truncated box from full-extent updates.
    Boxes touching the crop edge are also treated as truncated automatically.
    """
    label: str
    box: tuple[float, float, float, float]
    confidence: float
    complete: bool = True

    def __post_init__(self):
        if self.label not in OBJECT_CLASSES:
            raise ValueError(f'Unknown competition class: {self.label}')
        b = box_array(self.box)
        c = float(finite(self.confidence, 'confidence'))
        if not 0 <= c <= 1 or not isinstance(self.complete, bool):
            raise ValueError('Invalid detection confidence or completeness')
        object.__setattr__(self, 'box', tuple(float(v) for v in b))
        object.__setattr__(self, 'confidence', c)


@dataclass(frozen=True)
class RevisitConfig:
    birth_confidence: float = .6
    update_confidence: float = .4
    association_iou: float = .1
    association_distance: float = 1.0
    ambiguity_margin: float = .1
    duplicate_iou: float = .7
    crop_margin_pixels: float = 1.
    visible_misses_before_retirement: int = 3
    max_history: int = 6
    adapt_edges: bool = False

    def __post_init__(self):
        for name in ('birth_confidence', 'update_confidence', 'association_iou',
                     'ambiguity_margin', 'duplicate_iou'):
            if not 0 <= float(finite(getattr(self, name), name)) <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        if self.birth_confidence < self.update_confidence:
            raise ValueError('Birth confidence must be at least update confidence')
        if self.association_distance <= 0 or not np.isfinite(self.association_distance):
            raise ValueError('Association distance must be finite and positive')
        if self.crop_margin_pixels < 0 or not np.isfinite(self.crop_margin_pixels):
            raise ValueError('Crop margin must be finite and nonnegative')
        if self.visible_misses_before_retirement < 1 or self.max_history < 4:
            raise ValueError('Need positive retirement count and at least four history slots')


@dataclass
class RevisitedTrack:
    track_id: str
    label: str
    history: list
    confidence: float
    last_seen_tick: float
    last_seen_frame: int
    visible_misses: int = 0
    edge_slopes: list | None = None
    adaptation_checked: bool = False


def edge_slopes(model, history):
    """A bounded recent-observation fit, available only as an opt-in mode."""
    q = model.matrix[2]
    if len(history) < 3 or np.linalg.norm(q) < 1e-12:
        return None
    e = model.matrix[:, np.argmax(np.abs(q))]/q[np.argmax(np.abs(q))]
    ep = np.tile(e[:2], 2)
    ticks = np.array([r[0] for r in history]); boxes = np.array([r[1] for r in history])
    offsets = boxes-ep
    if np.ptp(ticks) < 2 or np.any(np.abs(offsets) < 2) or np.any(offsets*offsets[-1] <= 0):
        return None
    times = ticks-ticks.mean()
    slope = np.sum(times[:, None]/offsets, axis=0)/np.sum(times**2)
    return slope.tolist()


def edge_forecast(model, history, slopes, tick):
    anchor, box = history[-1]; box = np.array(box)
    q = model.matrix[2]; e = model.matrix[:, np.argmax(np.abs(q))]/q[np.argmax(np.abs(q))]
    ep = np.tile(e[:2], 2)
    inverse = 1/(box-ep)+(tick-anchor)*np.array(slopes)
    if np.any(inverse*(box-ep) <= 0):
        raise ProjectionError('Adaptive edge crosses its horizon')
    result = ep+1/inverse
    size = result[2:]-result[:2]
    base = model.box(box, anchor, tick); base_size = base[2:]-base[:2]
    if np.any(size <= 0) or np.any(size/base_size < .25) or np.any(size/base_size > 4):
        raise ProjectionError('Adaptive extent is unsupported')
    return result


class RevisitTracker:
    """Associate same-class detections, refresh complete boxes, predict gaps.

    Call update once per delivered frame, with detections from that frame only.
    A missing detection outside the current crop is not a disappearance.
    tick can be an independently measured cumulative motion clock; repeated
    image positions may have equal ticks, but frame_index must increase.
    """
    def __init__(self, model, sequence_id, config=None):
        if not sequence_id:
            raise ValueError('A sequence ID is required')
        self.model = model
        self.sequence_id = sequence_id
        self.config = config or RevisitConfig()
        self.tracks = {}
        self.next_id = 1
        self.last_frame = None
        self.last_tick = None
        self.events = []

    def _box(self, track, tick):
        if track.edge_slopes is not None:
            try:
                return edge_forecast(self.model, track.history, track.edge_slopes, tick)
            except ProjectionError:
                pass
        anchor, box = track.history[-1]
        return self.model.box(box, anchor, tick)

    def _accept(self, track, box, tick, frame, confidence):
        # Frozen render frames replace the same-position observation rather than
        # adding a zero-time velocity sample.
        previous = [r for r in track.history if r[0] < tick]
        slopes = edge_slopes(self.model, previous) if self.config.adapt_edges else None
        track.edge_slopes = None
        track.adaptation_checked = False
        if slopes is not None:
            try:
                candidate = edge_forecast(self.model, previous, slopes, tick)
                baseline = self.model.box(previous[-1][1], previous[-1][0], tick)
                error = np.mean(np.abs(candidate-box))
                old_error = np.mean(np.abs(baseline-box))
                # Qualify on the new observation BEFORE including it in a fit.
                track.adaptation_checked = True
                if error < .8*old_error and overlap(candidate, box) >= .5:
                    track.edge_slopes = edge_slopes(self.model, (previous+[[tick, box.tolist()]])[-self.config.max_history:])
            except ProjectionError:
                pass
        track.history = (previous+[[tick, box.tolist()]])[-self.config.max_history:]
        track.confidence = confidence
        track.last_seen_tick = tick; track.last_seen_frame = frame; track.visible_misses = 0

    def update(self, detections, view, tick, frame_index, *, detector_ran=True):
        tick = float(finite(tick, 'tick'))
        if view.source_size != self.model.source_size:
            raise ValueError('Wrong source dimensions')
        if self.last_frame is not None and frame_index <= self.last_frame:
            raise ValueError('Frame indices must increase; do not replay a request')
        if self.last_tick is not None and tick < self.last_tick:
            raise ValueError('Motion clock must not move backwards')
        self.model.mapping(tick, tick)
        # Validate every detection before mutating track state.
        incoming = []
        for detection in detections:
            d = detection if isinstance(detection, Detection) else Detection(**detection)
            source = view.box_to_source(d.box)
            margin = self.config.crop_margin_pixels
            complete = (d.complete and np.all(np.array(d.box[:2]) > margin) and
                        np.all(np.array(d.box[2:]) < np.array(view.image_size)-margin))
            if d.confidence >= self.config.update_confidence:
                incoming.append((d, source, complete))
        if not detector_ran and incoming:
            raise ValueError('Detections supplied with detector_ran=False')
        kept = []
        for row in sorted(incoming, key=lambda r: -r[0].confidence):
            if not any(row[0].label == other[0].label and overlap(row[1], other[1]) >= self.config.duplicate_iou for other in kept):
                kept.append(row)
        self.events = []
        predictions = {}
        for identity, track in list(self.tracks.items()):
            try:
                box = self._box(track, tick)
            except ProjectionError:
                self.events.append({'event': 'projection_unavailable', 'track_id': identity})
                continue
            if np.any(np.minimum(box[2:], view.source_size) <= np.maximum(box[:2], 0)):
                self.events.append({'event': 'source_exit', 'track_id': identity})
                del self.tracks[identity]
            else:
                predictions[identity] = box
        candidates = []
        had_candidate = set()
        region = np.array(view.region)
        for i, (d, box, complete) in enumerate(kept):
            for identity, predicted in predictions.items():
                track = self.tracks[identity]
                if d.label != track.label:
                    continue
                visible = np.r_[np.maximum(predicted[:2], region[:2]), np.minimum(predicted[2:], region[2:])]
                if np.any(visible[2:] <= visible[:2]):
                    continue
                intersection = overlap(visible, box)
                distance = np.linalg.norm((visible[:2]+visible[2:]-box[:2]-box[2:])/2)
                scale = max(8., np.linalg.norm(visible[2:]-visible[:2]))
                ratio = (box[2:]-box[:2])/(visible[2:]-visible[:2])
                if np.any(ratio < .25) or np.any(ratio > 4):
                    continue
                if intersection >= self.config.association_iou or distance/scale <= self.config.association_distance:
                    score = intersection+.5*max(0., 1-distance/scale)
                    candidates.append((score, i, identity)); had_candidate.add(i)
        matched_detections, matched_tracks = set(), set()
        for score, index, identity in sorted(candidates, reverse=True):
            if index in matched_detections or identity in matched_tracks:
                continue
            alternatives = [s for s, i, k in candidates if (i == index and k != identity) or (k == identity and i != index)]
            if alternatives and score-max(alternatives) < self.config.ambiguity_margin:
                continue
            d, box, complete = kept[index]; track = self.tracks[identity]
            if complete:
                before = predictions[identity]
                self._accept(track, box, tick, frame_index, d.confidence)
                self.events.append({'event': 'refresh', 'track_id': identity, 'prior_iou': overlap(before, box),
                                    'adapted': track.edge_slopes is not None})
            else:
                track.last_seen_tick = tick; track.last_seen_frame = frame_index; track.visible_misses = 0
                self.events.append({'event': 'partial_seen', 'track_id': identity})
            matched_detections.add(index); matched_tracks.add(identity)
        for index, (d, box, complete) in enumerate(kept):
            if index in matched_detections:
                continue
            if index in had_candidate:
                self.events.append({'event': 'ambiguous_detection', 'label': d.label})
                continue
            # Conflicting classifications at an existing object's location are
            # not enough to create a second prediction for the same object.
            if any(overlap(box, p) > .5 for p in predictions.values()):
                self.events.append({'event': 'conflicting_detection', 'label': d.label})
                continue
            if not complete or d.confidence < self.config.birth_confidence:
                continue
            identity = f'track-{self.next_id:05d}'; self.next_id += 1
            self.tracks[identity] = RevisitedTrack(identity, d.label, [[tick, box.tolist()]],
                                                 d.confidence, tick, frame_index)
            matched_tracks.add(identity)
            self.events.append({'event': 'birth', 'track_id': identity, 'label': d.label})
        ambiguous_tracks = {identity for _, index, identity in candidates if index not in matched_detections}
        if detector_ran:
            for identity, predicted in predictions.items():
                if identity in matched_tracks or identity in ambiguous_tracks:
                    continue
                # Only a full opportunity to see an object can count as a miss.
                margin = self.config.crop_margin_pixels*np.tile(view.scale, 2)
                if np.all(predicted[:2] > region[:2]+margin[:2]) and np.all(predicted[2:] < region[2:]-margin[2:]):
                    track = self.tracks[identity]; track.visible_misses += 1
                    if track.visible_misses >= self.config.visible_misses_before_retirement:
                        self.events.append({'event': 'visible_misses_retired', 'track_id': identity})
                        del self.tracks[identity]
        self.last_frame, self.last_tick = int(frame_index), tick
        return self.predictions(tick)

    def predictions(self, tick):
        tick = float(finite(tick, 'tick'))
        if self.last_tick is not None and tick < self.last_tick:
            raise ValueError('Cannot query before the latest update')
        rows = []
        for track in self.tracks.values():
            try:
                full = self._box(track, tick)
            except ProjectionError:
                continue
            clipped = np.clip(full, 0, np.tile(self.model.source_size, 2))
            if np.any(clipped[2:] <= clipped[:2]):
                continue
            rows.append({'track_id': track.track_id, 'object_id': track.label,
                         'bbox': (clipped/np.tile(self.model.source_size, 2)).tolist(),
                         'bbox_source_xyxy': clipped.tolist(), 'confidence': track.confidence,
                         'last_seen_frame': track.last_seen_frame, 'anchor_tick': track.history[-1][0],
                         'observations': len(track.history), 'adapted': track.edge_slopes is not None})
        return rows

    def response(self, request, *, tick=None, requested_view=None):
        if request['sequence_id'] != self.sequence_id:
            raise ValueError('Sequence changed; use a separate tracker')
        if (request['original_width'], request['original_height']) != self.model.source_size:
            raise ValueError('Source dimensions changed')
        rows = self.predictions(request['frame_index'] if tick is None else tick)
        output, counts = [], {}
        for row in sorted(rows, key=lambda r: -r['confidence']):
            label = row['object_id']
            if len(output) == 500 or counts.get(label, 0) == 100:
                continue
            counts[label] = counts.get(label, 0)+1
            output.append({key: row[key] for key in ('object_id', 'bbox', 'confidence')})
        return {'request_id': request['request_id'], 'frame': request['frame'],
                'annotations': output, 'requested_view': requested_view}

    def to_dict(self):
        return {'version': 1, 'model': self.model.to_dict(), 'sequence_id': self.sequence_id,
                'config': asdict(self.config), 'tracks': [asdict(t) for t in self.tracks.values()],
                'next_id': self.next_id, 'last_frame': self.last_frame, 'last_tick': self.last_tick}

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported revisit state version')
        result = cls(MotionModel.from_dict(data['model']), data['sequence_id'], RevisitConfig(**data['config']))
        for row in data['tracks']:
            track = RevisitedTrack(**row)
            if track.label not in OBJECT_CLASSES or not track.history:
                raise ValueError('Invalid saved track')
            for tick, box in track.history:
                finite(tick, 'saved tick'); box_array(box)
            result.tracks[track.track_id] = track
        result.next_id = data['next_id']; result.last_frame = data['last_frame']; result.last_tick = data['last_tick']
        return result
