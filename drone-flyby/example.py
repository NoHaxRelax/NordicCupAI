"""Live drone endpoint: detector, perspective tracker and upper-band camera sweep.

Each sequence gets its own workflow: two full-frame overviews calibrate the
scene motion, then the camera cycles L1 left, L0, L1 right, L0 across the
upper band. Every detection is placed in the organizer's box convention,
tracked while the camera looks elsewhere and refreshed whenever it is seen
whole again. The response always covers the whole source frame.

Configuration is by environment variables (defaults in brackets):

  DRONE_DETECTOR            ultralytics | none | oracle | module:factory  [ultralytics if DRONE_WEIGHTS]
  DRONE_WEIGHTS             local checkpoint path
  DRONE_DEVICE              cpu | cuda:0 | mps                             [cpu]
  DRONE_IMGSZ, DRONE_CONF   detector input size and confidence floor        [960, 0.25]
  DRONE_DETECT_EVERY        run the detector on every k-th frame            [1]
  DRONE_EXTENT_POLICY       detector | blend | prior                        [blend]
  DRONE_EMIT_PARTIALS, DRONE_ENTRY_TRACKS, DRONE_CLIP_LAST_INDEX            [1, 1, 1]
  DRONE_BIRTH_CONFIDENCE, DRONE_UPDATE_CONFIDENCE                           [0.6, 0.4]
  DRONE_VERTICAL_FRACTION   band of the L1 crops, 0 = top                   [0]
  DRONE_OVERVIEW_BETWEEN_SIDES  L0 between the L1 sides (0 = L1 centre)     [1]
  DRONE_CAMERA_MODE         l1 (upper L1 sweep) | l2_top (native L2 sweep of the top row) [l1]
  DRONE_REVISIT_EVERY, DRONE_REVISIT_MIN_AGE  every k-th frame aim L2 at the oldest reachable track [0, 6]
  DRONE_OBSERVE_MOTION      image-based motion clock for frozen/double steps [1]
  DRONE_LOG_DIR             per-sequence diagnostics JSONL                  [unset]
"""
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from dtos import (DroneFlybyPredictionDto, DroneFlybyPredictRequestDto,
                  DroneFlybyPredictResponseDto, RequestedViewDto)
from utils import decode_view
from tracking import Detection, DroneTrackingWorkflow, RevisitConfig, ViewGeometry
from tracking.revisit import frame_rows
from tracking.tracker import OBJECT_CLASSES as _CLASSES
from detectors import build_detector

logger = logging.getLogger(__name__)


def _flag(name, default):
    return os.environ.get(name, '1' if default else '0').strip() not in ('0', '', 'false', 'no')


SETTINGS = {
    'detect_every': max(1, int(os.environ.get('DRONE_DETECT_EVERY', '1'))),
    'vertical_fraction': float(os.environ.get('DRONE_VERTICAL_FRACTION', '0')),
    'overview_between_sides': _flag('DRONE_OVERVIEW_BETWEEN_SIDES', True),
    'observe_motion': _flag('DRONE_OBSERVE_MOTION', True),
    'camera_mode': os.environ.get('DRONE_CAMERA_MODE', 'l1'),
    'revisit_every': int(os.environ.get('DRONE_REVISIT_EVERY', '0')),
    'revisit_min_age': float(os.environ.get('DRONE_REVISIT_MIN_AGE', '6')),
    'log_dir': os.environ.get('DRONE_LOG_DIR') or None,
    'max_sessions': 4,
}
CONFIG = RevisitConfig(
    extent_policy=os.environ.get('DRONE_EXTENT_POLICY', 'blend'),
    emit_partials=_flag('DRONE_EMIT_PARTIALS', True),
    entry_tracks=_flag('DRONE_ENTRY_TRACKS', True),
    clip_last_index=_flag('DRONE_CLIP_LAST_INDEX', True),
    birth_confidence=float(os.environ.get('DRONE_BIRTH_CONFIDENCE', '0.6')),
    update_confidence=float(os.environ.get('DRONE_UPDATE_CONFIDENCE', '0.4')),
)
DETECTOR = build_detector()
logger.info('Detector: %s; config: %s; settings: %s', getattr(DETECTOR, 'name', type(DETECTOR).__name__), CONFIG, SETTINGS)


class Session:
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.lock = threading.Lock()
        self.workflow = self.new_workflow()
        self.prior = self.workflow.prior
        self.frames = 0
        self.failures = 0
        self.log = None
        if SETTINGS['log_dir']:
            directory = Path(SETTINGS['log_dir']); directory.mkdir(parents=True, exist_ok=True)
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in sequence_id)[:80]
            self.log = (directory/f'{safe}.jsonl').open('a')

    @staticmethod
    def new_workflow():
        return DroneTrackingWorkflow(CONFIG, observe_motion=SETTINGS['observe_motion'],
                                     vertical_fraction=SETTINGS['vertical_fraction'],
                                     overview_between_sides=SETTINGS['overview_between_sides'],
                                     camera_mode=SETTINGS['camera_mode'],
                                     revisit_every=SETTINGS['revisit_every'], revisit_min_age=SETTINGS['revisit_min_age'])

    def record(self, row):
        if self.log:
            self.log.write(json.dumps(row, allow_nan=False, default=str)+'\n'); self.log.flush()


_sessions = OrderedDict()
_sessions_lock = threading.Lock()


def _session(sequence_id):
    with _sessions_lock:
        session = _sessions.get(sequence_id)
        if session is None:
            session = _sessions[sequence_id] = Session(sequence_id)
            while len(_sessions) > SETTINGS['max_sessions']:
                _, old = _sessions.popitem(last=False)
                if old.log:
                    old.log.close()
        _sessions.move_to_end(sequence_id)
        return session


def _detections(image, request, view):
    """Run the detector and convert its rows; never raise into the frame."""
    if request['frame_index'] % SETTINGS['detect_every']:
        return [], False, 0.
    started = time.perf_counter()
    try:
        rows = DETECTOR(image, request)
    except Exception:
        logger.exception('Detector failed on frame %s', request['frame'])
        return [], False, (time.perf_counter()-started)*1000
    width, height = view.image_size
    detections = []
    for row in rows:
        try:
            box = np.clip(np.asarray(row['box'], float), 0, [width, height, width, height])
            if np.any(box[2:]-box[:2] < 1) or row['label'] not in _CLASSES:
                continue
            detections.append(Detection(row['label'], tuple(box.tolist()), float(min(1., max(0., row['confidence'])))))
            _RAW.append({'label': row['label'], 'box': [round(v, 1) for v in box.tolist()],
                         'confidence': round(float(row['confidence']), 3), 'family': row.get('family')})
        except (KeyError, TypeError, ValueError):
            continue
    return detections, True, (time.perf_counter()-started)*1000


_RAW = []  # raw detector rows of the current frame, for the diagnostics log


def _fallback(request, detections, session, view):
    """A valid answer from this frame's detections alone, holding the camera."""
    rows = frame_rows(detections, view, CONFIG, session.prior)
    size = np.tile(view.source_size, 2)
    return {'request_id': request['request_id'], 'frame': request['frame'],
            'annotations': [{'object_id': r['object_id'], 'confidence': r['confidence'],
                             'bbox': (np.array(r['bbox_source_xyxy'])/size).tolist()} for r in rows][:500],
            'requested_view': None}


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    started = time.perf_counter()
    req = request.model_dump()
    session = _session(req['sequence_id'])
    with session.lock:
        session.frames += 1
        if request.camera_command_feedback is not None:
            logger.warning('Camera command from frame %s was ignored: %s',
                           request.camera_command_feedback.frame, request.camera_command_feedback.reason)
        view = ViewGeometry.from_request(req)
        image = decode_view(request.view)
        _RAW.clear()
        detections, ran, detector_ms = _detections(image, req, view)
        raw_rows = list(_RAW)
        tracking_started = time.perf_counter()
        try:
            answer = session.workflow.process(req, detections, image=image, detector_ran=ran)
            session.failures = 0
            diagnostics = session.workflow.diagnostics
        except Exception:
            session.failures += 1
            logger.exception('Workflow failed on frame %s (failure %d)', req['frame'], session.failures)
            answer = _fallback(req, detections, session, view)
            diagnostics = {'status': 'fallback'}
            if session.failures >= 3:
                logger.error('Resetting workflow for sequence %s after repeated failures', req['sequence_id'])
                session.workflow = session.new_workflow(); session.failures = 0
        tracking_ms = (time.perf_counter()-tracking_started)*1000
        requested = answer.get('requested_view')
        response = DroneFlybyPredictResponseDto(
            request_id=req['request_id'], frame=req['frame'],
            annotations=[DroneFlybyPredictionDto(object_id=a['object_id'], bbox=list(a['bbox']),
                                                 confidence=float(a['confidence'])) for a in answer['annotations']],
            requested_view=RequestedViewDto(**requested) if requested else None)
        total_ms = (time.perf_counter()-started)*1000
        session.record({'frame': req['frame'], 'frame_index': req['frame_index'], 'level': req['view']['resolution_level'],
                        'region': req['view']['source_region_xyxy'], 'detections': len(detections), 'detector_ran': ran,
                        'annotations': len(response.annotations), 'requested_view': requested,
                        'detector_ms': round(detector_ms, 1), 'tracking_ms': round(tracking_ms, 1), 'total_ms': round(total_ms, 1),
                        'status': diagnostics.get('status'), 'timing': diagnostics.get('timing'),
                        'calibration_error': diagnostics.get('calibration_error'), 'events': diagnostics.get('events'),
                        'tracks': len(diagnostics.get('tracks') or []), 'raw_detections': raw_rows[:200],
                        'response': [{'object_id': a.object_id, 'bbox': [round(v, 5) for v in a.bbox], 'confidence': round(float(a.confidence), 3)}
                                     for a in response.annotations][:500]})
        logger.info('frame %s L%s: %d detections, %d annotations, detector %.0f ms, tracking %.0f ms, total %.0f ms',
                    req['frame'], req['view']['resolution_level'], len(detections), len(response.annotations),
                    detector_ms, tracking_ms, total_ms)
    return response
