"""Live drone endpoint: detector, perspective tracker and camera sweep.

Each sequence gets its own workflow: two full-frame overviews calibrate the
scene motion, then the camera sweeps (L1 band, or native L2 crops along the
top row). Every detection is placed in the organizer's box convention,
tracked while the camera looks elsewhere and refreshed whenever it is seen
whole again. The response always covers the whole source frame.

Two modes:

* synchronous (default): the detector runs inside the request, so the answer
  waits for it.
* asynchronous (``DRONE_ASYNC=1``): every request is answered at once from
  the tracker's forecasts, while a pool of detector worker processes works
  through the delivered views in the background; their detections are fed
  back into the tracker at the tick of the frame they came from, so a slow
  detector costs a few frames of delay per object instead of skipped frames.

Configuration is by environment variables (defaults in brackets):

  DRONE_DETECTOR            ultralytics | fixed_assets | none | oracle | module:factory
  DRONE_WEIGHTS, DRONE_BUNDLE, DRONE_PROJECT, DRONE_DEVICE, DRONE_IMGSZ, DRONE_CONF, DRONE_FAMILY_LEVELS, DRONE_FAMILY_MIN
  DRONE_BUNDLE_FAST         1 = exact GPU neighbours, threaded fits, concurrent branches [0]
  DRONE_ASYNC, DRONE_WORKERS, DRONE_BACKLOG   asynchronous detection, worker processes, queued views [0, 4, workers]
  DRONE_DETECT_EVERY        run the detector on every k-th frame (synchronous mode)  [1]
  DRONE_EXTENT_POLICY       detector | blend | prior                        [blend]
  DRONE_CLASS_EXTENT        JSON per-class override, e.g. {"small_tower":"prior"} [{}]
  DRONE_EMIT_PARTIALS, DRONE_ENTRY_TRACKS, DRONE_CLIP_LAST_INDEX            [1, 1, 1]
  DRONE_BIRTH_CONFIDENCE, DRONE_UPDATE_CONFIDENCE                           [0.6, 0.4]
  DRONE_CAMERA_MODE         l1 (upper L1 sweep) | l2_top (native L2 sweep of the top row) [l1]
  DRONE_VERTICAL_FRACTION, DRONE_OVERVIEW_BETWEEN_SIDES                     [0, 1]
  DRONE_REVISIT_EVERY, DRONE_REVISIT_MIN_AGE  every k-th frame aim L2 at the oldest reachable track [0, 6]
  DRONE_OBSERVE_MOTION      image-based motion clock for frozen/double steps [1]
  DRONE_CV_THREADS          cap OpenCV/torch CPU threads per process (0 = default) [0]
  DRONE_LOG_DIR             per-sequence diagnostics JSONL                  [unset]
"""
import base64
import collections
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
    'async': _flag('DRONE_ASYNC', False),
    'workers': max(1, int(os.environ.get('DRONE_WORKERS', '4'))),
}
SETTINGS['backlog'] = max(1, int(os.environ.get('DRONE_BACKLOG', str(SETTINGS['workers']))))
CONFIG = RevisitConfig(
    extent_policy=os.environ.get('DRONE_EXTENT_POLICY', 'blend'),
    emit_partials=_flag('DRONE_EMIT_PARTIALS', True),
    entry_tracks=_flag('DRONE_ENTRY_TRACKS', True),
    clip_last_index=_flag('DRONE_CLIP_LAST_INDEX', True),
    birth_confidence=float(os.environ.get('DRONE_BIRTH_CONFIDENCE', '0.6')),
    update_confidence=float(os.environ.get('DRONE_UPDATE_CONFIDENCE', '0.4')),
    class_extent=json.loads(os.environ.get('DRONE_CLASS_EXTENT', '{}')) or None,
)
# Several replays share one machine: cap the per-process thread pools so
# concurrent processes do not thrash (0 keeps the library defaults).
_THREADS = int(os.environ.get('DRONE_CV_THREADS', '0'))
if _THREADS > 0:
    import cv2
    cv2.setNumThreads(_THREADS)
    try:
        import torch
        torch.set_num_threads(_THREADS)
    except ImportError:
        pass


from detector_pool import DetectorPool


import multiprocessing as _mp
if SETTINGS['async'] and _mp.current_process().name == 'MainProcess':
    DETECTOR = None
    POOL = DetectorPool(SETTINGS['workers'], SETTINGS['backlog'])
elif SETTINGS['async']:
    DETECTOR = None; POOL = None  # a spawned worker importing this module must not build a pool
else:
    DETECTOR = build_detector()
    POOL = None
logger.info('Detector: %s; async=%s workers=%s; config: %s; settings: %s',
            getattr(DETECTOR, 'name', None) or os.environ.get('DRONE_DETECTOR'), SETTINGS['async'], SETTINGS['workers'], CONFIG, SETTINGS)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

class Session:
    def __init__(self, sequence_id):
        self.sequence_id = sequence_id
        self.lock = threading.Lock()
        self.workflow = self.new_workflow()
        self.prior = self.workflow.prior
        self.frames = 0
        self.failures = 0
        self.late = collections.Counter()
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


def _session(sequence_id, create=True):
    with _sessions_lock:
        session = _sessions.get(sequence_id)
        if session is None:
            if not create:
                return None
            session = _sessions[sequence_id] = Session(sequence_id)
            while len(_sessions) > SETTINGS['max_sessions']:
                _, old = _sessions.popitem(last=False)
                if old.log:
                    old.log.close()
        _sessions.move_to_end(sequence_id)
        return session


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #

def _rows_to_detections(rows, view, raw=None):
    width, height = view.image_size
    detections = []
    for row in rows:
        try:
            box = np.clip(np.asarray(row['box'], float), 0, [width, height, width, height])
            if np.any(box[2:]-box[:2] < 1) or row['label'] not in _CLASSES:
                continue
            detections.append(Detection(row['label'], tuple(box.tolist()), float(min(1., max(0., row['confidence'])))))
            if raw is not None:
                raw.append({'label': row['label'], 'box': [round(v, 1) for v in box.tolist()],
                            'confidence': round(float(row['confidence']), 3), 'family': row.get('family')})
        except (KeyError, TypeError, ValueError):
            continue
    return detections


def _detect_now(image, request, view, raw):
    """Synchronous detection; never raises into the frame."""
    if request['frame_index'] % SETTINGS['detect_every']:
        return [], False, 0.
    started = time.perf_counter()
    try:
        rows = DETECTOR(image, request)
    except Exception:
        logger.exception('Detector failed on frame %s', request['frame'])
        return [], False, (time.perf_counter()-started)*1000
    return _rows_to_detections(rows, view, raw), True, (time.perf_counter()-started)*1000


def _apply_late_results(current_session):
    """Feed finished background detections into their sessions' trackers."""
    applied = []
    for sequence_id, frame_index, rows, error, ms in POOL.drain():
        session = _session(sequence_id, create=False)
        if session is None:
            continue
        if error:
            logger.error('Background detector failed on frame index %s: %s', frame_index, error)
            session.late['errors'] += 1; continue
        held = session is not current_session
        if held:
            session.lock.acquire()
        try:
            view = session.workflow.frame_views.get(frame_index, (None, None))[1]
            if view is None:
                session.late['unknown_frame'] += 1; continue
            detections = _rows_to_detections(rows, view)
            result = session.workflow.late_detections(frame_index, detections)
            if result is None:
                session.late['not_ready'] += 1; continue
            births, refreshes = result
            session.late['births'] += births; session.late['refreshes'] += refreshes; session.late['frames'] += 1
            applied.append({'frame_index': frame_index, 'lag_frames': session.frames-1-frame_index, 'detections': len(detections),
                            'births': births, 'refreshes': refreshes, 'detector_ms': round(ms, 1)})
        finally:
            if held:
                session.lock.release()
    return applied


def _fallback(request, detections, session, view):
    """A valid answer from this frame's detections alone, holding the camera."""
    rows = frame_rows(detections, view, CONFIG, session.prior)
    size = np.tile(view.source_size, 2)
    return {'request_id': request['request_id'], 'frame': request['frame'],
            'annotations': [{'object_id': r['object_id'], 'confidence': r['confidence'],
                             'bbox': (np.array(r['bbox_source_xyxy'])/size).tolist()} for r in rows][:500],
            'requested_view': None}


# --------------------------------------------------------------------------- #
# The request handler
# --------------------------------------------------------------------------- #

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
        raw_rows = []
        late = []
        if POOL is not None:
            late = _apply_late_results(session)
            meta = {k: v for k, v in req.items() if k != 'view'}
            meta['view'] = {k: v for k, v in req['view'].items() if k != 'image'}
            POOL.submit(req['sequence_id'], req['frame_index'], meta, base64.b64decode(request.view.image))
            detections, ran, detector_ms = [], False, 0.
        else:
            detections, ran, detector_ms = _detect_now(image, req, view, raw_rows)
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
                        'late': late, 'late_totals': dict(session.late),
                        'pool': {'outstanding': POOL.outstanding, 'pending': len(POOL.pending), 'dropped': POOL.dropped, 'ready': POOL.ready, 'alive': POOL.alive()} if POOL else None,
                        'response': [{'object_id': a.object_id, 'bbox': [round(v, 5) for v in a.bbox], 'confidence': round(float(a.confidence), 3)}
                                     for a in response.annotations][:500]})
        logger.info('frame %s L%s: %d detections, %d annotations, detector %.0f ms, tracking %.0f ms, total %.0f ms, late %d',
                    req['frame'], req['view']['resolution_level'], len(detections), len(response.annotations),
                    detector_ms, tracking_ms, total_ms, len(late))
    return response
