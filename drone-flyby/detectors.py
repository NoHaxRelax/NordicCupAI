"""Detector adapters for the live endpoint.

A detector is any callable ``detector(image_bgr, request) -> rows`` where
``image_bgr`` is the decoded 960x540 view and ``request`` the request as a
dict. Each row is ``{'label': <competition class>, 'box': [x1, y1, x2, y2] in
delivered-image pixels, 'confidence': 0..1}``. Return boxes in the organizer's
box convention (the whole asset footprint, not the tight silhouette) and keep
boxes that touch the image edge: the tracker uses them as partial views.

Selection is by environment variable ``DRONE_DETECTOR``:

* ``ultralytics`` (default when ``DRONE_WEIGHTS`` is set): a local Ultralytics
  checkpoint with exactly the 16 competition class names.
* ``none``: no detector; the endpoint answers with tracks only (useful to test
  the camera and transport).
* ``fixed_assets``: the preparation project's frozen fixed-asset bundle
  (``DRONE_BUNDLE`` manifest, ``DRONE_PROJECT`` repo root). Seconds per view.
* ``oracle``: reads the organizer annotations of a local scene. Local plumbing
  tests only; it cannot run against the evaluation service.
* ``package.module:factory``: a custom factory returning a detector callable.
"""
import importlib
import json
import logging
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from dtos import OBJECT_CLASSES

logger = logging.getLogger(__name__)


class NullDetector:
    name = 'none'

    def __call__(self, image, request):
        return []


class UltralyticsDetector:
    name = 'ultralytics'

    def __init__(self, weights, *, device='cpu', image_size=960, confidence=.25, half=False):
        path = Path(weights)
        if not path.is_file():
            raise ValueError(f'Weights must be an existing local file, got {path}')
        # Importing Ultralytics resets OpenCV to a single thread process-wide,
        # which made the tracker's SIFT calibration four times slower. Keep
        # the caller's thread budget for the tracker.
        import cv2
        threads = cv2.getNumThreads()
        try:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError('DRONE_DETECTOR=ultralytics needs torch and ultralytics installed') from exc
            with redirect_stdout(sys.stderr):
                self.model = YOLO(str(path))
        finally:
            cv2.setNumThreads(threads)
        names = set(self.model.names.values())
        if names != set(OBJECT_CLASSES):
            raise ValueError(f'Checkpoint classes {sorted(names)} are not the competition classes')
        self.device, self.image_size, self.confidence, self.half = device, int(image_size), float(confidence), half
        # Warm up once so the first scored frame does not pay for lazy initialization.
        self(np.zeros((540, 960, 3), np.uint8), {})

    def __call__(self, image, request):
        with redirect_stdout(sys.stderr):
            result = self.model.predict(image, imgsz=self.image_size, device=self.device, conf=self.confidence,
                                        half=self.half, verbose=False)[0]
        rows = []
        for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
            rows.append({'label': self.model.names[int(cls)], 'box': [x1, y1, x2, y2], 'confidence': float(score)})
        return rows


class FixedAssetDetector:
    """The frozen fixed-asset bundle (geometry, pixel matching and CNNs) from the preparation project.

    Slow (seconds per view) but the strongest recognizer available offline.
    Needs ``DRONE_PROJECT`` on sys.path for its ``drone`` package, torch and
    ultralytics. Rows carry the recognition ``family`` for diagnostics.
    """
    name = 'fixed_assets'

    def __init__(self, bundle, project, *, device='cpu', min_confidence=0., family_min=None, family_levels=None):
        project = Path(project)
        if not (project/'drone').is_dir():
            raise ValueError(f'DRONE_PROJECT must contain the drone package, got {project}')
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        import cv2
        threads = cv2.getNumThreads()
        try:
            from drone.scratch_objects.bundle import load_bundle
            with redirect_stdout(sys.stderr):
                self.detector = load_bundle(bundle, device=device)
        finally:
            cv2.setNumThreads(threads)
        self.min_confidence = float(min_confidence)
        self.family_min = dict(family_min or {})
        # Recognition families that are only trustworthy at native or near-native
        # zoom can be restricted to those levels: {"pose_pixels": [2], ...}.
        self.family_levels = {k: set(int(v) for v in vals) for k, vals in (family_levels or {}).items()}

    def __call__(self, image, request):
        x1, y1, x2, y2 = request['view']['source_region_xyxy']
        scale = request['view']['width']/(x2-x1)  # delivered pixels per source pixel
        level = int(request['view'].get('resolution_level', 2))
        with redirect_stdout(sys.stderr):
            rows = self.detector.detect(np.ascontiguousarray(image), scale)
        out = []
        for r in rows:
            family = r.get('family') or r.get('method')
            if family in self.family_levels and level not in self.family_levels[family]:
                continue
            floor = max(self.min_confidence, self.family_min.get(family, 0.))
            if r['score'] < floor:
                continue
            out.append({'label': r['class'], 'box': list(map(float, r['bbox'])), 'confidence': float(min(1., r['score'])),
                        'family': family})
        return out


class OracleDetector:
    """Organizer boxes of a local scene as detections. For local tests only."""
    name = 'oracle'

    def __init__(self, scene_directory):
        self.directory = Path(scene_directory)/'annotations'
        if not self.directory.is_dir():
            raise ValueError(f'No annotations under {self.directory}')
        logger.warning('Oracle detector active: answers come from local ground truth, not a model')

    def __call__(self, image, request):
        frame = int(request['frame'])
        path = self.directory/f'frame_{frame:06d}.json'
        if not path.is_file():
            return []
        x1, y1, x2, y2 = request['view']['source_region_xyxy']
        width, height = request['view']['width'], request['view']['height']
        sx, sy = (x2-x1)/width, (y2-y1)/height
        rows = []
        for a in json.loads(path.read_text())['annotations']:
            b = np.array(a['bbox'], float)
            visible = [max(b[0], x1), max(b[1], y1), min(b[2], x2), min(b[3], y2)]
            if visible[2]-visible[0] <= 1 or visible[3]-visible[1] <= 1:
                continue
            rows.append({'label': a['object_id'], 'confidence': .9,
                         'box': [(visible[0]-x1)/sx, (visible[1]-y1)/sy, (visible[2]-x1)/sx, (visible[3]-y1)/sy]})
        return rows


def build_detector(environ=os.environ):
    kind = environ.get('DRONE_DETECTOR', 'ultralytics' if environ.get('DRONE_WEIGHTS') else 'none').strip()
    if kind == 'none':
        return NullDetector()
    if kind == 'ultralytics':
        weights = environ.get('DRONE_WEIGHTS')
        if not weights:
            raise ValueError('DRONE_DETECTOR=ultralytics needs DRONE_WEIGHTS')
        return UltralyticsDetector(weights, device=environ.get('DRONE_DEVICE', 'cpu'),
                                   image_size=int(environ.get('DRONE_IMGSZ', '960')),
                                   confidence=float(environ.get('DRONE_CONF', '0.25')),
                                   half=environ.get('DRONE_HALF', '0') == '1')
    if kind == 'fixed_assets':
        bundle, project = environ.get('DRONE_BUNDLE'), environ.get('DRONE_PROJECT')
        if not bundle or not project:
            raise ValueError('DRONE_DETECTOR=fixed_assets needs DRONE_BUNDLE (manifest.json) and DRONE_PROJECT (repo root)')
        family_min = json.loads(environ.get('DRONE_FAMILY_MIN', '{}'))
        family_levels = json.loads(environ.get('DRONE_FAMILY_LEVELS', '{}'))
        return FixedAssetDetector(bundle, project, device=environ.get('DRONE_DEVICE', 'cpu'),
                                  min_confidence=float(environ.get('DRONE_CONF', '0')), family_min=family_min,
                                  family_levels=family_levels)
    if kind == 'oracle':
        return OracleDetector(environ.get('DRONE_ORACLE_SCENE', str(Path(__file__).resolve().parent/'src/helsinki')))
    if ':' in kind:
        module, attribute = kind.split(':', 1)
        factory = getattr(importlib.import_module(module), attribute)
        detector = factory()
        if not callable(detector):
            raise TypeError(f'{kind} did not return a callable detector')
        return detector
    raise ValueError(f'Unknown DRONE_DETECTOR {kind!r}')
