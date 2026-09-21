"""Synthetic-trained YOLO detector for the live endpoint: ``DRONE_DETECTOR=synth_detector:build``.

The tiles the models were trained on are 256 px squares at L2 pixel scale; L0 and L1 tiles are the
same square degraded and upsampled. So a delivered L0 view is upsampled x4 and an L1 view x2 before
inference (``DRONE_SYNTH_SCALES``), and boxes are scaled back to delivered-image pixels. Several
checkpoints form an ensemble by concatenating their detections and applying class-wise NMS.

Environment:
  DRONE_SYNTH_WEIGHTS   comma-separated checkpoints (required)
  DRONE_SYNTH_SCALES    upsample factors for L0,L1,L2 (default 4,2,1)
  DRONE_SYNTH_CONF      confidence floor (default 0.25)
  DRONE_SYNTH_IOU       NMS IoU (default 0.6)
  DRONE_SYNTH_HALF      1 for fp16 (default 1)
  DRONE_SYNTH_MAXDET    detections per view (default 300)
  DRONE_DEVICE          torch device (default cuda:0)
  DRONE_SYNTH_CLASS_CONF JSON {class: floor} of per-class confidence floors (optional)
  DRONE_SYNTH_CLASS_MODELS JSON {class: [member indices]} routing: a class is taken only from the listed
                        ensemble members (0-based order of DRONE_SYNTH_WEIGHTS); unlisted classes from all
  DRONE_EXTENT_CLASS_POLICY JSON {class: detector|blend|prior}: per-class box extent policy overriding the
                        tracker's global DRONE_EXTENT_POLICY (installed by patching tracking.revisit at build time)
  DRONE_CANOPY_VETO     JSON {class: ring_threshold}: drop a detection of that class when the tree-canopy fraction of
                        the ring around its box (one box width/height on every side) exceeds the threshold. Ground
                        objects never stand in trees; on the validation scene 81% of phantom small_launcher boxes have
                        ring canopy > 0.4 and none of the true ones do.
"""
import json
import os
import sys
from contextlib import redirect_stdout

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from dtos import OBJECT_CLASSES


def canopy_mask(image_bgr):
    """Tree canopy in a delivered view: green (excess-green index), saturated and rough (local std of green)."""
    a = image_bgr.astype(np.float32)/255.
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    exg = 2*g-r-b
    v = a.max(-1)
    s = (v-a.min(-1))/np.maximum(v, 1e-6)
    m = cv2.blur(g, (9, 9))
    m2 = cv2.blur(g*g, (9, 9))
    std = np.sqrt(np.maximum(m2-m*m, 0.))
    mask = ((exg > .05) & (s > .18) & (std > .06)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    return cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)


def ring_canopy(mask, box):
    """Canopy fraction of the ring around a box (box grown by its own width and height on every side)."""
    x1, y1, x2, y2 = box
    w, h = max(1., x2-x1), max(1., y2-y1)
    X1, Y1 = int(max(0, x1-w)), int(max(0, y1-h))
    X2, Y2 = int(min(mask.shape[1], x2+w)), int(min(mask.shape[0], y2+h))
    outer = mask[Y1:Y2, X1:X2]
    if outer.size == 0:
        return 0.
    ix1, iy1 = int(max(0, x1))-X1, int(max(0, y1))-Y1
    ix2, iy2 = int(min(mask.shape[1], x2))-X1, int(min(mask.shape[0], y2))-Y1
    inner = outer[max(0, iy1):max(0, iy2), max(0, ix1):max(0, ix2)]
    n = outer.size-inner.size
    return float(outer.sum()-inner.sum())/n if n > 0 else 0.


class SynthDetector:
    name = 'synth'

    def __init__(self, weights, *, scales=(4., 2., 1.), device='cuda:0', conf=.25, iou=.6, half=True, max_det=300, class_conf=None,
                 class_models=None, canopy_veto=None):
        threads = cv2.getNumThreads()
        try:
            from ultralytics import YOLO
            with redirect_stdout(sys.stderr):
                self.models = [YOLO(str(w)) for w in weights]
        finally:
            cv2.setNumThreads(threads)
        self.names = None
        for m in self.models:
            names = [m.names[i] for i in range(len(m.names))]
            if set(names) != set(OBJECT_CLASSES):
                raise ValueError(f'Checkpoint classes {names} are not the competition classes')
            if self.names is None:
                self.names = names
            elif names != self.names:
                raise ValueError('Ensemble members must share the class order')
        self.scales = {0: float(scales[0]), 1: float(scales[1]), 2: float(scales[2])}
        self.device, self.conf, self.iou, self.half, self.max_det = device, float(conf), float(iou), bool(half), int(max_det)
        self.class_conf = {k: float(v) for k, v in (class_conf or {}).items()}
        self.class_models = {k: set(int(i) for i in v) for k, v in (class_models or {}).items()}
        self.canopy_veto = {k: float(v) for k, v in (canopy_veto or {}).items()}
        self.vetoed = 0
        for _ in range(3):  # warm up every input size a few times (kernel autotuning on the first calls)
            for level in (0, 1, 2):
                self({'__zeros__': True}, {'view': {'resolution_level': level}}, _image=np.zeros((540, 960, 3), np.uint8))

    def __call__(self, image, request, _image=None):
        image = _image if _image is not None else image
        level = int((request.get('view') or {}).get('resolution_level', 2))
        f = self.scales.get(level, 1.)
        mask = canopy_mask(image) if self.canopy_veto else None
        if f != 1:
            image = cv2.resize(image, (int(round(image.shape[1]*f)), int(round(image.shape[0]*f))), interpolation=cv2.INTER_LINEAR)
        boxes, scores, classes = [], [], []
        with redirect_stdout(sys.stderr):
            for i, m in enumerate(self.models):
                r = m.predict(image, imgsz=int(image.shape[1]), conf=self.conf, iou=self.iou, device=self.device, half=self.half,
                              max_det=self.max_det, verbose=False)[0]
                b, s, c = r.boxes.xyxy.cpu()/f, r.boxes.conf.cpu(), r.boxes.cls.cpu()
                if self.class_models:
                    keep = torch.tensor([i in self.class_models.get(self.names[int(ci)], {i}) for ci in c.tolist()], dtype=torch.bool)
                    b, s, c = b[keep], s[keep], c[keep]
                boxes.append(b)
                scores.append(s)
                classes.append(c)
        b, s, c = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
        if len(self.models) > 1 and len(b):
            keep = batched_nms(b.float(), s.float(), c, self.iou)
            b, s, c = b[keep], s[keep], c[keep]
        rows = []
        for bi, si, ci in zip(b.tolist(), s.tolist(), c.tolist()):
            label = self.names[int(ci)]
            if si < self.class_conf.get(label, 0.):
                continue
            if mask is not None and label in self.canopy_veto and ring_canopy(mask, bi) > self.canopy_veto[label]:
                self.vetoed += 1
                continue
            rows.append({'label': label, 'box': [float(v) for v in bi], 'confidence': float(si)})
        return rows


def install_class_extent_policy(environ=os.environ):
    """Per-class extent policy: wrap the placement helpers as bound inside tracking.revisit."""
    spec = json.loads(environ.get('DRONE_EXTENT_CLASS_POLICY', '{}'))
    if not spec:
        return
    import tracking.placement as pl
    import tracking.revisit as rv
    orig_norm, orig_partial = pl.normalize_extent, pl.place_partial

    def norm(label, box, prior, policy='detector', prior_weight=None, axes=(True, True)):
        return orig_norm(label, box, prior, spec.get(label, policy), None if label in spec else prior_weight, axes)

    def partial(label, box, view_region, source_size, prior, policy='detector', prior_weight=None, margin=1.):
        return orig_partial(label, box, view_region, source_size, prior, spec.get(label, policy), None if label in spec else prior_weight, margin)
    rv.normalize_extent, rv.place_partial = norm, partial
    print(f'per-class extent policy installed: {spec}', file=sys.stderr)


def build(environ=os.environ):
    install_class_extent_policy(environ)
    weights = [w for w in environ.get('DRONE_SYNTH_WEIGHTS', '').split(',') if w]
    if not weights:
        raise ValueError('DRONE_SYNTH_WEIGHTS is required')
    scales = [float(v) for v in environ.get('DRONE_SYNTH_SCALES', '4,2,1').split(',')]
    return SynthDetector(weights, scales=scales, device=environ.get('DRONE_DEVICE', 'cuda:0'),
                         conf=float(environ.get('DRONE_SYNTH_CONF', '0.25')), iou=float(environ.get('DRONE_SYNTH_IOU', '0.6')),
                         half=environ.get('DRONE_SYNTH_HALF', '1') == '1', max_det=int(environ.get('DRONE_SYNTH_MAXDET', '300')),
                         class_conf=json.loads(environ.get('DRONE_SYNTH_CLASS_CONF', '{}')),
                         class_models=json.loads(environ.get('DRONE_SYNTH_CLASS_MODELS', '{}')),
                         canopy_veto=json.loads(environ.get('DRONE_CANOPY_VETO', '{}')))
