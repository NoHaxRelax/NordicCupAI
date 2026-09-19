"""Several Ultralytics checkpoints behind Oscar's detector hook, merged per class.

    DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="a.pt,b.pt" ELIAS_IMGSZ="1280,1280" python api.py

Each model runs on the delivered view at its own input size. Detections of the same class that overlap
(IoU >= 0.5) are merged into one box: coordinates averaged with confidence weights, confidence = the mean
over ALL models (a model that stays silent counts as 0), so agreement is rewarded and a lone detection is
kept at reduced confidence instead of being dropped (low-confidence boxes are nearly free under AP).

Per-class routing (ELIAS_ROUTE, JSON like {"small_tower": 1, "ta-ta": 1}): a routed class is taken from that ONE
model only, at that model's own confidence, so the model that measured best on a class answers for it; classes
without a route are merged across models as above. Measure each model per class on the portal, route, re-measure.

An ELIAS_WEIGHTS entry written module:factory is another detector hook (for example a teammate's own detector),
built with no arguments and called as detector(image, request) -> rows of label, box, confidence, so a foreign
pipeline can keep its detector for most classes and take ours for the classes where ours measured better:

    DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="their.module:build,elias/release/both_m1280.pt,elias/release/F5_fixed_m1280.pt"
    ELIAS_ROUTE='{"helicopter": 1, "ta-ta": 2, "<every other class>": 0}'
"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout

import numpy as np


class Ensemble:
    name = 'elias-ensemble'

    def __init__(self, weights, sizes, device='cuda:0', conf=0.03, half=True, route=None, context=1.0, route_l2=None, augment=False):
        self.route = dict(route or {}); self.context = float(context); self.augment = bool(augment)   # flip and scale test-time augmentation
        self.route_l2 = dict(route_l2) if route_l2 else None   # a separate table for native (L2) views
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
        import cv2
        threads = cv2.getNumThreads()
        from ultralytics import YOLO
        with redirect_stdout(sys.stderr):
            # an entry written module:factory is any other detector hook (a teammate's), built and called like ours
            self.models = [_factory(w)() if ':' in w and not os.path.exists(w) else YOLO(w) for w in weights]
        cv2.setNumThreads(threads)
        self.sizes, self.device, self.conf, self.half = sizes, device, conf, half
        self(np.zeros((540, 960, 3), np.uint8), {})

    def _run(self, m, size, image, request):
        if not hasattr(m, 'predict'):                          # a hook detector: rows of label, box, confidence
            return [(r['label'], np.array(r['box'], float), float(r['confidence'])) for r in (m(image, request) or [])]
        with redirect_stdout(sys.stderr):
            r = m.predict(image, imgsz=size, device=self.device, conf=self.conf, half=self.half, verbose=False, augment=self.augment)[0]
        return [(m.names[int(c)], np.array([x1, y1, x2, y2]), float(s)) for x1, y1, x2, y2, s, c in r.boxes.data.cpu().tolist()]

    def __call__(self, image, request):
        per = [self._run(m, size, image, request) for m, size in zip(self.models, self.sizes)]
        rows, n = [], len(self.models)
        route = self.route
        try:
            if self.route_l2 is not None and int((request or {}).get('view', {}).get('resolution_level', 0)) == 2:
                route = self.route_l2
        except (TypeError, ValueError, AttributeError):
            pass
        for lab, k in route.items():                           # routed classes: one model answers, its own confidence
            rows += [{'label': l, 'box': b.tolist(), 'confidence': s} for l, b, s in per[k] if l == lab]
        pool = sorted([(lab, box, s, k) for k, dets in enumerate(per) for lab, box, s in dets if lab not in route], key=lambda d: -d[2])
        used = [False]*len(pool)
        for i, (lab, box, s, k) in enumerate(pool):
            if used[i]:
                continue
            used[i] = True; group = [(box, s, k)]
            for j in range(i+1, len(pool)):
                if not used[j] and pool[j][0] == lab and pool[j][3] not in {g[2] for g in group} and _iou(box, pool[j][1]) >= 0.5:
                    used[j] = True; group.append((pool[j][1], pool[j][2], pool[j][3]))
            w = np.array([g[1] for g in group]); merged = (np.stack([g[0] for g in group])*w[:, None]).sum(0)/w.sum()
            rows.append({'label': lab, 'box': merged.tolist(), 'confidence': float(w.sum()/n)})
        if self.context < 1.0 and rows:
            # context prior: objects never stand on water or in tree cover, so such boxes keep their place but at
            # a lower confidence (never deleted: low-confidence extras are nearly free under AP)
            from terrain import implausible
            for r in rows:
                if implausible(image, r['box'], grow=1.0):
                    r['confidence'] = float(r['confidence'])*self.context
        return rows


def _factory(spec):
    import importlib
    module, name = spec.rsplit(':', 1)
    return getattr(importlib.import_module(module), name)


def _iou(a, b):
    ix = max(0., min(a[2], b[2])-max(a[0], b[0])); iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def build():
    weights = [w for w in os.environ.get('ELIAS_WEIGHTS', '').split(',') if w]
    if not weights:
        raise ValueError('elias.ensemble:build needs ELIAS_WEIGHTS="a.pt,b.pt"')
    sizes = [int(v) for v in os.environ.get('ELIAS_IMGSZ', '1280').split(',')]
    sizes = (sizes*len(weights))[:len(weights)]
    import json
    route = json.loads(os.environ.get('ELIAS_ROUTE', '{}') or '{}')
    route_l2 = json.loads(os.environ.get('ELIAS_ROUTE_L2', '') or 'null')
    return Ensemble(weights, sizes, device=os.environ.get('DRONE_DEVICE', 'cuda:0'), conf=float(os.environ.get('ELIAS_CONF', '0.03')), route=route,
                    context=float(os.environ.get('ELIAS_CONTEXT', '1.0') or 1.0), route_l2=route_l2,
                    augment=os.environ.get('ELIAS_AUGMENT', '0') == '1')
