"""Several Ultralytics checkpoints behind Oscar's detector hook, merged per class.

    DRONE_DETECTOR=elias.ensemble:build ELIAS_WEIGHTS="a.pt,b.pt" ELIAS_IMGSZ="1280,1280" python api.py

Each model runs on the delivered view at its own input size. Detections of the same class that overlap
(IoU >= 0.5) are merged into one box: coordinates averaged with confidence weights, confidence = the mean
over ALL models (a model that stays silent counts as 0), so agreement is rewarded and a lone detection is
kept at reduced confidence instead of being dropped (low-confidence boxes are nearly free under AP).
"""
from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout

import numpy as np


class Ensemble:
    name = 'elias-ensemble'

    def __init__(self, weights, sizes, device='cuda:0', conf=0.03, half=True):
        import cv2
        threads = cv2.getNumThreads()
        from ultralytics import YOLO
        with redirect_stdout(sys.stderr):
            self.models = [YOLO(w) for w in weights]
        cv2.setNumThreads(threads)
        self.sizes, self.device, self.conf, self.half = sizes, device, conf, half
        self(np.zeros((540, 960, 3), np.uint8), {})

    def __call__(self, image, request):
        per = []
        for m, size in zip(self.models, self.sizes):
            with redirect_stdout(sys.stderr):
                r = m.predict(image, imgsz=size, device=self.device, conf=self.conf, half=self.half, verbose=False)[0]
            per.append([(m.names[int(c)], np.array([x1, y1, x2, y2]), float(s)) for x1, y1, x2, y2, s, c in r.boxes.data.cpu().tolist()])
        rows, n = [], len(self.models)
        pool = sorted([(lab, box, s, k) for k, dets in enumerate(per) for lab, box, s in dets], key=lambda d: -d[2])
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
        return rows


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
    return Ensemble(weights, sizes, device=os.environ.get('DRONE_DEVICE', 'cuda:0'), conf=float(os.environ.get('ELIAS_CONF', '0.03')))
