"""Live-endpoint detector factory: every class expert in parallel, optional verifier on top.

Contract (team repo drone-flyby/detectors.py): detector(image_bgr, request) -> rows of
{'label', 'box' [x1,y1,x2,y2] in delivered-image pixels, 'confidence' in [0,1]}, organizer-style
extents, edge boxes kept. Configure through environment variables:
  DRONE_EXPERT_BANK      expert sprite bank (default data/drone/expert-bank-20260918-v1 under DRONE_PROJECT)
  DRONE_EXPERT_CLASSES   comma list, default all 16
  DRONE_EXPERT_VERIFIER  optional verifier checkpoint; then confidence = verifier class probability
  DRONE_EXPERT_DEVICE    verifier device (cpu or cuda:0)
  DRONE_EXPERT_WORKERS   thread pool size for the experts (default 8)
  DRONE_EXPERT_MIN_CONF  drop rows below this confidence before handing them to the tracker (default 0)
  DRONE_EXPERT_GATES     fitted per-class logistic gates (fit_gates.py); confidence = gate probability
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCALE_FOR_LEVEL = {0: .25, 1: .5, 2: 1.}


class ExpertDetector:
    name = 'experts'

    def __init__(self, project, bank=None, classes=None, verifier=None, device='cpu', workers=8, min_confidence=0., log=None, gates=None):
        project = Path(project)
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from drone.experts.registry import make_expert, families, CLASSES, load_gates
        gate_table = load_gates(gates) if gates else None
        bank = Path(bank or project / 'data/drone/expert-bank-20260918-v1')
        self.classes = list(classes or CLASSES)
        self.experts = {}
        for name in self.classes:
            try:
                self.experts[name] = (make_expert(name, bank, gate_table), families(name)[0])
            except Exception as error:
                print(f'expert {name} unavailable: {error}', file=sys.stderr)
        self.verifier = None
        if verifier:
            from drone.experts.verifier import Verifier
            self.verifier = Verifier(verifier, device)
        self.pool = ThreadPoolExecutor(max_workers=int(workers))
        self.min_confidence = float(min_confidence)
        self.log = Path(log) if log else None
        self(np.zeros((540, 960, 3), np.uint8), {'view': {'resolution_level': 1}})  # warm-up

    def _run(self, name, image, scale, zoom):
        expert, family = self.experts[name]
        try:
            out = expert.detect(image, scale, zoom, explain=False)
            rows = out[family] if isinstance(out, dict) else out[0]
            return [dict(r, expert=name) for r in rows]
        except Exception as error:
            print(f'expert {name} failed: {error}', file=sys.stderr)
            return []

    def __call__(self, image, request):
        started = time.perf_counter()
        level = int((request.get('view') or {}).get('resolution_level', 1))
        scale = SCALE_FOR_LEVEL.get(level, .5)
        futures = [self.pool.submit(self._run, name, image, scale, level) for name in self.experts]
        rows = [r for f in futures for r in f.result()]
        if self.verifier is not None and rows:
            rows = self.verifier.annotate(image, rows)
            for r in rows:
                r['confidence'] = r['verifier_probability']
        else:
            for r in rows:
                r['confidence'] = float(r.get('score', 0.))
        out = [{'label': r['class'], 'box': [float(v) for v in r['bbox']], 'confidence': float(np.clip(r['confidence'], 0, 1)), 'expert': r['expert']}
               for r in rows if r['confidence'] >= self.min_confidence]
        if self.log:
            with self.log.open('a') as f:
                f.write(json.dumps(dict(frame=request.get('frame'), level=level, ms=round((time.perf_counter() - started) * 1000, 1), rows=len(out))) + '\n')
        return out


def factory():
    env = os.environ
    return ExpertDetector(env.get('DRONE_PROJECT', '/workspace/experts/project'), env.get('DRONE_EXPERT_BANK'),
                          [c for c in env.get('DRONE_EXPERT_CLASSES', '').split(',') if c] or None,
                          env.get('DRONE_EXPERT_VERIFIER'), env.get('DRONE_EXPERT_DEVICE', 'cpu'),
                          int(env.get('DRONE_EXPERT_WORKERS', '8')), float(env.get('DRONE_EXPERT_MIN_CONF', '0')), env.get('DRONE_EXPERT_LOG'), env.get('DRONE_EXPERT_GATES'))
