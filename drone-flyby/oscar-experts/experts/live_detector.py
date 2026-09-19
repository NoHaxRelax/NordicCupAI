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
  DRONE_EXPERT_NATIVE    1 (default): upsample the delivered view to native pixel scale before the experts
  DRONE_EXPERT_SIFT      0 (default): drop the SIFT comparison branch in live mode (slow, never fused)
  DRONE_EXPERT_PROCS     worker processes for the per-class stage (default 0 = thread pool); identical rows, no GIL
  DRONE_EXPERT_ROUTING   per-level class -> upsampling factor of the delivered view, JSON inline or file (overrides LEVEL_FACTORS per class)
  DRONE_EXPERT_GPU       cuda device for the shared FFT proposer (one scene transform per view for all classes); unset = CPU proposers
"""
import json
import os
import sys
import time

import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCALE_FOR_LEVEL = {0: .25, 1: .5, 2: 1.}


class ExpertDetector:
    name = 'experts'

    def __init__(self, project, bank=None, classes=None, verifier=None, device='cpu', workers=8, min_confidence=0., log=None, gates=None, native=True, gpu=None):
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
        if os.environ.get('DRONE_EXPERT_SIFT', '0') != '1':
            for expert, _ in self.experts.values():
                if hasattr(expert, 'sift'):
                    expert.sift = None
        self.shared = None
        if gpu:
            from drone.experts.gpu_proposer import SharedProposer
            self.shared = SharedProposer({n: e for n, (e, _) in self.experts.items()}, device=gpu)
        self.pool = ThreadPoolExecutor(max_workers=int(workers))
        # DRONE_EXPERT_PROCS=N: run the per-class stage in N worker processes (the experts' Python holds the GIL, so the
        # thread pool is effectively serial). Same code and inputs in each worker: identical rows.
        self.procs = None
        if int(os.environ.get('DRONE_EXPERT_PROCS', '0')) > 0:
            from drone.experts.proc_pool import ExpertProcessPool
            slow_first = ['large_launcher', 'hangar', 'medium_plane', 'ta-ta', 'small_plane', 'mine_roller', 'tank', 'jammer', 'small_tower',
                          'spacecraft', 'helicopter', 'jet_plane', 'large_tower', 'condor', 'small_launcher', 'medium_launcher']
            self.procs = ExpertProcessPool(int(os.environ['DRONE_EXPERT_PROCS']), project, bank, gates, list(self.experts),
                                           sift=os.environ.get('DRONE_EXPERT_SIFT', '0') == '1', cost_order=slow_first)
        self.min_confidence = float(min_confidence)
        self.native = bool(native)
        # DRONE_EXPERT_ROUTING: JSON (inline or a file path) {level: {class or "default": upsampling factor of the delivered view}}
        spec = os.environ.get('DRONE_EXPERT_ROUTING', '')
        if spec and not spec.lstrip().startswith('{'):
            spec = Path(spec).read_text()
        self.routing = {int(k): v for k, v in (json.loads(spec) if spec else {}).items()}
        raw = os.environ.get('DRONE_EXPERT_LEVEL_FACTORS', '')
        self.level_factors = {i: float(v) for i, v in enumerate(raw.split(','))} if raw else {}
        self.log = Path(log) if log else None
        self(np.zeros((540, 960, 3), np.uint8), {'view': {'resolution_level': 1}})  # warm-up

    def _run(self, name, image, scale, zoom, proposals=None):
        expert, family = self.experts[name]
        try:
            out = expert.detect(image, scale, zoom, explain=False, proposals=proposals) if proposals is not None else expert.detect(image, scale, zoom, explain=False)
            rows = out[family] if isinstance(out, dict) else out[0]
            return [dict(r, expert=name) for r in rows]
        except Exception as error:
            print(f'expert {name} failed: {error}', file=sys.stderr)
            return []

    def __call__(self, image, request):
        started = time.perf_counter()
        level = int((request.get('view') or {}).get('resolution_level', 1))
        scale = SCALE_FOR_LEVEL.get(level, .5)
        # The experts and their gates were fitted on tiles at native pixel scale with zoom-specific blur.
        # Upsample the delivered view to that scale (x2 at L1, x4 at L0) so every feature transfers as fitted.
        # DRONE_EXPERT_LEVEL_FACTORS overrides the upsample factor per level: "2,2,1" runs L0 at half the native
        # scale (94% of native L0 recall on training tiles at a quarter of the pixels; pass 11d2).
        default = self.level_factors.get(level, 1. / scale if self.native else 1.)
        # DRONE_EXPERT_ROUTING can send single classes to another factor at a level (e.g. native L0 only for ta-ta and the
        # planes): one scene and one proposer pass per factor, restricted to its classes; unset = the level factor for all.
        route = self.routing.get(level, {})
        factors = {name: float(route.get(name, route.get('default', default))) for name in self.experts}
        groups = {}
        for name, f in factors.items():
            groups.setdefault(f, []).append(name)
        scenes = []
        for f, names in groups.items():
            work = image if f == 1. else cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            scenes.append((names, work, scale * f))  # scale * f: pixels per source pixel the experts see; 1 = native
        multi = len(groups) > 1
        if self.procs is not None:
            # stream: each class's expert starts in its worker as soon as its proposals are done (GPU and CPU overlap)
            def produce(callback):
                if self.shared is not None:
                    for names, work, s_run in scenes:
                        self.shared.propose_all(work, s_run, level, classes=names if multi else None, on_done=callback)
            by_class = self.procs.run_stream(scenes, level, produce, set(self.shared.experts) if self.shared is not None else set())
        else:
            jobs = [(names, work, s_run, self.shared.propose_all(work, s_run, level, classes=names if multi else None) if self.shared is not None else {})
                    for names, work, s_run in scenes]
            futures = {name: self.pool.submit(self._run, name, work, s_run, level, shared.get(name)) for names, work, s_run, shared in jobs for name in names}
            by_class = {name: f.result() for name, f in futures.items()}
        rows = []
        for name in self.experts:  # same row order as before: class order
            f = factors[name]
            for r in by_class.get(name, []):
                if f != 1.:
                    r['bbox'] = [float(v) / f for v in r['bbox']]
                rows.append(r)
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
                          int(env.get('DRONE_EXPERT_WORKERS', '8')), float(env.get('DRONE_EXPERT_MIN_CONF', '0')), env.get('DRONE_EXPERT_LOG'), env.get('DRONE_EXPERT_GATES'), env.get('DRONE_EXPERT_NATIVE', '1') == '1', env.get('DRONE_EXPERT_GPU'))
