"""Per-stage latency of the live expert detector on real-sized views (L1, L2, L0) cut from a reference frame.

Reports, per view: upsample, shared proposer, per-class fine pose (max and sum), verifier, total.
This is the number to drive toward 300 ms.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame', type=Path, default=Path('data/drone/reference/helsinki/images/frame_000005.png'))
    p.add_argument('--gates', default='data/drone/expert-gates-v2.json')
    p.add_argument('--verifier', default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--levels', nargs='+', type=int, default=[1, 2, 0])
    p.add_argument('--repeat', type=int, default=2)
    a = p.parse_args()
    os.environ.setdefault('DRONE_EXPERT_SIFT', '0')
    from drone.experts.live_detector import ExpertDetector
    det = ExpertDetector(Path('.').resolve(), gates=a.gates, verifier=a.verifier, device='cuda:0', workers=a.workers, gpu='cuda:0', min_confidence=0.)
    full = cv2.imread(str(a.frame))
    regions = {1: (960, 0, 2880, 1080), 2: (2400, 540, 3360, 1080), 0: (0, 0, 3840, 2160)}
    for level in a.levels:
        x1, y1, x2, y2 = regions[level]
        view = cv2.resize(full[y1:y2, x1:x2], (960, 540), interpolation=cv2.INTER_AREA)
        req = {'view': {'resolution_level': level}, 'frame': 5}
        for rep in range(a.repeat):
            stages = {}
            t0 = time.perf_counter()
            factor = 1. / {0: .25, 1: .5, 2: 1.}[level]
            work = view if factor == 1. else cv2.resize(view, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
            stages['upsample'] = time.perf_counter() - t0
            t1 = time.perf_counter(); shared = det.shared.propose_all(work, 1., level); stages['proposer'] = time.perf_counter() - t1
            per = {}
            for name in det.experts:
                t2 = time.perf_counter(); det._run(name, work, 1., level, shared.get(name)); per[name] = time.perf_counter() - t2
            stages['fine_sum_sequential'] = sum(per.values()); stages['fine_slowest'] = sorted(per.items(), key=lambda kv: -kv[1])[:3]
            t3 = time.perf_counter(); rows = det(view, req); stages['end_to_end_threaded'] = time.perf_counter() - t3
            print(json.dumps(dict(level=level, repeat=rep, rows=len(rows), **{k: (round(v, 3) if isinstance(v, float) else [(n, round(s, 2)) for n, s in v]) for k, v in stages.items()})))


if __name__ == '__main__':
    main()
