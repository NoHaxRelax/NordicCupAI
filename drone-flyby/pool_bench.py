"""Throughput of the asynchronous detector pool on real views at the frame clock.

    DRONE_DETECTOR=fixed_assets DRONE_BUNDLE=... DRONE_PROJECT=... DRONE_DEVICE=0 \
    DRONE_BUNDLE_FAST=1 DRONE_BUNDLE_WORKERS=4 DRONE_WORKERS=6 DRONE_BACKLOG=6 DRONE_WORKER_THREADS=8 \
    python pool_bench.py --frames /path/to/reconstructed-validation --count 90 --rate 3

Feeds delivered-size views (native L2 crops of the top row, an L1 quarter and
the L0 overview, in the mix the L2 sweep produces) to the same DetectorPool the
endpoint uses, at ``--rate`` views per second, and reports what came back:
completed views, dropped views, detector time per view and the submit-to-result
latency. The latency in frames is what a late detection costs the tracker.
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from detector_pool import DetectorPool

MIX = ['L2', 'L2', 'L2', 'L0', 'L2', 'L2', 'L2', 'L1']


def views_for(frame_path):
    source = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    for x in (0, 960, 1920, 2880):
        yield 'L2', [x, 0, x+960, 540], source[0:540, x:x+960]
    yield 'L1', [0, 0, 1920, 1080], cv2.resize(source[0:1080, 0:1920], (960, 540), interpolation=cv2.INTER_AREA)
    yield 'L0', [0, 0, 3840, 2160], cv2.resize(source, (960, 540), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--frame-numbers', type=int, nargs='*', default=[40, 100, 140, 200])
    p.add_argument('--count', type=int, default=90)
    p.add_argument('--rate', type=float, default=3.)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    by_level = {'L2': [], 'L1': [], 'L0': []}
    for n in a.frame_numbers:
        for level, region, image in views_for(a.frames/f'frame_{n:06d}.png'):
            ok, png = cv2.imencode('.png', image)
            by_level[level].append((n, region, png.tobytes()))
    workers = int(os.environ.get('DRONE_WORKERS', '4')); backlog = int(os.environ.get('DRONE_BACKLOG', str(workers)))
    loading = time.perf_counter()
    pool = DetectorPool(workers, backlog)
    print(json.dumps({'workers': workers, 'backlog': backlog, 'ready': pool.ready, 'load_s': round(time.perf_counter()-loading, 1)}), flush=True)
    submitted = {}
    done = []
    started = time.perf_counter()
    for i in range(a.count):
        level = MIX[i % len(MIX)]
        n, region, png = by_level[level][i % len(by_level[level])]
        meta = {'sequence_id': 'bench', 'frame': n, 'frame_index': i, 'request_id': f'r{i}',
                'view': {'source_region_xyxy': region, 'width': 960, 'height': 540, 'resolution_level': int(level[1])}}
        target = started+i/a.rate
        while time.perf_counter() < target:
            for item in pool.drain():
                done.append((time.perf_counter(),)+item)
            time.sleep(.005)
        submitted[i] = (level, time.perf_counter())
        pool.submit('bench', i, meta, png)
        for item in pool.drain():
            done.append((time.perf_counter(),)+item)
    deadline = time.perf_counter()+120
    while len(done)+pool.dropped < a.count and time.perf_counter() < deadline:
        for item in pool.drain():
            done.append((time.perf_counter(),)+item)
        time.sleep(.02)
    elapsed = time.perf_counter()-started
    rows = []
    for item in done:
        if len(item) == 5:  # drained before the first submit (cannot happen, kept for safety)
            continue
        finished, sequence_id, frame_index, det_rows, error, ms = item
        level, sent = submitted[frame_index]
        rows.append({'frame_index': frame_index, 'level': level, 'detector_ms': round(ms, 1),
                     'latency_ms': round((finished-sent)*1000, 1), 'detections': len(det_rows), 'error': error})
    summary = {'submitted': a.count, 'completed': len(rows), 'dropped': pool.dropped, 'errors': sum(1 for r in rows if r['error']),
               'elapsed_s': round(elapsed, 1), 'views_per_s': round(len(rows)/elapsed, 2), 'rate': a.rate,
               'latency_ms': {'p50': float(np.median([r['latency_ms'] for r in rows])) if rows else None,
                              'p90': float(np.percentile([r['latency_ms'] for r in rows], 90)) if rows else None,
                              'max': max((r['latency_ms'] for r in rows), default=None)},
               'detector_ms_by_level': {lvl: float(np.median([r['detector_ms'] for r in rows if r['level'] == lvl]))
                                        for lvl in ('L2', 'L1', 'L0') if any(r['level'] == lvl for r in rows)},
               'first_error': next((r['error'] for r in rows if r['error']), None)}
    print(json.dumps(summary), flush=True)
    if a.output:
        a.output.write_text(json.dumps({'summary': summary, 'rows': rows, 'env': {k: v for k, v in os.environ.items() if k.startswith('DRONE_')}}, indent=1)+'\n')
    pool.close()


if __name__ == '__main__':
    main()
