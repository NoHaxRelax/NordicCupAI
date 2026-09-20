#!/usr/bin/env python3
"""Test several box sizes for one class in a single attempt, and read which one hits off the score.

Probing sizes one at a time costs an attempt each and the organiser only runs one at a time. Instead every
candidate size is answered on the same track in its own confidence band, so the bands are matched in strict
order: if the k-th size is the first that hits, the (k-1) sizes above it are false positives ranked ahead of
every true positive, and the interpolated precision at the end is about r/(k-1+r) for r = frames answered over
truth frames. The score therefore lands on a ladder with one rung per size, and 0 if none of them hits.

Sizes are given at the track's midpoint and grown along it with the flight-map perspective scale, since a
fixed size can only fit the middle of a track.

    python3 size_ladder.py --cls medium_launcher --sizes 40x44,46x50,34x38,52x56,28x32 --truth 98 --tag mlladder
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground
from persp_probe import cluster, scale_at

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--cls', required=True)
    ap.add_argument('--sizes', required=True, help='comma list WxH, most likely first')
    ap.add_argument('--instance', type=int, default=None, help='1-based; default every instance')
    ap.add_argument('--truth', type=float, required=True, help='best estimate of the class truth frame count')
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    inst = [k for k in cluster(rows, a.radius) if k['n'] >= 5]
    if a.instance:
        inst = [inst[a.instance - 1]]
    sizes = a.sizes.split(',')
    n = len(sizes)
    pbf = defaultdict(list)
    for rank, spec in enumerate(sizes):
        bw, bh = (float(v) for v in spec.split('x'))
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        for k in inst:
            fs = sorted(f for f, _ in k['rows'])
            ref = fs[len(fs) // 2]
            s0 = scale_at(G, ref, (k['x'], k['y']))
            for f, b in k['rows']:
                r = (scale_at(G, f, (k['x'], k['y'])) / s0) ** a.gamma
                x0, y0, x1, y1 = b['bbox']
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = bw * r / W, bh * r / H
                nb = [max(0.0, cx - w / 2) if x0 > 0 else 0.0, max(0.0, cy - h / 2) if y0 > 0 else 0.0,
                      min(1.0, cx + w / 2) if x1 < 1 else 1.0, min(1.0, cy + h / 2) if y1 < 1 else 1.0]
                if nb[2] - nb[0] > 0.001 and nb[3] - nb[1] > 0.001:
                    pbf[str(f)].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in nb],
                                        'confidence': conf})
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    m = len({f for k in inst for f, _ in k['rows']})
    rec = m / a.truth
    print(f'{name}: {n} sizes over {m} frames of {len(inst)} instance(s), '
          f'{sum(len(v) for v in pbf.values())} boxes; truth assumed {a.truth:g}')
    print('  rank  size      confidence   board score if this is the first size that hits')
    for rank, spec in enumerate(sizes):
        print(f'  {rank + 1:4d}  {spec:8s}  {round(0.95 - 0.9 * rank / max(1, n - 1), 4):8.4f}   '
              f'{rec * rec / (rank + rec) / 13:.5f}')
    print(f'  none                          {0.0:.5f}')


if __name__ == '__main__':
    main()
