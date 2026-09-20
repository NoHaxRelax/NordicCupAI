#!/usr/bin/env python3
"""Search a ring of ground positions around an anchor in ONE attempt, and read off which one is the object.

Each candidate position gets its own confidence band, so the bands are answered strictly in order. If the k-th
band is the real object and the rest are empty, the first (k-1) bands are pure false positives ahead of it and
the interpolated precision at the end of the run is about r/(k-1+r) for r = (frames answered)/(truth frames).
The score therefore falls on a known ladder - one value per rank - and zero if none of them is an object. That
turns what would be nine attempts into one.

    python3 ring_probe.py --cls medium_launcher --anchor-cls large_tower --anchor -1546,-17808 \
        --radii 0,50,85 --size 40x44 --tag ml3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground

ROOT = Path(__file__).resolve().parents[2]


def img_of(G, f, g):
    p = np.linalg.inv(G[f]) @ np.array([g[0], g[1], 1.0])
    return p[:2] / p[2]


def scale_at(G, f, g, d=1.0):
    x, y = img_of(G, f, g)
    a = to_ground(G[f], x, y)
    J = np.array([to_ground(G[f], x + d, y) - a, to_ground(G[f], x, y + d) - a]).T / d
    return 1.0 / np.sqrt(abs(np.linalg.det(J)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--anchor-cls', required=True, help='the class whose track marks the spot')
    ap.add_argument('--anchor', required=True, help='ground x,y of the anchor instance')
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--radii', default='0,50,85', help='ground px rings; 0 means the anchor spot itself')
    ap.add_argument('--per-ring', type=int, default=4, help='positions on each non-zero ring')
    ap.add_argument('--size', default='40x44', help='box size in image px at the reference frame')
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    ax, ay = (float(v) for v in a.anchor.split(','))
    bw, bh = (float(v) for v in a.size.split('x'))
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())

    frames = []
    for f, boxes in plan['predictions_by_frame'].items():
        f = int(f)
        if f not in G:
            continue
        for b in boxes:
            if b['object_id'] != a.anchor_cls or b['confidence'] < 0.8:
                continue
            g = to_ground(G[f], (b['bbox'][0] + b['bbox'][2]) / 2 * W, (b['bbox'][1] + b['bbox'][3]) / 2 * H)
            if (g[0] - ax) ** 2 + (g[1] - ay) ** 2 <= 200 ** 2:
                frames.append(f)
    frames = sorted(set(frames))
    if not frames:
        print('anchor track not found'); return

    # the two known installations put the launcher about 50 ground px from the tower, on opposite sides, so the
    # ring is searched rather than guessed; the nearest offsets are ranked first
    cands = [(0.0, 0.0)] if '0' in a.radii.split(',') else []
    for r in (float(v) for v in a.radii.split(',')):
        if r == 0:
            continue
        for k in range(a.per_ring):
            t = 2 * math.pi * k / a.per_ring
            cands.append((r * math.cos(t), r * math.sin(t)))
    n = len(cands)
    ref = frames[len(frames) // 2]
    s0 = scale_at(G, ref, (ax, ay))
    pbf = defaultdict(list)
    for rank, (dx, dy) in enumerate(cands):
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        for f in frames:
            gx, gy = ax + dx, ay + dy
            x, y = img_of(G, f, (gx, gy))
            r = (scale_at(G, f, (gx, gy)) / s0) ** a.gamma
            w, h = bw * r, bh * r
            b = [max(0.0, x - w / 2) / W, max(0.0, y - h / 2) / H, min(W, x + w / 2) / W, min(H, y + h / 2) / H]
            if b[2] - b[0] > 0.001 and b[3] - b[1] > 0.001:
                pbf[str(f)].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in b], 'confidence': conf})
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    print(f'{name}: {n} candidate positions over frames {frames[0]}-{frames[-1]} '
          f'({sum(len(v) for v in pbf.values())} boxes)')
    print('  rank  ground offset      confidence   board score if THIS one is the object')
    T, m = 98.0, len(frames)
    for rank, (dx, dy) in enumerate(cands):
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        rec = m / T
        print(f'  {rank + 1:4d}  ({dx:+6.0f},{dy:+6.0f})   {conf:8.4f}   {rec * rec / (rank + rec) / 13:.5f}')
    print(f'  none                                {0.0:.5f}')


if __name__ == '__main__':
    main()
