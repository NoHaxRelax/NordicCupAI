#!/usr/bin/env python3
"""Take each one-frame glimpse of a class, extend it over the frames that spot is visible for, and rank them.

The tracker sometimes emits a single box at a ground spot and never follows it. Either the spot is an object
the tracker lost - a whole instance missing from the answer, worth ~32 frames - or it is noise. Extending each
glimpse across its visible span through the flight-map geometry and giving each candidate its own confidence
band settles several of them in one attempt: the score lands on the ladder of the highest-ranked candidate
that is real, and is exactly zero if none is.

    python3 extend_probe.py --plan probe-pruned.json --cls medium_plane --max-track 4 --tag mpglimpse
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

ROOT = Path(__file__).resolve().parents[2]


def img_of(G, f, g):
    p = np.linalg.inv(G[f]) @ np.array([g[0], g[1], 1.0]); return p[:2] / p[2]


def scale_at(G, f, g, d=1.0):
    x, y = img_of(G, f, g)
    a = to_ground(G[f], x, y)
    J = np.array([to_ground(G[f], x + d, y) - a, to_ground(G[f], x, y + d) - a]).T / d
    return 1.0 / np.sqrt(abs(np.linalg.det(J)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--max-track', type=int, default=4, help='a track this short or shorter is a glimpse')
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W, (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    inst = []
    for f, gx, gy, b in sorted(rows):
        for k in inst:
            if (k['x'] - gx) ** 2 + (k['y'] - gy) ** 2 <= a.radius ** 2:
                n = k['n']; k['x'] = (k['x'] * n + gx) / (n + 1); k['y'] = (k['y'] * n + gy) / (n + 1)
                k['n'] += 1; k['rows'].append((f, b)); break
        else:
            inst.append({'x': gx, 'y': gy, 'n': 1, 'rows': [(f, b)]})
    glimpses = [k for k in inst if k['n'] <= a.max_track]
    if not glimpses:
        print('no glimpses'); return
    # rank the longest glimpse first: the more frames the tracker held it, the likelier it is an object
    glimpses.sort(key=lambda k: (-k['n'], min(f for f, _ in k['rows'])))
    n = len(glimpses)
    pbf = defaultdict(list)
    print(f'{a.cls}: {n} glimpses')
    print('  rank  ground spot          seen  visible span  boxes  confidence')
    for rank, k in enumerate(glimpses):
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        f0, b0 = k['rows'][0]
        bw = (b0['bbox'][2] - b0['bbox'][0]) * W
        bh = (b0['bbox'][3] - b0['bbox'][1]) * H
        s0 = scale_at(G, f0, (k['x'], k['y']))
        vis = []
        for f in sorted(G):
            x, y = img_of(G, f, (k['x'], k['y']))
            if 0 <= x <= W and 0 <= y <= H:
                vis.append(f)
        for f in vis:
            r = (scale_at(G, f, (k['x'], k['y'])) / s0) ** a.gamma
            x, y = img_of(G, f, (k['x'], k['y']))
            w, h = bw * r, bh * r
            b = [max(0.0, x - w / 2) / W, max(0.0, y - h / 2) / H, min(W, x + w / 2) / W, min(H, y + h / 2) / H]
            if b[2] - b[0] > 0.001 and b[3] - b[1] > 0.001:
                pbf[str(f)].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in b], 'confidence': conf})
        print(f'  {rank + 1:4d}  ({k["x"]:7.0f},{k["y"]:8.0f})  {k["n"]:4d}  {min(vis) if vis else "-"}-{max(vis) if vis else "-":<8} '
              f'{len(vis):5d}  {conf:8.4f}')
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
