#!/usr/bin/env python3
"""Resize a class's answer boxes so they follow the flight's perspective instead of staying one fixed size.

An object's box grows as the drone closes on it: the large tower goes 51 px wide on frame 38 to 67 on frame 66,
a third larger. Several classes are answered at one size for their whole span, which can only hit the middle of
the track. The flight map gives the growth exactly - the local image-px-per-ground-px scale at the object's own
ground spot - so a single reference size per instance fixes every frame of it.

Checked against the large tower: predicted widths 51.3 53.4 55.7 58.1 60.7 63.5 66.6 69.8 against the tracker's
51 53 54 58 60 62 64 67. The slight over-prediction at the near end is the object's own height, which the ground
plane does not model, so the exponent is damped a little below 1.

    python3 persp_probe.py --plan probe-pruned.json --cls medium_launcher --sizes 40x44,44x48 --tag ml
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


def scale_at(G, f, g, d=1.0):
    p = np.linalg.inv(G[f]) @ np.array([g[0], g[1], 1.0])
    x, y = p[:2] / p[2]
    a = to_ground(G[f], x, y)
    J = np.array([to_ground(G[f], x + d, y) - a, to_ground(G[f], x, y + d) - a]).T / d
    return 1.0 / np.sqrt(abs(np.linalg.det(J)))


def cluster(rows, radius):
    inst = []
    for f, gx, gy, b in sorted(rows):
        for k in inst:
            if (k['x'] - gx) ** 2 + (k['y'] - gy) ** 2 <= radius ** 2:
                n = k['n']; k['x'] = (k['x'] * n + gx) / (n + 1); k['y'] = (k['y'] * n + gy) / (n + 1)
                k['n'] += 1; k['rows'].append((f, b)); break
        else:
            inst.append({'x': gx, 'y': gy, 'n': 1, 'rows': [(f, b)]})
    return inst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--sizes', required=True, help='comma list WxH: the size at each instance\'s midpoint frame')
    ap.add_argument('--gamma', type=float, default=0.9, help='damping on the geometric growth')
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--floor', type=float, default=0.8)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= a.floor and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W, (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    inst = cluster(rows, a.radius)
    inst = [k for k in inst if k['n'] >= 5]
    print(f'{a.cls}: {len(inst)} instances ' + ', '.join(f"{min(f for f,_ in k['rows'])}-{max(f for f,_ in k['rows'])}" for k in inst))
    for spec in a.sizes.split(','):
        bw, bh = (float(v) for v in spec.split('x'))
        pbf = defaultdict(list)
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
                                        'confidence': b['confidence']})
        name = f'probe-{a.tag}-p{spec}'
        Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                               'predictions_by_frame': dict(pbf)}))
        print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
