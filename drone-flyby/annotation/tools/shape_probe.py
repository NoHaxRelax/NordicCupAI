#!/usr/bin/env python3
"""One scripted single-class plan per box shape, at the centres the pipeline already found.

Used to find the organiser's box for a class whose answers keep falling short of IoU 0.5: the centre is right (the
tracker follows the object) but the extent is a blend of the detector and a prior from the other scene. Each probe
answers one class only, so it can never score more than 1/13 of the board.

    python3 shape_probe.py --cls large_tower --instance 1 --sizes 38x63,45x75,50x85 --out-dir probes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import FALSE, H, W, live_boxes, load_geometry, near, to_ground

ROOT = Path(__file__).resolve().parents[2]


def instances(cls, G, radius=110):
    pts = []
    for f, b, c in live_boxes(cls):
        if f not in G or near(G, f, b, FALSE.get(cls, []), 110):
            continue
        g = to_ground(G[f], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        pts.append((g[0], g[1], f, b, c))
    out = []
    for x, y, f, b, c in sorted(pts, key=lambda t: t[2]):
        for k in out:
            if (k['x'] - x) ** 2 + (k['y'] - y) ** 2 <= radius ** 2:
                n = k['n']; k['x'] = (k['x'] * n + x) / (n + 1); k['y'] = (k['y'] * n + y) / (n + 1); k['n'] += 1
                k['rows'].append((f, b, c)); break
        else:
            out.append({'x': x, 'y': y, 'n': 1, 'rows': [(f, b, c)]})
    return [k for k in out if k['n'] >= 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--instance', type=int, default=None, help='index of the instance, or all of them')
    ap.add_argument('--sizes', required=True, help='comma list WxH, or WxH+dx+dy to shift the centre')
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    inst = instances(a.cls, G)
    print(f'{a.cls}: {len(inst)} instances ' + ', '.join(f"{i}:{min(f for f,_,_ in k['rows'])}-{max(f for f,_,_ in k['rows'])}" for i, k in enumerate(inst)))
    rows = [r for i, k in enumerate(inst) if a.instance is None or i == a.instance for r in k['rows']]
    for spec in a.sizes.split(','):
        body = spec.split('+')
        w, h = (float(v) for v in body[0].split('x'))
        dx = float(body[1]) if len(body) > 1 else 0.0
        dy = float(body[2]) if len(body) > 2 else 0.0
        pbf = {}
        for f, b, c in rows:
            cx, cy = (b[0] + b[2]) / 2 + dx, (b[1] + b[3]) / 2 + dy
            box = [max(0.0, cx - w / 2) / W, max(0.0, cy - h / 2) / H, min(float(W), cx + w / 2) / W, min(float(H), cy + h / 2) / H]
            if box[2] - box[0] > 0.002 and box[3] - box[1] > 0.002:
                pbf.setdefault(str(f), []).append({'object_id': a.cls, 'bbox': box, 'confidence': 0.9})
        tag = a.tag or f"{a.cls.replace('_', '')}{'' if a.instance is None else a.instance}"
        name = f'probe-{tag}-{spec}'
        Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270], 'predictions_by_frame': pbf}))
        print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
