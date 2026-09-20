#!/usr/bin/env python3
"""Turn boxes drawn in the review page into a corrected plan.

Each drawn box is one frame of one instance. Two frames of the same instance give both the size and how it
grows down the track, and the flight-map perspective scale says how to carry that to every other frame, so a
pair of drawn boxes fixes a whole track. Where two frames were drawn the growth is taken from the geometry and
the reference size from their mean; where only one was drawn, that size is used at that frame and scaled.

A drawn box also carries a centre offset from the tracker's own box, which is applied to every frame of the
instance - the tracker's centres drift a few pixels off the object and that alone costs IoU.

    python3 apply_answers.py --answers box-answers.json --plan probe-pruned.json --out probe-drawn.json
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
    ap.add_argument('--answers', required=True)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--gamma', type=float, default=0.9)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    drawn = json.loads(Path(a.answers).read_text())['answers']

    # group the plan's band-1 boxes into instances, the same way the review page numbered them
    byc = defaultdict(list)
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                byc[b['object_id']].append((int(f), g[0], g[1], b))
    inst = {c: [k for k in cluster(rows, a.radius) if k['n'] >= 8] for c, rows in byc.items()}

    # attach each drawn box to the instance whose track covers that frame near that point
    fixes = defaultdict(list)
    absent, unmatched = [], []
    for d in drawn:
        if d['kind'] != 'extent':
            (absent if d['absent'] else fixes[('new', d['cls'])]).append(d)
            continue
        if d['absent']:
            absent.append(d); continue
        cx = (d['box'][0] + d['box'][2]) / 2
        cy = (d['box'][1] + d['box'][3]) / 2
        best, bestd = None, 1e9
        for i, k in enumerate(inst.get(d['cls'], [])):
            for f, b in k['rows']:
                if f != d['frame']:
                    continue
                ox = (b['bbox'][0] + b['bbox'][2]) / 2 * W
                oy = (b['bbox'][1] + b['bbox'][3]) / 2 * H
                dd = np.hypot(ox - cx, oy - cy)
                if dd < bestd:
                    best, bestd = (i, f, b), dd
        if best is None or bestd > 160:
            unmatched.append(d); continue
        i, f, b = best
        fixes[(d['cls'], i)].append((f, d['box'], b))

    pbf = defaultdict(list)
    changed = 0
    applied = {}
    for f, boxes in plan['predictions_by_frame'].items():
        pbf[f] = [dict(b) for b in boxes]
    for key, items in fixes.items():
        if key[0] == 'new':
            continue
        cls, i = key
        k = inst[cls][i]
        ref_specs = []
        for f, db, b in items:
            s = scale_at(G, f, (k['x'], k['y']))
            ow = (b['bbox'][2] - b['bbox'][0]) * W
            oh = (b['bbox'][3] - b['bbox'][1]) * H
            ocx = (b['bbox'][0] + b['bbox'][2]) / 2 * W
            ocy = (b['bbox'][1] + b['bbox'][3]) / 2 * H
            ref_specs.append({'f': f, 's': s, 'w': db[2] - db[0], 'h': db[3] - db[1],
                              'dx': (db[0] + db[2]) / 2 - ocx, 'dy': (db[1] + db[3]) / 2 - ocy,
                              'ow': ow, 'oh': oh})
        # a size defined at scale 1: the drawn size divided by that frame's own perspective scale
        base_w = float(np.mean([r['w'] / r['s'] ** a.gamma for r in ref_specs]))
        base_h = float(np.mean([r['h'] / r['s'] ** a.gamma for r in ref_specs]))
        # the centre offset is carried in units of the object's own size, so it grows with the track too
        off_x = float(np.mean([r['dx'] / r['w'] for r in ref_specs]))
        off_y = float(np.mean([r['dy'] / r['h'] for r in ref_specs]))
        applied[f'{cls} #{i + 1}'] = {
            'from': [round(float(np.median([r['ow'] for r in ref_specs])), 1),
                     round(float(np.median([r['oh'] for r in ref_specs])), 1)],
            'drawn': [[round(r['w'], 1), round(r['h'], 1)] for r in ref_specs],
            'offset_frac': [round(off_x, 3), round(off_y, 3)], 'frames': len(k['rows'])}
        for f, b in k['rows']:
            s = scale_at(G, f, (k['x'], k['y'])) ** a.gamma
            w, h = base_w * s, base_h * s
            x0, y0, x1, y1 = b['bbox']
            cx = (x0 + x1) / 2 * W + off_x * w
            cy = (y0 + y1) / 2 * H + off_y * h
            nb = [max(0.0, cx - w / 2) / W if x0 > 0 else 0.0, max(0.0, cy - h / 2) / H if y0 > 0 else 0.0,
                  min(W, cx + w / 2) / W if x1 < 1 else 1.0, min(H, cy + h / 2) / H if y1 < 1 else 1.0]
            for t in pbf[str(f)]:
                if t is not b and t.get('bbox') == b['bbox'] and t['object_id'] == cls:
                    t['bbox'] = [round(v, 6) for v in nb]; changed += 1; break
            else:
                for t in pbf[str(f)]:
                    if t['object_id'] == cls and t['bbox'] == b['bbox']:
                        t['bbox'] = [round(v, 6) for v in nb]; changed += 1; break
    plan['predictions_by_frame'] = {f: v for f, v in pbf.items() if v}
    plan['name'] = Path(a.out).stem
    Path(a.out).write_text(json.dumps(plan))
    print(f'{a.out}: {changed} boxes rewritten from {len(drawn)} drawn answers')
    for k, v in sorted(applied.items()):
        print(f"  {k:22s} tracker {v['from']} -> drawn {v['drawn']}, "
              f"offset {v['offset_frac']} of box, {v['frames']} frames")
    if absent:
        print('  marked absent: ' + ', '.join(f"{d['cls']} f{d['frame']}" for d in absent))
    if unmatched:
        print('  could not match to a track: ' + ', '.join(f"{d['cls']} f{d['frame']}" for d in unmatched))


if __name__ == '__main__':
    main()
