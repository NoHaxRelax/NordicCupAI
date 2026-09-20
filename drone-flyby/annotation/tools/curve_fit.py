#!/usr/bin/env python3
"""Refit each instance's box size on the perspective curve instead of a straight line in frame number.

The track-end probes say the ends are hitting - about 10 of 12 end frames on both large_launcher and
large_tower - so the remaining misses are in the middle of tracks, which is where a straight line through two
drawn frames deviates most. Projected size goes as the local image scale, not linearly with frame number, and
that curve is convex: a chord between two points on it runs BELOW the curve in between.

With two drawn frames per instance both the reference size and the exponent are determined exactly:
    w(f) = w0 * (s(f)/s(ref))^g,  g = log(w2/w1) / log(s2/s1)
so the fit passes through both drawn boxes and bends the way the geometry says in between. The exponent is
clamped to a sane band, since two noisy measurements can imply an absurd one.

    python3 curve_fit.py --answers box-answers.json --cls large_tower --plan probe-dr-largetower.json --out x.json
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
    ap.add_argument('--cls', required=True)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--clamp', type=float, default=3.0, help='max |exponent|')
    ap.add_argument('--match', type=float, default=140.0)
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    drawn = [d for d in json.loads(Path(a.answers).read_text())['answers']
             if d['cls'] == a.cls and not d['absent'] and d['box']]
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    inst = [k for k in cluster(rows, 110.0) if k['n'] >= 5]
    samples = defaultdict(list)
    for d in drawn:
        cx, cy = (d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2
        best, bd = None, 1e9
        for i, k in enumerate(inst):
            for f, b in k['rows']:
                if f != d['frame']:
                    continue
                dd = float(np.hypot((b['bbox'][0] + b['bbox'][2]) / 2 * W - cx,
                                    (b['bbox'][1] + b['bbox'][3]) / 2 * H - cy))
                if dd < bd:
                    best, bd = i, dd
        if best is not None and bd <= a.match:
            samples[best].append(d)

    pbf = {f: [dict(b) for b in v] for f, v in plan['predictions_by_frame'].items()}
    for i, ds in sorted(samples.items()):
        ds.sort(key=lambda d: d['frame'])
        if len(ds) < 2:
            continue
        k = inst[i]
        ss = [scale_at(G, d['frame'], (k['x'], k['y'])) for d in ds]
        ws = [d['box'][2] - d['box'][0] for d in ds]
        hs = [d['box'][3] - d['box'][1] for d in ds]
        def expo(vs):
            if abs(np.log(ss[-1] / ss[0])) < 1e-9:
                return 0.0
            return float(np.clip(np.log(vs[-1] / vs[0]) / np.log(ss[-1] / ss[0]), -a.clamp, a.clamp))
        gw, gh = expo(ws), expo(hs)
        ref, w0, h0 = ss[0], ws[0], hs[0]
        for f, b in k['rows']:
            s = scale_at(G, f, (k['x'], k['y']))
            w, h = w0 * (s / ref) ** gw, h0 * (s / ref) ** gh
            x0, y0, x1, y1 = b['bbox']
            cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
            nb = [max(0.0, cx - w / 2) / W if x0 > 0 else 0.0, max(0.0, cy - h / 2) / H if y0 > 0 else 0.0,
                  min(W, cx + w / 2) / W if x1 < 1 else 1.0, min(H, cy + h / 2) / H if y1 < 1 else 1.0]
            for t in pbf[str(f)]:
                if t['object_id'] == a.cls and t['bbox'] == b['bbox']:
                    t['bbox'] = [round(v, 6) for v in nb]; break
        lo, hi = min(f for f, _ in k['rows']), max(f for f, _ in k['rows'])
        print(f'  #{i+1} frames {lo}-{hi}: exponents w {gw:+.2f} h {gh:+.2f} '
              f'-> {w0*(scale_at(G,lo,(k["x"],k["y"]))/ref)**gw:.0f}x{h0*(scale_at(G,lo,(k["x"],k["y"]))/ref)**gh:.0f} '
              f'at {lo}, {w0*(scale_at(G,hi,(k["x"],k["y"]))/ref)**gw:.0f}x{h0*(scale_at(G,hi,(k["x"],k["y"]))/ref)**gh:.0f} at {hi}')
    out = {f: [b for b in v if b['object_id'] == a.cls] for f, v in pbf.items()}
    out = {f: v for f, v in sorted(out.items(), key=lambda kv: int(kv[0])) if v}
    Path(a.out).write_text(json.dumps({'name': Path(a.out).stem, 'target': [480, 270],
                                       'predictions_by_frame': out}))
    print(f'{a.out}: {sum(len(v) for v in out.values())} boxes')


if __name__ == '__main__':
    main()
