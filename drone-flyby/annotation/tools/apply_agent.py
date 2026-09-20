#!/usr/bin/env python3
"""Fold an agent's sparse per-instance measurements into a class's boxes, and drop the tracks it marked absent.

The agents measured a handful of frames per instance rather than every frame, so each instance's width, height
and centre offset are fitted as a straight line against frame number and evaluated on the whole track. Entries
marked absent name a stray track - a box on a road verge, on bare grass, on another class's tail - and those
frames are removed outright, which is a free precision gain.

    python3 apply_agent.py --answers agent-boxes-small_plane.json --cls small_plane --out probe-sp-agent.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground
from persp_probe import cluster

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-best-v2.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--match', type=float, default=140.0)
    ap.add_argument('--radius', type=float, default=110.0,
                    help='ground px that count as one instance; the parked planes sit ~50 apart, so the '
                         'default merges them and averages three different aircraft into one fit')
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    ans = json.loads(Path(a.answers).read_text())['answers']

    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    inst = [k for k in cluster(rows, a.radius) if k['n'] >= 5]

    drop_frames = {a_['frame'] for a_ in ans if a_['absent']}
    samples = defaultdict(list)
    for d in ans:
        if d['absent'] or not d['box']:
            continue
        cx, cy = (d['box'][0] + d['box'][2]) / 2, (d['box'][1] + d['box'][3]) / 2
        best, bd = None, 1e9
        for i, k in enumerate(inst):
            for f, b in k['rows']:
                if f != d['frame']:
                    continue
                ox = (b['bbox'][0] + b['bbox'][2]) / 2 * W
                oy = (b['bbox'][1] + b['bbox'][3]) / 2 * H
                dd = float(np.hypot(ox - cx, oy - cy))
                if dd < bd:
                    best, bd = (i, b, ox, oy), dd
        if best is None or bd > a.match:
            print(f"  unmatched drawing: {d['inst']} f{d['frame']} (nearest box {bd:.0f} px)"); continue
        i, b, ox, oy = best
        samples[i].append({'f': d['frame'], 'w': d['box'][2] - d['box'][0], 'h': d['box'][3] - d['box'][1],
                           'dx': cx - ox, 'dy': cy - oy})

    pbf = {f: [dict(b) for b in v] for f, v in plan['predictions_by_frame'].items()}
    fit = lambda xs, ys: (0.0, ys[0]) if len(xs) == 1 else tuple(np.polyfit(np.array(xs, float), np.array(ys, float), 1))
    rewritten = 0
    for i, ss in sorted(samples.items()):
        ss.sort(key=lambda s: s['f'])
        fs = [s['f'] for s in ss]
        aw, bw = fit(fs, [s['w'] for s in ss]); ah, bh = fit(fs, [s['h'] for s in ss])
        ax_, bx = fit(fs, [s['dx'] for s in ss]); ay_, by = fit(fs, [s['dy'] for s in ss])
        k = inst[i]
        lo, hi = min(f for f, _ in k['rows']), max(f for f, _ in k['rows'])
        for f, b in k['rows']:
            fc = min(max(f, fs[0] - 10), fs[-1] + 10)
            w, h = max(6.0, aw * fc + bw), max(6.0, ah * fc + bh)
            x0, y0, x1, y1 = b['bbox']
            cx = (x0 + x1) / 2 * W + ax_ * fc + bx
            cy = (y0 + y1) / 2 * H + ay_ * fc + by
            nb = [max(0.0, cx - w / 2) / W if x0 > 0 else 0.0, max(0.0, cy - h / 2) / H if y0 > 0 else 0.0,
                  min(W, cx + w / 2) / W if x1 < 1 else 1.0, min(H, cy + h / 2) / H if y1 < 1 else 1.0]
            for t in pbf[str(f)]:
                if t['object_id'] == a.cls and t['bbox'] == b['bbox']:
                    t['bbox'] = [round(v, 6) for v in nb]; rewritten += 1; break
        print(f"  #{i+1} frames {lo}-{hi}: {len(ss)} measurements -> "
              f"{aw*lo+bw:.0f}x{ah*lo+bh:.0f} at {lo}, {aw*hi+bw:.0f}x{ah*hi+bh:.0f} at {hi}, "
              f"centre shift {ax_*hi+bx:+.0f},{ay_*hi+by:+.0f} at {hi}")

    dropped = 0
    for f in drop_frames:
        keep = [b for b in pbf.get(str(f), []) if b['object_id'] != a.cls or b['confidence'] < 0.8]
        dropped += len(pbf.get(str(f), [])) - len(keep)
        pbf[str(f)] = keep
    out = {f: [b for b in v if b['object_id'] == a.cls] for f, v in pbf.items()}
    out = {f: v for f, v in sorted(out.items(), key=lambda kv: int(kv[0])) if v}
    Path(a.out).write_text(json.dumps({'name': Path(a.out).stem, 'target': [480, 270],
                                       'predictions_by_frame': out}))
    print(f'{a.out}: {sum(len(v) for v in out.values())} boxes, {rewritten} rewritten, '
          f'{dropped} stray boxes dropped on frames {sorted(drop_frames)}')


if __name__ == '__main__':
    main()
