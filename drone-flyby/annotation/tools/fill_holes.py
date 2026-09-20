#!/usr/bin/env python3
"""Answer the frames where an instance is inside the picture but the plan has no box on it.

Every object sits at a fixed ground spot, so the flight map says exactly which frames see it; a frame in that
range with no box is a truth frame thrown away. The box comes from the instance's own answered frames, carried
by the perspective scale, so the filled frames match the ones the tracker did get.

    python3 fill_holes.py --plan probe-best-v2.json --out probe-filled.json --classes tank,small_launcher
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
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', required=True)
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--min-track', type=int, default=6)
    ap.add_argument('--conf', type=float, default=0.86, help='below the tracker\'s own boxes, above the rest')
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    want = set(a.classes.split(','))
    pbf = {f: [dict(b) for b in v] for f, v in plan['predictions_by_frame'].items()}
    added = defaultdict(int)
    for cls in want:
        rows = []
        for f, boxes in plan['predictions_by_frame'].items():
            for b in boxes:
                if b['object_id'] == cls and b['confidence'] >= 0.8 and int(f) in G:
                    g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                                  (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                    rows.append((int(f), g[0], g[1], b))
        for k in cluster(rows, a.radius):
            if k['n'] < a.min_track:
                continue
            have = {f for f, _ in k['rows']}
            fs = sorted(have)
            ref = fs[len(fs) // 2]
            s0 = scale_at(G, ref, (k['x'], k['y']))
            sizes = [((b['bbox'][2] - b['bbox'][0]) * W / (scale_at(G, f, (k['x'], k['y'])) / s0) ** a.gamma,
                      (b['bbox'][3] - b['bbox'][1]) * H / (scale_at(G, f, (k['x'], k['y'])) / s0) ** a.gamma)
                     for f, b in k['rows']
                     if b['bbox'][1] > 0.003 and b['bbox'][3] < 0.997]
            if not sizes:
                continue
            bw = float(np.median([s[0] for s in sizes])); bh = float(np.median([s[1] for s in sizes]))
            for f in sorted(G):
                if f in have or f < min(fs) - 40 or f > max(fs) + 40:
                    continue
                p = np.linalg.inv(G[f]) @ np.array([k['x'], k['y'], 1.0])
                x, y = p[:2] / p[2]
                if not (0 <= x <= W and 0 <= y <= H):
                    continue
                r = (scale_at(G, f, (k['x'], k['y'])) / s0) ** a.gamma
                w, h = bw * r, bh * r
                nb = [max(0.0, x - w / 2) / W, max(0.0, y - h / 2) / H,
                      min(W, x + w / 2) / W, min(H, y + h / 2) / H]
                if nb[2] - nb[0] < 0.001 or nb[3] - nb[1] < 0.001:
                    continue
                pbf.setdefault(str(f), []).append({'object_id': cls, 'bbox': [round(v, 6) for v in nb],
                                                   'confidence': a.conf})
                added[cls] += 1
    plan['predictions_by_frame'] = {f: v for f, v in sorted(pbf.items(), key=lambda kv: int(kv[0])) if v}
    plan['name'] = Path(a.out).stem
    Path(a.out).write_text(json.dumps(plan))
    print(f'{a.out}: {sum(len(v) for v in plan["predictions_by_frame"].values())} boxes')
    for c, n in sorted(added.items()):
        print(f'  {c:16s} +{n} filled frames')


if __name__ == '__main__':
    main()
