#!/usr/bin/env python3
"""Which frames should see each instance, against which frames the plan answers it.

Every object sits at a fixed spot on the ground; the flight map's per-frame homography says exactly where that
spot lands in each frame. Inverting it gives the frame range an instance is inside the picture for - the truth
frames, occlusion aside - so a hole in the plan shows up as a frame in the range with no box on it.

    python3 coverage.py --plan artifacts/drone-verifier-20260919/probes/probe-pruned.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import CLASSES, H, W, load_geometry, to_ground

ROOT = Path(__file__).resolve().parents[2]


def to_image(G, gx, gy):
    p = np.linalg.inv(G) @ np.array([gx, gy, 1.0])
    return p[:2] / p[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--radius', type=float, default=90.0, help='ground px that count as the same instance')
    ap.add_argument('--margin', type=float, default=0.0, help='image px an instance may sit outside the frame')
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    answers = defaultdict(list)
    for f, boxes in plan['predictions_by_frame'].items():
        f = int(f)
        if f not in G:
            continue
        for b in boxes:
            x = (b['bbox'][0] + b['bbox'][2]) / 2 * W
            y = (b['bbox'][1] + b['bbox'][3]) / 2 * H
            answers[b['object_id']].append((f, *to_ground(G[f], x, y)))
    total_hole = 0
    for cls in CLASSES:
        inst = []
        for f, gx, gy in sorted(answers[cls]):
            for k in inst:
                if (k['x'] - gx) ** 2 + (k['y'] - gy) ** 2 <= a.radius ** 2:
                    n = k['n']; k['x'] = (k['x'] * n + gx) / (n + 1); k['y'] = (k['y'] * n + gy) / (n + 1)
                    k['n'] += 1; k['frames'].add(f); break
            else:
                inst.append({'x': gx, 'y': gy, 'n': 1, 'frames': {f}})
        for i, k in enumerate(sorted(inst, key=lambda d: min(d['frames']))):
            vis = set()
            for f in sorted(G):
                x, y = to_image(G[f], k['x'], k['y'])
                if -a.margin <= x <= W + a.margin and -a.margin <= y <= H + a.margin:
                    vis.add(f)
            holes = sorted(vis - k['frames'])
            extra = sorted(k['frames'] - vis)
            total_hole += len(holes)
            flag = '' if not holes else '   HOLES ' + ','.join(str(h) for h in holes[:14]) + ('...' if len(holes) > 14 else '')
            ex = '' if not extra else f'   outside {len(extra)}'
            print(f'{cls:16s} #{i} ground ({k["x"]:7.0f},{k["y"]:8.0f})  visible {min(vis) if vis else "-"}-{max(vis) if vis else "-"} '
                  f'({len(vis):3d})  answered {len(k["frames"]):3d}{flag}{ex}')
    print(f'\ntotal frames visible but unanswered: {total_hole}')


if __name__ == '__main__':
    main()
