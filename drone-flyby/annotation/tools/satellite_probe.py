#!/usr/bin/env python3
"""Search for a satellite object beside an anchor track, at the radius satellites actually sit at.

An earlier ring probe sampled radii 0, 40 and 75 ground px and came back 0 twice, which looked like proof of
absence. It was not: the confirmed small-launcher pair sits 24-37 ground px from its tower, and the medium
launcher at installation 3 sits about 21 from its tower, so both searches had their blind spot exactly where
the object was. This samples the radius band that satellites are known to occupy, and also tries the two image
offsets the confirmed pair uses, which is the likeliest arrangement if the layout repeats.

Candidates are ranked in confidence bands, so one attempt reads off which one is real.

    python3 satellite_probe.py --cls small_launcher --anchor-cls small_tower --anchor 343,-6058 \
        --size 23x30 --tag sl-t1fine
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
CONFIRMED_IMAGE_OFFSETS = [(27.0, 6.0), (4.0, 31.0)]   # the tower-2 pair, both proven on the API


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--anchor-cls', required=True)
    ap.add_argument('--anchor', required=True)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-pruned.json'))
    ap.add_argument('--radii', default='24,32,40')
    ap.add_argument('--dirs', type=int, default=8)
    ap.add_argument('--size', default='23x30')
    ap.add_argument('--truth', type=float, default=160.0)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    ax, ay = (float(v) for v in a.anchor.split(','))
    bw, bh = (float(v) for v in a.size.split('x'))
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())

    anchor = {}
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] != a.anchor_cls or b['confidence'] < 0.8 or int(f) not in G:
                continue
            g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W, (b['bbox'][1] + b['bbox'][3]) / 2 * H)
            if (g[0] - ax) ** 2 + (g[1] - ay) ** 2 <= 200 ** 2:
                anchor[int(f)] = [b['bbox'][0] * W, b['bbox'][1] * H, b['bbox'][2] * W, b['bbox'][3] * H]
    if not anchor:
        print('anchor track not found'); return

    cands = [('image offset ' + str(o), o, None) for o in CONFIRMED_IMAGE_OFFSETS]
    for r in (float(v) for v in a.radii.split(',')):
        for k in range(a.dirs):
            t = 2 * math.pi * k / a.dirs
            cands.append((f'ground r={r:g} dir={k}', None, (r * math.cos(t), r * math.sin(t))))
    n = len(cands)
    pbf = defaultdict(list)
    for rank, (label, ioff, goff) in enumerate(cands):
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        for f, tb in anchor.items():
            cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            if ioff is not None:
                x, y = cx + ioff[0], cy + ioff[1]
            else:
                p = np.linalg.inv(G[f]) @ np.array([ax + goff[0], ay + goff[1], 1.0])
                x, y = p[:2] / p[2]
            nb = [max(0.0, x - bw / 2) / W, max(0.0, y - bh / 2) / H,
                  min(W, x + bw / 2) / W, min(H, y + bh / 2) / H]
            if nb[2] - nb[0] > 0.001 and nb[3] - nb[1] > 0.001:
                pbf[str(f)].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in nb], 'confidence': conf})
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    m = len(anchor)
    rec = m / a.truth
    print(f'{name}: {n} candidates over {m} frames, {sum(len(v) for v in pbf.values())} boxes')
    for rank, (label, _, _) in enumerate(cands[:6]):
        print(f'  {rank + 1:3d} {label:26s} -> {rec * rec / (rank + rec) / 13:.5f}')
    print(f'  ... lowest rung {rec * rec / (n - 1 + rec) / 13:.5f}, none 0.00000')


if __name__ == '__main__':
    main()
