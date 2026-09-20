#!/usr/bin/env python3
"""Test several row-dependent downward shifts for one class in a single attempt.

Three agents and the review page independently found the same bias: near the bottom of the frame the answer
box sits 5-17 px ABOVE the object, and the error grows with image row. It is corrected in the five classes
that were measured by hand; the other eight still carry it. The shift applied here is k*(cy-1080) for boxes
below mid-frame, so it is zero at the middle and k*1080 at the bottom edge.

Each k gets its own confidence band, so the bands match in strict order and the score lands on a ladder with
one rung per k - the first k that hits is the one to use.

    python3 shift_ladder.py --cls tank --ks 0.011,0.018,0.006,0.025 --truth 220 --tag tk-shift
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W, H = 3840, 2160


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes/probe-full.json'))
    ap.add_argument('--cls', required=True)
    ap.add_argument('--ks', required=True, help='comma list, most likely first')
    ap.add_argument('--pivot', type=float, default=1080.0, help='row below which the shift starts')
    ap.add_argument('--truth', type=float, required=True)
    ap.add_argument('--floor', type=float, default=0.8)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    ks = [float(v) for v in a.ks.split(',')]
    n = len(ks)
    pbf = defaultdict(list)
    boxes_per_k = 0
    for rank, k in enumerate(ks):
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        cnt = 0
        for f, boxes in plan['predictions_by_frame'].items():
            for b in boxes:
                if b['object_id'] != a.cls or b['confidence'] < a.floor:
                    continue
                x0, y0, x1, y1 = b['bbox']
                cy = (y0 + y1) / 2 * H
                dy = k * max(0.0, cy - a.pivot) / H
                nb = [x0, min(1.0, y0 + dy) if y0 > 0 else 0.0, x1, min(1.0, y1 + dy)]
                if nb[3] - nb[1] > 0.001:
                    pbf[f].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in nb], 'confidence': conf})
                    cnt += 1
        boxes_per_k = cnt
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    rec = min(1.0, boxes_per_k / a.truth)
    print(f'{name}: {n} shifts over {len(pbf)} frames, {boxes_per_k} boxes each')
    print('  rank  k        px at bottom   board score if this is the first k that hits')
    for rank, k in enumerate(ks):
        print(f'  {rank + 1:4d}  {k:<7.3f}  {k * 1080:6.1f}         {rec * rec / (rank + rec) / 13:.5f}')


if __name__ == '__main__':
    main()
