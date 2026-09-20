#!/usr/bin/env python3
"""Find how much larger the organiser's box is than the visible object, in one attempt.

Boxes drawn tight around the silhouette scored 0.195 on medium_launcher where the tracker's looser boxes
scored 0.519, so the organiser's box is bigger than what the eye outlines - most likely the object's full
three-dimensional extent projected, not its footprint. The drawn boxes still have the better centre and the
better aspect, so the useful question is only the factor.

Each factor is answered in its own confidence band, so they are matched in strict order and the score lands
on a ladder with one rung per factor: the first factor that hits is the one the organiser uses.

    python3 inflate_ladder.py --plan probe-drawn.json --cls medium_launcher --factors 1.2,1.35,1.5,1.65 --truth 97
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
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--factors', required=True, help='comma list, most likely first; F or FWxFH')
    ap.add_argument('--truth', type=float, required=True)
    ap.add_argument('--floor', type=float, default=0.8)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    facs = a.factors.split(',')
    n = len(facs)
    pbf = defaultdict(list)
    frames = set()
    for rank, spec in enumerate(facs):
        fw, fh = (float(v) for v in spec.split('x')) if 'x' in spec else (float(spec), float(spec))
        conf = round(0.95 - 0.9 * rank / max(1, n - 1), 4)
        for f, boxes in plan['predictions_by_frame'].items():
            for b in boxes:
                if b['object_id'] != a.cls or b['confidence'] < a.floor:
                    continue
                x0, y0, x1, y1 = b['bbox']
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * fw, (y1 - y0) * fh
                nb = [max(0.0, cx - w / 2) if x0 > 0 else 0.0, max(0.0, cy - h / 2) if y0 > 0 else 0.0,
                      min(1.0, cx + w / 2) if x1 < 1 else 1.0, min(1.0, cy + h / 2) if y1 < 1 else 1.0]
                if nb[2] - nb[0] > 0.001 and nb[3] - nb[1] > 0.001:
                    pbf[f].append({'object_id': a.cls, 'bbox': [round(v, 6) for v in nb], 'confidence': conf})
                    frames.add(f)
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': dict(pbf)}))
    rec = len(frames) / a.truth
    print(f'{name}: {n} factors over {len(frames)} frames, {sum(len(v) for v in pbf.values())} boxes')
    print('  rank  factor  confidence   board score if this is the first factor that hits')
    for rank, spec in enumerate(facs):
        print(f'  {rank + 1:4d}  {spec:6s}  {round(0.95 - 0.9 * rank / max(1, n - 1), 4):8.4f}   '
              f'{rec * rec / (rank + rec) / 13:.5f}')
    print(f'  none                        {0.0:.5f}')


if __name__ == '__main__':
    main()
