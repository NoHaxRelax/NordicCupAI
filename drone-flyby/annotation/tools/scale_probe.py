#!/usr/bin/env python3
"""Single-class probes that grow or shrink a plan's band-1 boxes about their centre.

Finds the extent the organiser actually uses for a class whose centres are right but whose size is a blend of
the detector and a prior from the other scene, without re-deriving the track. One class per probe, so an
attempt can never exceed 1/13 of the board.

    python3 scale_probe.py --plan probe-pruned.json --cls helicopter --scales 1.0,1.25,1.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--scales', required=True, help='comma list, or S or SxSY')
    ap.add_argument('--floor', type=float, default=0.8)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    short = a.cls.replace('_', '').replace('-', '')
    for spec in a.scales.split(','):
        sx, sy = (float(v) for v in (spec.split('x') * 2)[:2]) if 'x' in spec else (float(spec), float(spec))
        pbf = {}
        for f, boxes in plan['predictions_by_frame'].items():
            keep = []
            for b in boxes:
                if b['object_id'] != a.cls or b['confidence'] < a.floor:
                    continue
                x0, y0, x1, y1 = b['bbox']
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * sx, (y1 - y0) * sy
                # a box clipped by the frame edge keeps its clipped side: the truth there is the visible strip
                nb = [max(0.0, cx - w / 2) if x0 > 0 else 0.0, max(0.0, cy - h / 2) if y0 > 0 else 0.0,
                      min(1.0, cx + w / 2) if x1 < 1 else 1.0, min(1.0, cy + h / 2) if y1 < 1 else 1.0]
                if nb[2] - nb[0] > 0.001 and nb[3] - nb[1] > 0.001:
                    keep.append({'object_id': a.cls, 'bbox': [round(v, 6) for v in nb],
                                 'confidence': b['confidence']})
            if keep:
                pbf[f] = keep
        name = f"probe-{short}-s{spec}".replace('.', '')
        Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                               'predictions_by_frame': pbf}))
        print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
