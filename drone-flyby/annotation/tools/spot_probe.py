#!/usr/bin/env python3
"""A single-class probe answering one ground position, to settle whether a track is a real object.

Each object sits at a fixed ground spot; the flight map turns every answer box back into that spot, so a track
can be pulled out of the plan by position instead of by frame range. A position that is not an object scores
exactly zero, which is the cheapest possible test: one attempt, one class, a fraction of 1/13 of the board.

    python3 spot_probe.py --plan probe-pruned.json --cls small_launcher --spot -464,-3817 --tag sl1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--spot', required=True, help='ground x,y; repeatable as x,y;x,y')
    ap.add_argument('--radius', type=float, default=90.0)
    ap.add_argument('--invert', action='store_true', help='answer everything EXCEPT these spots')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    spots = [tuple(float(v) for v in s.split(',')) for s in a.spot.split(';')]
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    pbf = {}
    for f, boxes in plan['predictions_by_frame'].items():
        keep = []
        for b in boxes:
            if b['object_id'] != a.cls:
                continue
            if int(f) not in G:
                continue
            g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W, (b['bbox'][1] + b['bbox'][3]) / 2 * H)
            hit = any((g[0] - sx) ** 2 + (g[1] - sy) ** 2 <= a.radius ** 2 for sx, sy in spots)
            if hit != a.invert:
                keep.append(b)
        if keep:
            pbf[f] = keep
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': pbf}))
    print(f'  {name}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames')


if __name__ == '__main__':
    main()
