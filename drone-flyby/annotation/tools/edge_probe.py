#!/usr/bin/env python3
"""Answer only the first and last frames of each track, to find out whether the track ends are hitting at all.

Every class left above 0.92 is a handful of frames short, and the arithmetic says those frames are not
missing boxes - they are boxes that miss. The likeliest place is the track ends, where the object is entering
or leaving the frame, the box is clipped, and a straight-line fit through the middle extrapolates worst. This
answers ONLY those frames, so the score says directly whether they hit: near the ceiling means they are fine
and the misses are elsewhere, near zero means every end frame is wasted.

    python3 edge_probe.py --cls large_launcher --n 2 --truth 81 --tag ll-edge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_best_plan import H, W, load_geometry, to_ground
from persp_probe import cluster

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--cls', required=True)
    ap.add_argument('--n', type=int, default=2, help='frames from each end of each track')
    ap.add_argument('--truth', type=float, required=True)
    ap.add_argument('--invert', action='store_true', help='answer the interior instead of the ends')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'artifacts/drone-verifier-20260919/probes'))
    a = ap.parse_args()
    G, _ = load_geometry(ROOT / 'artifacts/drone-flight-map-20260919/validation/meta.json')
    plan = json.loads(Path(a.plan).read_text())
    rows = []
    for f, boxes in plan['predictions_by_frame'].items():
        for b in boxes:
            if b['object_id'] == a.cls and b['confidence'] >= 0.8 and int(f) in G:
                g = to_ground(G[int(f)], (b['bbox'][0] + b['bbox'][2]) / 2 * W,
                              (b['bbox'][1] + b['bbox'][3]) / 2 * H)
                rows.append((int(f), g[0], g[1], b))
    want = set()
    for k in cluster(rows, 110.0):
        if k['n'] < 5:
            continue
        fs = sorted(f for f, _ in k['rows'])
        want |= set(fs[:a.n]) | set(fs[-a.n:])
    pbf = {}
    n = 0
    for f, boxes in plan['predictions_by_frame'].items():
        sel = [b for b in boxes if b['object_id'] == a.cls and b['confidence'] >= 0.8
               and ((int(f) in want) != a.invert)]
        if sel:
            pbf[f] = [dict(b) for b in sel]; n += len(sel)
    name = f'probe-{a.tag}'
    Path(a.out_dir, f'{name}.json').write_text(json.dumps({'name': name, 'target': [480, 270],
                                                           'predictions_by_frame': pbf}))
    print(f'{name}: {n} boxes over {len(pbf)} frames '
          f'({"interior" if a.invert else "track ends"}), ceiling if all hit {n / a.truth / 13:.5f} board')


if __name__ == '__main__':
    main()
