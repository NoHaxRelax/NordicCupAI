#!/usr/bin/env python3
"""Assemble the final plan by taking each class from whichever variant scored best on its own attempt.

Classes were measured one at a time, so the board total is the mean of those per-class results and the best
plan is simply the best variant of each class put together. Nothing about one class's boxes can change
another's score: COCO computes AP per class.

    python3 merge_best.py --pick medium_launcher=probe-drawn.json --default probe-pruned.json --out final.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['hangar', 'tank', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower', 'mine_roller',
           'small_tower', 'small_plane', 'medium_plane', 'small_launcher', 'medium_launcher', 'ta-ta']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--default', required=True, help='plan every class comes from unless picked otherwise')
    ap.add_argument('--pick', action='append', default=[], help='CLASS=plan.json, repeatable')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base = Path(a.default)
    src = {c: base for c in CLASSES}
    for p in a.pick:
        c, f = p.split('=', 1)
        src[c] = Path(f) if Path(f).exists() else base.parent / f
    cache = {}
    for p in set(src.values()):
        cache[p] = json.loads(p.read_text())
    pbf, per = {}, Counter()
    for cls in CLASSES:
        plan = cache[src[cls]]
        for f, boxes in plan['predictions_by_frame'].items():
            keep = [dict(b) for b in boxes if b['object_id'] == cls]
            if keep:
                pbf.setdefault(f, []).extend(keep)
                per[cls] += len(keep)
    out = {'name': Path(a.out).stem, 'target': [480, 270],
           'predictions_by_frame': dict(sorted(pbf.items(), key=lambda kv: int(kv[0])))}
    Path(a.out).write_text(json.dumps(out))
    print(f'{a.out}: {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames, '
          f'max {max(len(v) for v in pbf.values())} per frame')
    for cls in CLASSES:
        print(f'  {cls:16s} {per[cls]:5d}  from {src[cls].name}')


if __name__ == '__main__':
    main()
