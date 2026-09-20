#!/usr/bin/env python3
"""Thin a plan's per-frame alternates so the extra candidates stop outranking the hits underneath them.

COCO sorts a class's whole detection list by confidence, so a band of alternates that misses sits ahead of
every hit in the bands below it and cuts their precision. Where two candidates in one frame are the same
object at different sizes, only the top one can ever be the true positive; the rest are pure false positives.
This keeps, per frame and class, the highest-confidence box of each overlapping group.

    python3 prune_plan.py --plan probe-full.json --out probe-pruned.json --iou 0.1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Tracks confirmed false and dropped outright. The jet_plane track that crosses frames 40-52 climbs the image
# while every ground object descends; rendering those frames shows forest canopy, no aircraft, so v8's boxes
# there are a false positive on tree shadow.
FALSE_TRACKS = [('jet_plane', 40, 52, (1240, 0, 1470, 300))]


def in_track(cls, f, bb):
    for c, f0, f1, (x0, y0, x1, y1) in FALSE_TRACKS:
        if c == cls and f0 <= f <= f1:
            cx, cy = (bb[0] + bb[2]) / 2 * 3840, (bb[1] + bb[3]) / 2 * 2160
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
    return False


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iou', type=float, default=0.1, help='suppress a box overlapping a kept one by more than this')
    ap.add_argument('--floor', type=float, default=0.0, help='drop boxes under this confidence')
    ap.add_argument('--classes', default=None, help='comma list; default every class')
    ap.add_argument('--iou-per-class', default='small_launcher=0.4',
                    help='comma list CLS=IOU; the small launchers sit close enough that boxes 23 px wide '
                         'overlap at different positions, so they need a looser threshold than the rest')
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    only = set(a.classes.split(',')) if a.classes else None
    per_cls = dict((k, float(v)) for k, v in (t.split('=') for t in a.iou_per_class.split(',') if t))
    pbf, kept, dropped = {}, 0, 0
    for f, boxes in plan['predictions_by_frame'].items():
        out = []
        for cls, group in sorted(defaultdict(list, {c: [b for b in boxes if b['object_id'] == c]
                                                    for c in {b['object_id'] for b in boxes}}).items()):
            if only is not None and cls not in only:
                out.extend(group); continue
            keep = []
            for b in sorted(group, key=lambda d: -d['confidence']):
                if in_track(cls, int(f), b['bbox']):
                    dropped += 1; continue
                if b['confidence'] < a.floor:
                    dropped += 1; continue
                if any(iou(b['bbox'], k['bbox']) > per_cls.get(cls, a.iou) for k in keep):
                    dropped += 1; continue
                keep.append(b)
            out.extend(keep); kept += len(keep)
        if out:
            pbf[f] = out
    plan['predictions_by_frame'] = pbf
    plan['name'] = Path(a.out).stem
    Path(a.out).write_text(json.dumps(plan))
    per = defaultdict(int)
    for boxes in pbf.values():
        for b in boxes:
            per[b['object_id']] += 1
    print(f'kept {sum(len(v) for v in pbf.values())} boxes over {len(pbf)} frames, dropped {dropped}')
    for c, n in sorted(per.items()):
        print(f'  {c:16s} {n:5d}')


if __name__ == '__main__':
    main()
