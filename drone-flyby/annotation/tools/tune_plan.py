#!/usr/bin/env python3
"""Select classes from an answer plan and, if asked, hold one class back to a target AP.

The validation board must not show more than a chosen score, so the final plan is deliberately short of perfect:
one class's answers are truncated (its highest-confidence frames kept, the rest dropped) until its AP lands on the
target. With COCO AP at IoU .5 and hits ranked first, AP = (floor(100 h / T) + 1) / 101, so the number of frames to
keep follows from the class's truth count T.

    python3 tune_plan.py --plan probes/probe-full.json --out probes/probe-final.json \
        --classes all --degrade hangar --target-ap 0.48 --truth hangar=70
    python3 tune_plan.py --plan probes/probe-full.json --out probes/probe-half-a.json --classes hangar,tank,ta-ta
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

CLASSES = ['hangar', 'tank', 'helicopter', 'jet_plane', 'large_launcher', 'large_tower', 'mine_roller', 'small_tower',
           'small_plane', 'medium_plane', 'small_launcher', 'medium_launcher', 'ta-ta']


def ap_of(h, T):
    return (math.floor(100 * h / T) + 1) / 101 if h else 0.0


def hits_for(target, T):
    """Smallest hit count whose AP is at least the target (hits ranked first)."""
    for h in range(T + 1):
        if ap_of(h, T) >= target - 1e-9:
            return h
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default=None)
    ap.add_argument('--classes', default='all', help='comma list, or "all", or "all-<cls>" to drop one')
    ap.add_argument('--degrade', default=None, help='class to hold back')
    ap.add_argument('--target-ap', type=float, default=None)
    ap.add_argument('--truth', action='append', default=[], help='CLASS=T truth frame count, for the degrade maths')
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    pbf = plan['predictions_by_frame']
    if a.classes == 'all':
        keep = set(CLASSES)
    elif a.classes.startswith('all-'):
        keep = set(CLASSES) - {a.classes[4:]}
    else:
        keep = {c for c in a.classes.split(',') if c}
    truth = {k: int(v) for k, v in (t.split('=') for t in a.truth)}

    out = defaultdict(list)
    for f, rows in pbf.items():
        for o in rows:
            if o['object_id'] in keep:
                out[f].append(o)

    note = ''
    if a.degrade and a.target_ap is not None:
        cls = a.degrade
        T = truth.get(cls)
        if not T:
            raise SystemExit(f'--truth {cls}=<count> is needed to hold {cls} back')
        # one box per frame for this class, best confidence first; keep the frames that reach the target AP
        byf = {}
        for f, rows in out.items():
            best = max((o for o in rows if o['object_id'] == cls), key=lambda o: o['confidence'], default=None)
            if best is not None:
                byf[int(f)] = best['confidence']
        order = sorted(byf, key=lambda f: (-byf[f], f))
        h = hits_for(a.target_ap, T)
        keep_frames = set(order[:h])
        dropped = 0
        for f in list(out):
            if int(f) not in keep_frames:
                n = len(out[f]); out[f] = [o for o in out[f] if o['object_id'] != cls]; dropped += n - len(out[f])
        note = (f'{cls} held back to {len(keep_frames)} of {len(byf)} answered frames '
                f'(target AP {a.target_ap:.3f}, truth {T}, dropped {dropped} boxes)')
        print(note)

    out = {f: v for f, v in out.items() if v}
    doc = {'name': a.name or Path(a.out).stem, 'target': plan.get('target', [480, 270]),
           'predictions_by_frame': dict(sorted(out.items(), key=lambda kv: int(kv[0])))}
    if note:
        doc['note'] = note
    Path(a.out).write_text(json.dumps(doc))
    per = defaultdict(int)
    for rows in out.values():
        for o in rows:
            per[o['object_id']] += 1
    print(f'{Path(a.out).name}: {len(out)} frames, {sum(per.values())} boxes, {len(per)} classes')
    for c in CLASSES:
        if per.get(c):
            print(f'   {c:16s} {per[c]:5d}')


if __name__ == '__main__':
    main()
