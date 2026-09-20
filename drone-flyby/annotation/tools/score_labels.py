#!/usr/bin/env python3
"""Score a replay's EMITTED answer against the mined validation labels, offline (no API).

Truth comes from the two label files the mining sessions produced:

  probe-anchored-labels.json  {"classes": {<class>: {<frame>: [{"bbox_source_xyxy": [...], "status": ...}]}}}
      source pixels, 1-based frames. status 'accepted' = a scripted attempt holding exactly those boxes
      scored as if every box matched, so each box is within IoU 0.5 of the organiser's truth box.
  mined-labels-v2.json        {"labels_by_frame": {<frame>: [{"object_id": ..., "bbox": [x0,y0,x1,y1]}]}}
      bbox normalised against 3840x2160.

Matching follows the organiser's convention: per frame, greedy by confidence, one emitted box per truth
box, a hit at IoU >= 0.5. Because an 'accepted' truth box is itself only known to within IoU 0.5 of the
real one, a genuine improvement should land well above the threshold - the per-frame IoU column is the
number to read, not just the hit count.

    python3 score_labels.py replays/baseline replays/adapt --truth probe-anchored-labels.json \
        --classes hangar --frames 53:143 --per-frame
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

W, H = 3840, 2160


def iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.
    inter = (x1-x0)*(y1-y0)
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter/area) if area > 0 else 0.


def load_truth(path, statuses=('accepted', 'accepted-most', 'live', 'estimate')):
    """-> {class: {frame: [box in source px]}}, plus the status of each box."""
    raw = json.loads(Path(path).read_text())
    out, status = {}, {}
    if 'classes' in raw:
        for label, frames in raw['classes'].items():
            for frame, boxes in frames.items():
                for item in boxes:
                    if item.get('status', 'accepted') not in statuses:
                        continue
                    out.setdefault(label, {}).setdefault(int(frame), []).append(list(item['bbox_source_xyxy']))
                    status[label] = item.get('status')
    elif 'labels_by_frame' in raw:
        for frame, boxes in raw['labels_by_frame'].items():
            for item in boxes:
                b = item['bbox']
                out.setdefault(item['object_id'], {}).setdefault(int(frame), []).append(
                    [b[0]*W, b[1]*H, b[2]*W, b[3]*H])
    else:
        raise SystemExit(f'{path}: unknown label format')
    return out, status


def load_emitted(folder):
    """-> {frame: [(class, box in source px, confidence)]} from the endpoint's own jsonl log."""
    rows = {}
    for f in sorted(Path(folder).glob('*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if 'response' not in r or 'frame' not in r:
                continue
            boxes = []
            for a in r['response'] or []:
                b = a['bbox']
                boxes.append((a['object_id'], [b[0]*W, b[1]*H, b[2]*W, b[3]*H], float(a.get('confidence', 1.))))
            rows[int(r['frame'])] = boxes
    return rows


def match_frame(truth_boxes, emitted, threshold):
    """Greedy by confidence. -> [(truth index, iou or 0)], plus the count of unmatched emitted boxes."""
    taken = {}
    for _, box, _ in sorted(emitted, key=lambda e: -e[2]):
        best, best_iou = None, 0.
        for i, t in enumerate(truth_boxes):
            if i in taken:
                continue
            v = iou(box, t)
            if v > best_iou:
                best, best_iou = i, v
        if best is not None and best_iou >= threshold:
            taken[best] = best_iou
    return taken


def score(emitted_rows, truth, label, frames, threshold):
    per_frame = []
    for frame in frames:
        truth_boxes = truth.get(label, {}).get(frame, [])
        mine = [e for e in emitted_rows.get(frame, []) if e[0] == label]
        taken = match_frame(truth_boxes, mine, threshold)
        # best IoU per truth box regardless of the threshold, for diagnosis
        best = []
        for t in truth_boxes:
            best.append(max((iou(b, t) for _, b, _ in mine), default=0.))
        per_frame.append({'frame': frame, 'truth': len(truth_boxes), 'emitted': len(mine),
                          'hits': len(taken), 'best_iou': best})
    return per_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('replays', nargs='+', help='replay folders holding the endpoint jsonl log')
    ap.add_argument('--truth', required=True)
    ap.add_argument('--classes', default='', help='comma list; default = every class in the truth file')
    ap.add_argument('--frames', default='', help='a:b inclusive frame window; default = every truth frame')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--statuses', default='accepted', help='which label statuses to trust (anchored file only)')
    ap.add_argument('--per-frame', action='store_true')
    a = ap.parse_args()

    truth, _ = load_truth(a.truth, tuple(s.strip() for s in a.statuses.split(',') if s.strip()))
    labels = [c.strip() for c in a.classes.split(',') if c.strip()] or sorted(truth)
    runs = {Path(r).name: load_emitted(r) for r in a.replays}

    for label in labels:
        frames = sorted(truth.get(label, {}))
        if a.frames:
            lo, hi = (int(v) for v in a.frames.split(':'))
            frames = [f for f in frames if lo <= f <= hi]
        if not frames:
            continue
        total = sum(len(truth[label][f]) for f in frames)
        print(f'\n== {label}: {total} truth boxes over {len(frames)} frames '
              f'({frames[0]}-{frames[-1]}), IoU >= {a.iou}')
        results = {}
        for name, rows in runs.items():
            per = score(rows, truth, label, frames, a.iou)
            hits = sum(p['hits'] for p in per)
            ious = [v for p in per for v in p['best_iou']]
            extra = sum(max(0, p['emitted']-p['truth']) for p in per)
            results[name] = per
            print(f'  {name:28s} hits {hits:3d}/{total:3d}  mean IoU {np.mean(ious):.3f}  '
                  f'median {np.median(ious):.3f}  min {np.min(ious):.3f}  extra boxes {extra}')
        if a.per_frame:
            names = list(runs)
            print('  frame | ' + ' | '.join(f'{n[:22]:>22s}' for n in names))
            for i, frame in enumerate(frames):
                cells = []
                for n in names:
                    p = results[n][i]
                    mark = 'HIT ' if p['hits'] == p['truth'] else 'MISS'
                    cells.append(f"{mark} iou {' '.join(f'{v:.2f}' for v in p['best_iou']):>10s}")
                if any('MISS' in c for c in cells) or a.per_frame:
                    print(f'  {frame:5d} | ' + ' | '.join(f'{c:>22s}' for c in cells))


if __name__ == '__main__':
    main()
