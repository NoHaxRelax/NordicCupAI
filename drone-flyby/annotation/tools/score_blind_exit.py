#!/usr/bin/env python3
"""How far does the hangar's blind forecast drift off the object, measured against an independent truth.

The anchored labels cannot answer this for frames 111-142: those 'accepted' boxes ARE the old pipeline's
own output (they scored as hits, which only proves they sit within IoU 0.5 of the truth), so scoring a
forecast change against them rewards reproducing the bug. The independent measurement is
exit-analysis/exit-measure.json: the deployed detector run on the L2 and L1 views the camera could
legally have taken at each of frames 120-143, i.e. where the hangar actually was while the sweep was
looking elsewhere. Frame 143 has no detection (the object is a strip at the bottom edge); there the
organiser-bisected truth from probe-anchored-labels.json is used instead.

Reported per frame: centre error, top and bottom edge error, and IoU, of the EMITTED box against that
truth. Extent policy shifts the edges, so centre and bottom error are the forecast-only numbers.

    python3 score_blind_exit.py replays/hx-base replays/hx-adapt --measure exit-measure.json \
        --anchored probe-anchored-labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

W, H = 3840, 2160


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.
    inter = (x1-x0)*(y1-y0)
    return float(inter/((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter))


def load_emitted(folder, label='hangar'):
    out = {}
    for f in sorted(Path(folder).glob('*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if 'response' not in r or 'frame' not in r:
                continue
            best = None
            for a in r['response'] or []:
                if a['object_id'] != label:
                    continue
                b = a['bbox']
                box = [b[0]*W, b[1]*H, b[2]*W, b[3]*H]
                if best is None or a.get('confidence', 1) > best[1]:
                    best = (box, a.get('confidence', 1))
            if best:
                out[int(r['frame'])] = best[0]
    return out


def truth_boxes(measure_path, anchored_path, prefer='L2'):
    """-> {frame: (box, source)}. Detector-on-legal-view where it exists, the bisected strip at 143."""
    measure = json.loads(Path(measure_path).read_text())
    out = {}
    for frame, row in measure.items():
        box = row.get(prefer) or row.get('L1')
        if box:
            out[int(frame)] = (list(box[:4]), prefer if row.get(prefer) else 'L1')
    anchored = json.loads(Path(anchored_path).read_text())['classes']['hangar']
    for frame, items in anchored.items():
        f = int(frame)
        if f not in out and items and items[0].get('status') == 'accepted':
            out[f] = (list(items[0]['bbox_source_xyxy']), 'anchored')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('replays', nargs='+')
    ap.add_argument('--measure', required=True)
    ap.add_argument('--anchored', required=True)
    ap.add_argument('--first', type=int, default=127, help='first blind frame (last full sighting is 126/127)')
    ap.add_argument('--last', type=int, default=143)
    a = ap.parse_args()

    truth = truth_boxes(a.measure, a.anchored)
    runs = {Path(r).name: load_emitted(r) for r in a.replays}
    frames = [f for f in sorted(truth) if a.first <= f <= a.last]

    print(f'truth: detector on legal L2/L1 views, {a.first}-142; organiser-bisected strip at 143')
    header = 'frame src  ' + '  '.join(f'{n[:20]:>34s}' for n in runs)
    print(header)
    print(' ' * 11 + '  '.join(f'{"dcy   dtop  dbot   IoU":>34s}' for _ in runs))
    totals = {n: {'dcy': [], 'dtop': [], 'dbot': [], 'iou': []} for n in runs}
    for f in frames:
        t, src = truth[f]
        cells = []
        for n, rows in runs.items():
            box = rows.get(f)
            if box is None:
                cells.append(f'{"(no box)":>34s}'); continue
            dcy = ((box[1]+box[3])/2)-((t[1]+t[3])/2)
            dtop = box[1]-t[1]; dbot = box[3]-t[3]; v = iou(box, t)
            totals[n]['dcy'].append(dcy); totals[n]['dtop'].append(dtop)
            totals[n]['dbot'].append(dbot); totals[n]['iou'].append(v)
            cells.append(f'{dcy:+7.1f} {dtop:+6.1f} {dbot:+6.1f} {v:6.2f}'.rjust(34))
        print(f'{f:5d} {src:4s} ' + '  '.join(cells))
    print('\nmean |error| over the blind stretch (px), and mean IoU:')
    for n, t in totals.items():
        if not t['iou']:
            continue
        print(f'  {n:24s} |dcy| {np.mean(np.abs(t["dcy"])):6.1f}  |dtop| {np.mean(np.abs(t["dtop"])):6.1f}  '
              f'|dbot| {np.mean(np.abs(t["dbot"])):6.1f}  IoU {np.mean(t["iou"]):.3f}')


if __name__ == '__main__':
    main()
