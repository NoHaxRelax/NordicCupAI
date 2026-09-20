#!/usr/bin/env python3
"""Measure every instance's real extent in the frames, so the answer boxes stop inheriting the tracker's size prior.

The pipeline blends the detector box with a prior fitted on the other scene, which makes the boxes 25-30 % too large
for several classes: measured by eye on an 8x grid, large tower 1 is 28x55 px where the answer is 47x67, and the
IoU against a tight truth box lands at 0.49 - just under the 0.5 the organiser needs. This runs on the pod, where
the frames are, and segments the object around each answer box:

  * crop a window 2.6x the answer box around its centre
  * model the background from the border ring of that window (median Lab colour and spread)
  * mark pixels far from the background, close the holes, and keep the blob nearest the centre
  * the blob's bounding box is the measurement; the median over several frames of the instance is the answer

Output: JSON of {class: [{ground, frames, n, offset, scale, obj, ours}]} with the offset and the scale to apply to
the tracker's box. Frames where the box touches the border are skipped.

    python3 measure_extents.py --frames /root/data/reconstructed-validation --boxes boxes.json --out extents.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np


def measure(img, box, grow=2.6):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(w, h) * grow / 2
    ix0, iy0 = int(round(cx - r)), int(round(cy - r))
    ix1, iy1 = int(round(cx + r)), int(round(cy + r))
    if ix0 < 0 or iy0 < 0 or ix1 > img.shape[1] or iy1 > img.shape[0]:
        return None
    crop = img[iy0:iy1, ix0:ix1]
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(2, int(0.12 * crop.shape[0]))
    ring = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                           lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    spread = np.median(np.abs(ring - bg), axis=0) + 1e-3
    d = np.linalg.norm((lab - bg) / np.maximum(spread, 2.0), axis=2)
    mask = (d > 4.0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    ccx, ccy = crop.shape[1] / 2, crop.shape[0] / 2
    best, bestscore = None, None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < max(12, 0.02 * w * h):
            continue
        dist = np.hypot(cent[i][0] - ccx, cent[i][1] - ccy)
        if dist > 0.45 * crop.shape[0]:
            continue
        score = a / (1 + dist)                       # big and central
        if bestscore is None or score > bestscore:
            best, bestscore = i, score
    if best is None:
        return None
    # merge any blob that overlaps the winner's rows and columns (a tower's slab, a launcher's arms)
    bx, by = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP]
    bw, bh = stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT]
    for i in range(1, n):
        if i == best or stats[i, cv2.CC_STAT_AREA] < 8:
            continue
        ox, oy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ow, oh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if ox < bx + bw + 3 and ox + ow > bx - 3 and oy < by + bh + 3 and oy + oh > by - 3:
            nx0, ny0 = min(bx, ox), min(by, oy)
            bx, by = nx0, ny0
            bw, bh = max(bx + bw, ox + ow) - nx0, max(by + bh, oy + oh) - ny0
    obj = [ix0 + bx, iy0 + by, ix0 + bx + bw, iy0 + by + bh]
    return {'obj': obj, 'w': bw, 'h': bh, 'dx': (obj[0] + obj[2]) / 2 - cx, 'dy': (obj[1] + obj[3]) / 2 - cy}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--boxes', required=True, help='JSON: {class: [[instance_id, frame, x0,y0,x1,y1], ...]}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-instance', type=int, default=9)
    a = ap.parse_args()
    data = json.loads(open(a.boxes).read())
    cache = {}
    out = defaultdict(list)
    for cls, rows in data.items():
        byinst = defaultdict(list)
        for iid, f, x0, y0, x1, y1 in rows:
            byinst[iid].append((f, [x0, y0, x1, y1]))
        for iid, items in sorted(byinst.items()):
            items.sort()
            step = max(1, len(items) // a.per_instance)
            picks = items[::step][:a.per_instance]
            meas = []
            for f, box in picks:
                if f not in cache:
                    if len(cache) > 24:
                        cache.clear()
                    cache[f] = cv2.imread(f'{a.frames}/frame_{f:06d}.png', cv2.IMREAD_COLOR)
                img = cache[f]
                if img is None:
                    continue
                m = measure(img, box)
                if m:
                    m['frame'] = f; m['ours'] = [box[2] - box[0], box[3] - box[1]]
                    meas.append(m)
            if len(meas) < 3:
                out[cls].append({'instance': iid, 'n': len(meas), 'status': 'too few measurements'})
                continue
            sx = float(np.median([m['w'] / m['ours'][0] for m in meas]))
            sy = float(np.median([m['h'] / m['ours'][1] for m in meas]))
            dx = float(np.median([m['dx'] for m in meas]))
            dy = float(np.median([m['dy'] for m in meas]))
            out[cls].append({'instance': iid, 'n': len(meas), 'frames': [meas[0]['frame'], meas[-1]['frame']],
                             'scale': [round(sx, 3), round(sy, 3)], 'offset': [round(dx, 1), round(dy, 1)],
                             'obj_median': [round(float(np.median([m['w'] for m in meas])), 1),
                                            round(float(np.median([m['h'] for m in meas])), 1)],
                             'ours_median': [round(float(np.median([m['ours'][0] for m in meas])), 1),
                                             round(float(np.median([m['ours'][1] for m in meas])), 1)]})
    json.dump(out, open(a.out, 'w'), indent=1)
    for cls, items in out.items():
        for it in items:
            if it.get('status'):
                print(f"{cls:16s} inst {it['instance']}: {it['status']}")
            else:
                print(f"{cls:16s} inst {it['instance']} n={it['n']:2d} frames {it['frames']}  object {it['obj_median']} "
                      f"vs ours {it['ours_median']}  scale {it['scale']} offset {it['offset']}")


if __name__ == '__main__':
    main()
