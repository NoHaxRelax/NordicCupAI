"""Second search for the classes the team never labelled in the validation flight, with the night's detectors.

Differences from find_unlabelled.py: native L2 tiles (25 per frame, two overlapping grids) plus six L1 tiles,
inference at 1280, detections that sit on a pseudo-labelled object or inside a known keep-out zone are dropped so
only UNEXPLAINED objects remain, linking is class-agnostic along the flight's motion field, and every chain
reports its label mix. Anything on the ground tracks perfectly along the field, so a long chain proves only that
the detector fires consistently on that spot; the contact sheet and a portal replay decide.

    python elias/find_unlabelled2.py --weights elias/release/helsinki_only_m1280.pt --tag hel
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
L2 = [(2, cx, cy) for cy in (270, 810, 1350, 1890) for cx in (480, 1440, 2400, 3360)] + \
     [(2, cx, cy) for cy in (540, 1080, 1620) for cx in (960, 1920, 2880)]
L1 = [(1, cx, cy) for cy in (540, 1620) for cx in (960, 1920, 2880)]


def step(x, y):
    return x+0.00741*(x-1924), y+52.3+0.01476*y


def iou(a, b):
    ix = max(0., min(a[2], b[2])-max(a[0], b[0])); iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--conf', type=float, default=0.05); ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--min-len', type=int, default=5); ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--classes', nargs='+', default=None, help='keep only these labels (default all)')
    ap.add_argument('--top', type=int, default=36)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights); names = model.names
    hidden = json.loads((HERE/'data'/'validation_hidden.json').read_text())['zones']
    frames = [p for p in sorted((ROOT/'src'/'validation'/'images').glob('frame_*.png')) if p.stat().st_size > 1_000_000][::a.step]
    dets, t0 = {}, time.time()
    for k, p in enumerate(frames):
        fr = int(p.stem.split('_')[1]); img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        ann = json.loads((ROOT/'src'/'validation'/'annotations'/f'{p.stem}.json').read_text())['annotations']
        keep = [np.array(x['bbox'], float) for x in ann]+([np.array(hidden[str(fr)], float)] if str(fr) in hidden else [])
        tiles = L2+L1; views = []
        for lvl, cx, cy in tiles:
            f = 1 if lvl == 2 else 2
            crop = img[cy-270*f:cy+270*f, cx-480*f:cx+480*f]
            views.append(crop if f == 1 else cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA))
        rows = []
        for i in range(0, len(views), 8):
            for (lvl, cx, cy), res in zip(tiles[i:i+8], model.predict(views[i:i+8], imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)):
                f = 1 if lvl == 2 else 2
                for x1, y1, x2, y2, s, c in res.boxes.data.cpu().tolist():
                    if min(x1, y1) < 2 or x2 > 958 or y2 > 538:
                        continue      # cut by the tile edge: another tile holds it whole
                    rows.append((names[int(c)], np.array([x1*f+cx-480*f, y1*f+cy-270*f, x2*f+cx-480*f, y2*f+cy-270*f]), float(s), lvl))
        out = []
        for cls, b, s, lvl in rows:
            c = (b[:2]+b[2:])/2
            if any(kb[0]-8 <= c[0] <= kb[2]+8 and kb[1]-8 <= c[1] <= kb[3]+8 for kb in keep):
                continue      # explained by a team label or a known keep-out zone
            if a.classes and cls not in a.classes:
                continue
            out.append((cls, b, s, lvl))
        out.sort(key=lambda r: -r[2]); final = []
        for r in out:
            if all(iou(r[1], q[1]) < 0.5 for q in final):
                final.append(r)
        dets[fr] = final
        if k % 25 == 0:
            print(f'{k}/{len(frames)} frame {fr}: {len(rows)} raw, {len(final)} unexplained  {time.time()-t0:.0f}s', flush=True)
    tracks = []
    for fr in sorted(dets):
        for cls, b, s, lvl in dets[fr]:
            x, y = float((b[0]+b[2])/2), float((b[1]+b[3])/2); w, h = float(b[2]-b[0]), float(b[3]-b[1]); best = None
            for t in tracks:
                gap = fr-t['last']
                if gap <= 0 or gap > 4:
                    continue
                px, py = t['x'], t['y']
                for _ in range(gap):
                    px, py = step(px, py)
                d = float(np.hypot(px-x, py-y))
                if d < 20+6*gap and (best is None or d < best[0]):
                    best = (d, t)
            if best:
                t = best[1]; t.update(x=x, y=y, last=fr); t['pts'].append((fr, x, y, w, h, s, cls, lvl))
            else:
                tracks.append({'x': x, 'y': y, 'last': fr, 'pts': [(fr, x, y, w, h, s, cls, lvl)]})
    keep_t = [t for t in tracks if len({p[0] for p in t['pts']}) >= a.min_len]
    for t in keep_t:
        t['n'] = len({p[0] for p in t['pts']}); t['conf'] = float(np.mean([p[5] for p in t['pts']]))
        t['labels'] = dict(Counter(p[6] for p in t['pts']).most_common()); t['score'] = t['n']*t['conf']
        t['cls'] = next(iter(t['labels']))
    keep_t.sort(key=lambda t: -t['score'])
    print(f"{len(frames)} frames, {sum(len(v) for v in dets.values())} unexplained detections, {len(tracks)} chains, {len(keep_t)} with >= {a.min_len} frames")
    sheet = []
    for k, t in enumerate(keep_t[:a.top]):
        p0, p1 = t['pts'][0], t['pts'][-1]; mid = t['pts'][len(t['pts'])//2]
        print(f"  {k:2d} {t['cls']:<14} frames {p0[0]:>3}-{p1[0]:<3} n {t['n']:2d} conf {t['conf']:.2f} at x {mid[1]:.0f} y {mid[2]:.0f} size {np.median([p[3] for p in t['pts']]):.0f}x{np.median([p[4] for p in t['pts']]):.0f}  labels {t['labels']}")
        img = cv2.imread(str(ROOT/'src'/'validation'/'images'/f'frame_{mid[0]:06d}.png'))
        x, y = int(mid[1]), int(mid[2]); r = int(max(70, 1.6*max(mid[3], mid[4])))
        ox, oy = max(0, x-r), max(0, y-r); crop = img[oy:y+r, ox:x+r].copy()
        if crop.size:
            sc = 240/crop.shape[0]; crop = cv2.resize(crop, (max(1, int(crop.shape[1]*sc)), 240), interpolation=cv2.INTER_CUBIC)
            bx, by = int((mid[1]-mid[3]/2-ox)*sc), int((mid[2]-mid[4]/2-oy)*sc)
            cv2.rectangle(crop, (bx, by), (int(bx+mid[3]*sc), int(by+mid[4]*sc)), (0, 255, 255), 1)
            pad = np.zeros((240, 240, 3), np.uint8); w = min(240, crop.shape[1]); pad[:, :w] = crop[:, :w]
            cv2.putText(pad, f"{k} {t['cls'][:12]} f{mid[0]} n{t['n']} c{t['conf']:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1, cv2.LINE_AA); sheet.append(pad)
    if sheet:
        sheet += [np.zeros((240, 240, 3), np.uint8)]*((-len(sheet)) % 6)
        cv2.imwrite(str(HERE/'out'/f'unlabelled2_{a.tag}.jpg'), np.vstack([np.hstack(sheet[i:i+6]) for i in range(0, len(sheet), 6)]))
    (HERE/'out'/f'unlabelled2_{a.tag}.json').write_text(json.dumps(keep_t, indent=0))


if __name__ == '__main__':
    main()
