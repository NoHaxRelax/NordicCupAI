"""Find objects of the classes the team never labelled in the validation flight.

The portal scores validation over 13 classes, the pseudo-labels cover 11, so two of condor, jammer,
small_plane, spacecraft and ta-ta fly past unlabelled (worth 2/13 = 0.154 there, and present as
unlabelled "background" in every synthetic view cut from those frames). A detector that knows all 16
classes from the OTHER scene has no opinion about this one: run it on six L1 tiles per frame, keep
detections of the five classes, and link them with the flight's own motion map
(dx = 0.0074 (x - 1924), dy = 52.3 + 0.0147 y per frame). A real object is a long, consistent track.

    python elias/find_unlabelled.py --weights elias/out/weights/A_hel_to_val_s.best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIDDEN = ['condor', 'jammer', 'small_plane', 'spacecraft', 'ta-ta']
TILES = [(1, x, y) for y in (540, 1080, 1620) for x in (960, 1920, 2880)]


def step(x, y):
    return x+0.00741*(x-1924), y+52.3+0.01476*y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True); ap.add_argument('--conf', type=float, default=0.15)
    ap.add_argument('--min-len', type=int, default=6); ap.add_argument('--out', default=str(HERE/'out'/'hidden_tracks.json'))
    ap.add_argument('--classes', nargs='+', default=HIDDEN)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights); names = model.names
    frames = sorted((ROOT/'src'/'validation'/'images').glob('frame_*.png'))
    dets = {}
    for p in frames:
        if p.stat().st_size < 1_000_000:
            continue
        fr = int(p.stem.split('_')[1]); img = cv2.imread(str(p), cv2.IMREAD_COLOR); rows = []
        views = [cv2.resize(img[cy-540:cy+540, cx-960:cx+960], (960, 540), interpolation=cv2.INTER_AREA) for _, cx, cy in TILES]
        for (lvl, cx, cy), res in zip(TILES, model.predict(views, imgsz=960, conf=a.conf, verbose=False, device=0)):
            for x1, y1, x2, y2, s, c in res.boxes.data.cpu().tolist():
                if names[int(c)] in a.classes:
                    rows.append((names[int(c)], (x1+x2)+cx-960, (y1+y2)+cy-540, (x2-x1)*2, (y2-y1)*2, s))   # centre in source px
        dets[fr] = rows
    # greedy linking along the motion map, per class
    tracks = []
    for fr in sorted(dets):
        for cls, x, y, w, h, s in dets[fr]:
            best = None
            for t in tracks:
                if t['cls'] != cls or t['last'] >= fr or fr-t['last'] > 4:
                    continue
                px, py = t['x'], t['y']
                for _ in range(fr-t['last']):
                    px, py = step(px, py)
                d = np.hypot(px-x, py-y)
                if d < 25+6*(fr-t['last']) and (best is None or d < best[0]):
                    best = (d, t)
            if best:
                t = best[1]; t.update(x=x, y=y, last=fr); t['pts'].append((fr, x, y, w, h, s))
            else:
                tracks.append({'cls': cls, 'x': x, 'y': y, 'last': fr, 'pts': [(fr, x, y, w, h, s)]})
    keep = [t for t in tracks if len({p[0] for p in t['pts']}) >= a.min_len]
    keep.sort(key=lambda t: -len(t['pts']))
    print(f"{len(frames)} frames, {sum(len(v) for v in dets.values())} raw detections of {a.classes}, {len(tracks)} chains, {len(keep)} with >= {a.min_len} frames")
    sheet = []
    for k, t in enumerate(keep[:24]):
        fr0, fr1 = t['pts'][0][0], t['pts'][-1][0]; conf = np.mean([p[5] for p in t['pts']])
        print(f"  {t['cls']:<12} frames {fr0:>3}-{fr1:<3} ({len(t['pts'])} hits)  mean conf {conf:.2f}  x {t['pts'][0][1]:.0f}  size {np.median([p[3] for p in t['pts']]):.0f}x{np.median([p[4] for p in t['pts']]):.0f}")
        mid = t['pts'][len(t['pts'])//2]; img = cv2.imread(str(ROOT/'src'/'validation'/'images'/f'frame_{mid[0]:06d}.png'))
        x, y = int(mid[1]), int(mid[2]); r = int(max(60, 1.2*max(mid[3], mid[4])))
        crop = img[max(0, y-r):y+r, max(0, x-r):x+r]
        if crop.size:
            crop = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"{k} {t['cls']} f{mid[0]} n{len(t['pts'])}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1, cv2.LINE_AA); sheet.append(crop)
    if sheet:
        sheet += [np.zeros((200, 200, 3), np.uint8)]*((-len(sheet)) % 6)
        cv2.imwrite(str(Path(a.out).with_suffix('.jpg')), np.vstack([np.hstack(sheet[i:i+6]) for i in range(0, len(sheet), 6)]))
    Path(a.out).write_text(json.dumps([{'cls': t['cls'], 'pts': t['pts']} for t in keep], indent=0))


if __name__ == '__main__':
    main()
