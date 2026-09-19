"""Two contact sheets for judging unlabelled-object candidates by eye.

  ref      every class as it appears in the Helsinki reference frames (organiser boxes), native pixels at 2x
  chains   selected chains of elias/out/unlabelled2_<tag>.json at their first, middle and last frame, 3x zoom

    python elias/candidate_sheets.py ref
    python elias/candidate_sheets.py chains --tag hel --ids 0 1 2 3 5 7 8 9 15 17 22
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def tile(img, cx, cy, r, zoom, text, box=None):
    x0, y0 = int(max(0, cx-r)), int(max(0, cy-r)); crop = img[y0:int(cy+r), x0:int(cx+r)]
    if not crop.size:
        return np.zeros((int(2*r*zoom), int(2*r*zoom), 3), np.uint8)
    crop = cv2.resize(crop, (int(crop.shape[1]*zoom), int(crop.shape[0]*zoom)), interpolation=cv2.INTER_CUBIC)
    if box is not None:
        cv2.rectangle(crop, (int((box[0]-x0)*zoom), int((box[1]-y0)*zoom)), (int((box[2]-x0)*zoom), int((box[3]-y0)*zoom)), (0, 255, 255), 1)
    side = int(2*r*zoom); pad = np.zeros((side, side, 3), np.uint8); h, w = min(side, crop.shape[0]), min(side, crop.shape[1]); pad[:h, :w] = crop[:h, :w]
    cv2.putText(pad, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
    return pad


def cmd_ref(a):
    best = {}
    for f in sorted((ROOT/'src'/'helsinki'/'annotations').glob('*.json')):
        for x in json.loads(f.read_text())['annotations']:
            b = x['bbox']; area = (b[2]-b[0])*(b[3]-b[1])
            if b[0] > 5 and b[1] > 5 and b[2] < 3835 and b[3] < 2155 and (x['object_id'] not in best or area > best[x['object_id']][0]):
                best[x['object_id']] = (area, f.stem, b)
    tiles = []
    for cls in sorted(best):
        area, stem, b = best[cls]; img = cv2.imread(str(ROOT/'src'/'helsinki'/'images'/f'{stem}.png'))
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; r = max(60, 0.9*max(b[2]-b[0], b[3]-b[1]))
        t = tile(img, cx, cy, r, 240/(2*r), f"{cls} {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}", b)
        tiles.append(cv2.resize(t, (240, 240)))
    tiles += [np.zeros((240, 240, 3), np.uint8)]*((-len(tiles)) % 4)
    cv2.imwrite(str(HERE/'out'/'ref_classes.jpg'), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    print('elias/out/ref_classes.jpg', sorted(best))


def cmd_chains(a):
    T = json.loads((HERE/'out'/f'unlabelled2_{a.tag}.json').read_text())
    rows = []
    for k in a.ids:
        t = T[k]; pts = t['pts']; picks = [pts[0], pts[len(pts)//2], pts[-1]]
        row = []
        for p in picks:
            fr, x, y, w, h = p[0], p[1], p[2], p[3], p[4]
            img = cv2.imread(str(ROOT/'src'/'validation'/'images'/f'frame_{fr:06d}.png'))
            row.append(tile(img, x, y, 80, 3, f"{k} {t['cls'][:12]} f{fr} {w:.0f}x{h:.0f} c{p[5]:.2f}", [x-w/2, y-h/2, x+w/2, y+h/2]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(HERE/'out'/f'chains_{a.tag}.jpg'), np.vstack(rows))
    print(f'elias/out/chains_{a.tag}.jpg')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['ref', 'chains']); ap.add_argument('--tag', default='hel'); ap.add_argument('--ids', type=int, nargs='*', default=[])
    a = ap.parse_args()
    {'ref': cmd_ref, 'chains': cmd_chains}[a.cmd](a)
