"""What terrain sits under the objects? Elias and Oscar's claim: objects spawn in plausible places (tanks on roads,
most things on grass or paved ground, nothing on water or dense forest). If true, the synthetic generator should
stop pasting sprites on water and forest (it teaches the detector that dark blobs in trees are launchers), and a
runtime context prior can down-weight detections over water and forest.

Every labelled object (Helsinki organiser boxes, validation pseudo-labels every 4th frame, the objects our search
found) gets its background ring classified by simple colour rules on the 4K frame: water (blue or very dark and
flat), forest (dark green with strong texture), grass (green, smoother), paved (grey, low saturation), sand/bare
(bright, low saturation, warm). The same rules on random frame locations give the prior. A contact sheet shows
six backdrops per class for a human check.

    python elias/backdrop_study.py
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TERRAIN = ['water', 'forest', 'grass', 'paved', 'sand', 'other']


def classify(ring_bgr):
    hsv = cv2.cvtColor(ring_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    h, s, v = hsv[:, 0]*2, hsv[:, 1]/255, hsv[:, 2]/255          # hue in degrees
    gray = cv2.cvtColor(ring_bgr, cv2.COLOR_BGR2GRAY)
    tex = float(np.std(cv2.Laplacian(gray, cv2.CV_32F)))
    green = np.mean((h > 60) & (h < 170) & (s > 0.2) & (v > 0.1))
    blue = np.mean((h > 185) & (h < 250) & (s > 0.25))
    grey = np.mean((s < 0.18) & (v > 0.25))
    warm = np.mean((h < 60) & (s > 0.12) & (v > 0.45))
    mv, ms = float(np.mean(v)), float(np.mean(s))
    if blue > 0.45 or (mv < 0.22 and tex < 6):
        return 'water', tex
    if green > 0.5 and (mv < 0.42 or tex > 22):
        return 'forest', tex
    if green > 0.45:
        return 'grass', tex
    if grey > 0.5:
        return 'paved', tex
    if warm > 0.4 or (ms < 0.3 and mv > 0.55):
        return 'sand', tex
    return 'other', tex


def ring(img, box, grow=1.0):
    x1, y1, x2, y2 = [int(v) for v in box]; w, h = x2-x1, y2-y1
    X1, Y1, X2, Y2 = max(0, int(x1-grow*w)), max(0, int(y1-grow*h)), min(img.shape[1], int(x2+grow*w)), min(img.shape[0], int(y2+grow*h))
    patch = img[Y1:Y2, X1:X2].copy()
    patch[y1-Y1:y2-Y1, x1-X1:x2-X1] = 0          # blank the object itself
    pix = patch.reshape(-1, 3); pix = pix[pix.sum(1) > 0]
    return pix.reshape(-1, 1, 3) if len(pix) else None, img[Y1:Y2, X1:X2]


def main():
    rng = random.Random(0)
    objs = []      # (class, scene, frame, box)
    for f in sorted((ROOT/'src'/'helsinki'/'annotations').glob('*.json')):
        for a in json.loads(f.read_text())['annotations']:
            objs.append((a['object_id'], 'helsinki', int(f.stem.split('_')[1]), a['bbox']))
    for f in sorted((ROOT/'src'/'validation'/'annotations').glob('*.json'))[::4]:
        for a in json.loads(f.read_text())['annotations']:
            objs.append((a['object_id'], 'validation', int(f.stem.split('_')[1]), a['bbox']))
    h2 = HERE/'data'/'validation_hidden2.json'
    if h2.exists():
        chains = {t: json.loads((HERE/'out'/f'unlabelled2_{t}.json').read_text()) for t in ('hel', 'both')}
        for t in json.loads(h2.read_text())['tracks']:
            if t['what'] == 'dark blob':
                continue
            pts = chains[t['search']][t['chain']]['pts'][::5]
            for p in pts:
                objs.append((t['what'], 'validation', p[0], [p[1]-p[3]/2, p[2]-p[4]/2, p[1]+p[3]/2, p[2]+p[4]/2]))
    by_frame = defaultdict(list)
    for o in objs:
        by_frame[(o[1], o[2])].append(o)
    per_class = defaultdict(Counter); prior = Counter(); tiles = defaultdict(list); n_obj = 0
    for (scene, fr), items in sorted(by_frame.items()):
        p = ROOT/'src'/scene/'images'/f'frame_{fr:06d}.png'
        if not p.exists() or p.stat().st_size < 1_000_000:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        for cls, _, _, box in items:
            pix, crop = ring(img, box)
            if pix is None:
                continue
            terrain, tex = classify(pix); per_class[cls][terrain] += 1; n_obj += 1
            if len(tiles[cls]) < 40:
                tiles[cls].append((crop, terrain))
        for _ in range(8):      # the prior: random 120 px windows
            x, y = rng.randint(0, 3840-120), rng.randint(0, 2160-120)
            terrain, _ = classify(img[y:y+120, x:x+120].reshape(-1, 1, 3)); prior[terrain] += 1
    total_prior = sum(prior.values())
    print(f"{'class':<16}{'n':>5}" + ''.join(f'{t:>8}' for t in TERRAIN))
    print(f"{'RANDOM PRIOR':<16}{total_prior:>5}" + ''.join(f'{prior[t]/total_prior:>8.2f}' for t in TERRAIN))
    rows = {}
    for cls in sorted(per_class):
        n = sum(per_class[cls].values()); rows[cls] = {t: per_class[cls][t]/n for t in TERRAIN}
        print(f"{cls:<16}{n:>5}" + ''.join(f'{per_class[cls][t]/n:>8.2f}' for t in TERRAIN))
    allc = Counter(); [allc.update(c) for c in per_class.values()]
    print(f"{'ALL OBJECTS':<16}{n_obj:>5}" + ''.join(f'{allc[t]/n_obj:>8.2f}' for t in TERRAIN))
    (HERE/'out'/'backdrop_study.json').write_text(json.dumps({'prior': {t: prior[t]/total_prior for t in TERRAIN}, 'per_class': rows, 'all': {t: allc[t]/n_obj for t in TERRAIN}}, indent=1))
    sheet = []
    for cls in sorted(tiles):
        row = []
        for crop, terrain in rng.sample(tiles[cls], min(6, len(tiles[cls]))):
            t = cv2.resize(crop, (110, 110), interpolation=cv2.INTER_AREA)
            cv2.putText(t, terrain, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1, cv2.LINE_AA); row.append(t)
        row += [np.zeros((110, 110, 3), np.uint8)]*(6-len(row))
        lab = np.zeros((110, 130, 3), np.uint8); cv2.putText(lab, cls[:14], (2, 60), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        sheet.append(np.hstack([lab]+row))
    cv2.imwrite(str(HERE/'out'/'backdrop_sheet.jpg'), np.vstack(sheet))
    print('elias/out/backdrop_sheet.jpg')


if __name__ == '__main__':
    main()
