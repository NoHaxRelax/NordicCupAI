"""Per class, view and inference size: does the detector find the object, and what does it call it?

Helsinki has organiser truth for all 16 classes (one physical instance each; both night models trained on sprites
cut from it, so Helsinki numbers are a SEEN-instance upper bound). Validation has the team's pseudo-labels for 11
classes, with boxes off the organisers' convention, so matching is by centre as well as by IoU. For every labelled
object the script renders the L0 overview, the deployed top-band L1 view when the object lies inside that band,
a best-case L1 view centred on it, and an L2 view centred on it with jitter, then runs the detector at each input
size. "located" = any box of any class on the object; "right" = a box with the right class on it; "iou50" = right
class and IoU >= 0.5 with the label.

    python elias/class_probe.py --weights elias/release/both_m1280.pt --scene helsinki --imgsz 1280 1536 1920
    python elias/class_probe.py --weights elias/release/both_m1280.pt --scene validation --step 2 \
        --classes small_launcher medium_launcher medium_plane
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FACTOR = {0: 4, 1: 2, 2: 1}
BOUNDS = {1: (960, 2880, 540, 1620), 2: (480, 3360, 270, 1890)}
KINDS = ['L0', 'L1band', 'L1best', 'L2']


def iou(a, b):
    ix = max(0., min(a[2], b[2])-max(a[0], b[0])); iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def render(img, level, cx, cy):
    f = FACTOR[level]; x1, y1 = cx-480*f, cy-270*f
    crop = img[y1:y1+540*f, x1:x1+960*f]
    return (crop if f == 1 else cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)), (x1, y1, f)


def views_for_object(b, rng):
    ox, oy = (b[0]+b[2])/2, (b[1]+b[3])/2
    out = [('L0', 0, 1920, 1080)]
    x0, x1, y0, y1 = BOUNDS[1]
    out.append(('L1best', 1, int(np.clip(ox+rng.integers(-200, 201), x0, x1)), int(np.clip(oy+rng.integers(-100, 101), y0, y1))))
    if b[3] < 1080:
        out.append(('L1band', 1, min((960, 1920, 2880), key=lambda c: abs(c-ox)), 540))
    x0, x1, y0, y1 = BOUNDS[2]
    out.append(('L2', 2, int(np.clip(ox+rng.integers(-150, 151), x0, x1)), int(np.clip(oy+rng.integers(-80, 81), y0, y1))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True); ap.add_argument('--scene', required=True, choices=['helsinki', 'validation'])
    ap.add_argument('--imgsz', type=int, nargs='+', default=[1280]); ap.add_argument('--classes', nargs='+', default=None)
    ap.add_argument('--step', type=int, default=1); ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--json', default='')
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights); names = model.names; rng = np.random.default_rng(a.seed)
    excl = json.loads((HERE/'data'/'validation_exclude.json').read_text()).get('exclude', []) if a.scene == 'validation' else []

    def excluded(o, fr):
        t = o.get('track_id') or o.get('track') or ''
        return any(e.get('track') == t and e.get('first', -1) <= fr <= e.get('last', 10**9) for e in excl)

    frames = []
    for p in sorted((ROOT/'src'/a.scene/'images').glob('frame_*.png')):
        if p.stat().st_size > 1_000_000:
            frames.append((p, json.loads((ROOT/'src'/a.scene/'annotations'/f'{p.stem}.json').read_text())['annotations']))
    frames = frames[::a.step]
    agg = defaultdict(lambda: {'n': 0, 'loc': 0, 'loose': 0, 'c50': 0, 'conf': [], 'wrong': Counter()})
    for img_path, ann in frames:
        fr = int(img_path.stem.split('_')[1])
        objs = [o for o in ann if (not a.classes or o['object_id'] in a.classes) and not excluded(o, fr)]
        if not objs:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        keys, wants = [], []
        for i, o in enumerate(objs):
            for kind, lvl, cx, cy in views_for_object(np.array(o['bbox'], float), rng):
                key = (lvl, cx, cy)
                if key not in keys:
                    keys.append(key)
                wants.append((i, kind, key))
        imgs, offs = [], {}
        for key in keys:
            v, off = render(img, *key); imgs.append(v); offs[key] = off
        for sz in a.imgsz:
            res_by_key = {}
            for s in range(0, len(imgs), 8):
                for key, res in zip(keys[s:s+8], model.predict(imgs[s:s+8], imgsz=sz, conf=a.conf, verbose=False, device=0)):
                    res_by_key[key] = [(names[int(c)], np.array([x1, y1, x2, y2]), float(sc)) for x1, y1, x2, y2, sc, c in res.boxes.data.cpu().tolist()]
            for i, kind, key in wants:
                o = objs[i]; x1, y1, f = offs[key]; b = np.array(o['bbox'], float); d = (b-np.array([x1, y1, x1, y1]))/f
                ix = max(0., min(d[2], 960)-max(d[0], 0)); iy = max(0., min(d[3], 540)-max(d[1], 0))
                if ix*iy < 0.9*max(1e-6, (d[2]-d[0])*(d[3]-d[1])):
                    continue
                cx, cy = (d[0]+d[2])/2, (d[1]+d[3])/2; tol = 0.5*max(d[2]-d[0], d[3]-d[1])
                st = agg[(o['object_id'], kind, sz)]; st['n'] += 1
                located = [(lab, bx, sc) for lab, bx, sc in res_by_key[key]
                           if iou(bx, d) >= 0.3 or np.hypot((bx[0]+bx[2])/2-cx, (bx[1]+bx[3])/2-cy) <= tol]
                if not located:
                    continue
                st['loc'] += 1
                right = [r for r in located if r[0] == o['object_id']]
                if right:
                    st['loose'] += 1; st['conf'].append(max(r[2] for r in right))
                    if any(iou(r[1], d) >= 0.5 for r in right):
                        st['c50'] += 1
                else:
                    st['wrong'][max(located, key=lambda r: r[2])[0]] += 1
    print(f'weights {a.weights}  scene {a.scene}  frames {len(frames)}')
    print(f"{'class':<16}{'view':<8}{'size':>5}{'n':>5}{'located':>9}{'right':>7}{'iou50':>7}{'conf':>6}  called instead")
    rows = {}
    for (cls, kind, sz), st in sorted(agg.items(), key=lambda kv: (kv[0][0], KINDS.index(kv[0][1]), kv[0][2])):
        n = st['n']; conf = float(np.mean(st['conf'])) if st['conf'] else 0.
        print(f"{cls:<16}{kind:<8}{sz:>5}{n:>5}{st['loc']/n:>9.2f}{st['loose']/n:>7.2f}{st['c50']/n:>7.2f}{conf:>6.2f}  {dict(st['wrong'].most_common(3))}")
        rows[f'{cls}|{kind}|{sz}'] = {'n': n, 'located': st['loc']/n, 'right': st['loose']/n, 'iou50': st['c50']/n, 'conf': conf, 'wrong': dict(st['wrong'])}
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
