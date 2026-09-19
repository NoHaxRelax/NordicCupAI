"""View-level evaluation and hard-negative mining for elias/detector.py.

A window classifier is judged by accuracy, a detector by what it does to a whole 960x540 view, where
about two thousand windows are background. This module renders views exactly as the evaluator does and

    python elias/views.py eval --weights W.pt --scene validation          # AP50, recall, false positives per view
    python elias/views.py mine --weights W.pt --scene helsinki --out elias/out/neg_helsinki_r1.npz

`eval` must be run on the scene the weights never saw. `mine` must be run on the TRAINING scene only:
it collects the windows behind confident detections that touch no labelled object (SPEC container,
label 16) so the next training round can learn that they are terrain.

On `validation` the labels cover 11 of the 13 classes that are present, so detections of the five
unlabelled classes cannot be judged: eval reports them separately and mine skips them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from dtos import OBJECT_CLASSES  # noqa: E402
from detector import FACTOR, WIN, WindowDetector  # noqa: E402

BOUNDS = {0: (1920, 1920, 1080, 1080), 1: (960, 2880, 540, 1620), 2: (480, 3360, 270, 1890)}
UNLABELLED = {'validation': {'condor', 'jammer', 'small_plane', 'spacecraft', 'ta-ta'}, 'helsinki': set()}


class YoloViews:
    """An Ultralytics checkpoint behind the same detect(view, level) interface, optionally re-scored by the
    window classifier: confidence <- confidence * p_classifier(label) ** power (a verifier re-ranks, it never
    deletes: low-confidence boxes are free under AP, confident false positives are not)."""
    def __init__(self, weights, verifier=None, imgsz=960, conf=0.03, power=1.0, relabel=False):
        self.relabel = relabel
        from ultralytics import YOLO
        self.model = YOLO(weights); self.imgsz, self.conf, self.power = imgsz, conf, power
        self.ver = WindowDetector(verifier) if verifier else None

    def detect(self, view, level):
        import torch
        res = self.model.predict(view, imgsz=self.imgsz, conf=self.conf, verbose=False, device=0)[0]
        rows = [{'label': self.model.names[int(c)], 'box': [x1, y1, x2, y2], 'confidence': float(sc)}
                for x1, y1, x2, y2, sc, c in res.boxes.data.cpu().tolist()]
        if not self.ver or not rows:
            return rows
        from detector import spec_scale, SCALES
        f = FACTOR[level]; wins, sidx = [], []
        for r in rows:
            x1, y1, x2, y2 = r['box']; s = spec_scale(max(x2-x1, y2-y1)*f, f); red = s//f
            small = view if red == 1 else cv2.resize(view, (960//red, 540//red), interpolation=cv2.INTER_AREA)
            pad = np.zeros((small.shape[0]+WIN, small.shape[1]+WIN, 3), np.uint8); pad[WIN//2:WIN//2+small.shape[0], WIN//2:WIN//2+small.shape[1]] = small
            cx, cy = int(round((x1+x2)/2/red)), int(round((y1+y2)/2/red))
            wins.append(pad[cy:cy+WIN, cx:cx+WIN]); sidx.append(SCALES.index(min(s, 8)))
        x = torch.from_numpy(np.stack(wins)).to(self.ver.device).permute(0, 3, 1, 2).float().div_(255.).sub_(0.45).div_(0.25)
        prob = self.ver._logits(x.contiguous(memory_format=torch.channels_last), torch.tensor(sidx, device=self.ver.device)).softmax(1).cpu().numpy()
        extra = []
        for r, p in zip(rows, prob):
            k = int(np.argmax(p[:len(OBJECT_CLASSES)])); own = list(OBJECT_CLASSES).index(r['label'])
            if self.relabel and k != own and p[k] > 2*p[own]:
                # a second opinion that disagrees strongly: report BOTH labels (a wrong extra costs little,
                # the right one gains a true positive), the classifier's first
                extra.append({'label': OBJECT_CLASSES[k], 'box': r['box'], 'confidence': float(r['confidence']*p[k]**self.power)})
            r['confidence'] = float(r['confidence']*p[own]**self.power)
        return rows+extra


def frames_of(scene, step):
    out = []
    for f in sorted((ROOT/'src'/scene/'annotations').glob('*.json')):
        img = ROOT/'src'/scene/'images'/(f.stem+'.png')
        if img.exists() and img.stat().st_size > 1_000_000:
            out.append((img, json.loads(f.read_text())['annotations']))
    return out[::step]


def render(img, level, cx, cy):
    f = FACTOR[level]; x1, y1 = cx-480*f, cy-270*f
    crop = img[y1:y1+540*f, x1:x1+960*f]
    return (crop if f == 1 else cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)), (x1, y1, f)


def views_for(ann, rng, n_l1, n_l2):
    """The deployed top-band L1 centres plus random ones, and L2 views on objects plus random ones."""
    v = [(0, 1920, 1080)]+[(1, x, 540) for x in (960, 1920, 2880)]
    for _ in range(n_l1):
        v.append((1, int(rng.integers(960, 2881)), int(rng.integers(540, 1621))))
    boxes = [a['bbox'] for a in ann]
    for k in range(n_l2):
        if boxes and k % 2 == 0:
            b = boxes[int(rng.integers(len(boxes)))]
            cx = (b[0]+b[2])/2+rng.integers(-300, 301); cy = (b[1]+b[3])/2+rng.integers(-180, 181)
        else:
            cx, cy = rng.integers(480, 3361), rng.integers(270, 1891)
        v.append((2, int(np.clip(cx, 480, 3360)), int(np.clip(cy, 270, 1890))))
    return v


def gt_in_view(ann, x1, y1, f):
    """[(class, box in delivered px, visible fraction)] for objects touching the view."""
    out = []
    for a in ann:
        b = np.array(a['bbox'], float); d = np.array([(b[0]-x1)/f, (b[1]-y1)/f, (b[2]-x1)/f, (b[3]-y1)/f])
        ix = max(0., min(d[2], 960)-max(d[0], 0)); iy = max(0., min(d[3], 540)-max(d[1], 0))
        vis = ix*iy/max(1e-6, (d[2]-d[0])*(d[3]-d[1]))
        if vis > 0:
            out.append((a['object_id'], d, vis))
    return out


def iou(a, b):
    ix = max(0., min(a[2], b[2])-max(a[0], b[0])); iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def ap101(scores, tp, n_gt):
    if n_gt == 0:
        return None
    order = np.argsort(-np.array(scores)); tp = np.array(tp, float)[order]
    ctp = np.cumsum(tp); prec = ctp/(np.arange(len(tp))+1); rec = ctp/n_gt
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0. for r in np.linspace(0, 1, 101)]))


def cmd_eval(a):
    det = (YoloViews(a.weights, a.verifier or None, a.imgsz, power=a.power, relabel=a.relabel) if a.yolo else WindowDetector(a.weights, propose=a.propose, accept=a.accept))
    rng = np.random.default_rng(a.seed); skip = UNLABELLED[a.scene]
    trained = set(a.classes.split(',')) if a.classes else set(OBJECT_CLASSES)
    per = {c: {'s': [], 'tp': [], 'n': 0} for c in OBJECT_CLASSES}
    stats = {l: {'views': 0, 'fp50': 0, 'unver50': 0, 'gt': 0, 'hit50': 0, 'loc': 0, 'ms': []} for l in (0, 1, 2)}
    for img_path, ann in frames_of(a.scene, a.step):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        for level, cx, cy in views_for(ann, rng, a.n_l1, a.n_l2):
            view, (x1, y1, f) = render(img, level, cx, cy)
            t0 = time.perf_counter(); rows = det.detect(view, level); st = stats[level]; st['ms'].append((time.perf_counter()-t0)*1000)
            gts = gt_in_view(ann, x1, y1, f); used = set(); st['views'] += 1
            full = [i for i, g in enumerate(gts) if g[2] >= 0.6 and g[0] in trained]
            for i in full:
                per[gts[i][0]]['n'] += 1; st['gt'] += 1
            for r in sorted(rows, key=lambda r: -r['confidence']):
                if r['label'] in skip:
                    st['unver50'] += r['confidence'] >= 0.5; continue
                best, bi = 0., -1
                for i, g in enumerate(gts):
                    o = iou(r['box'], g[1])
                    if o > best:
                        best, bi = o, i
                if bi >= 0 and best >= 0.5 and gts[bi][0] == r['label'] and bi not in used and bi in full:
                    used.add(bi); per[r['label']]['s'].append(r['confidence']); per[r['label']]['tp'].append(1)
                    st['hit50'] += r['confidence'] >= 0.5
                elif bi >= 0 and best >= 0.3 and (bi not in full):
                    continue  # touches a partly visible or untrained object: neither right nor wrong
                else:
                    per[r['label']]['s'].append(r['confidence']); per[r['label']]['tp'].append(0)
                    st['fp50'] += r['confidence'] >= 0.5
            st['loc'] += sum(any(iou(r['box'], gts[i][1]) >= 0.3 for r in rows) for i in full)
    aps = {c: ap101(v['s'], v['tp'], v['n']) for c, v in per.items() if v['n'] and c not in skip and c in trained}
    print(f"weights {a.weights}\nscene {a.scene}  mAP50 {np.mean(list(aps.values())):.3f} over {len(aps)} classes")
    print('  ' + '  '.join(f'{c}:{v:.2f}' for c, v in sorted(aps.items(), key=lambda kv: -kv[1])))
    for l, st in stats.items():
        if st['views']:
            print(f"  L{l}: views {st['views']:4d}  GT {st['gt']:4d}  recall@.5conf {st['hit50']/max(1, st['gt']):.2f}  located {st['loc']/max(1, st['gt']):.2f}  "
                  f"FP/view {st['fp50']/st['views']:.1f}  unverifiable/view {st['unver50']/st['views']:.1f}  {np.median(st['ms']):.0f} ms")
    if a.json:
        Path(a.json).write_text(json.dumps({'mAP50': float(np.mean(list(aps.values()))), 'ap': aps,
                                            'levels': {str(l): {k: (v if k != 'ms' else float(np.median(v or [0]))) for k, v in st.items()} for l, st in stats.items()}}, indent=1))


def cmd_mine(a):
    det = WindowDetector(a.weights, propose=a.propose, accept=a.accept)
    rng = np.random.default_rng(a.seed); skip = UNLABELLED[a.scene]
    X, meta = [], []
    for img_path, ann in frames_of(a.scene, a.step):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR); frame = int(img_path.stem.split('_')[1])
        for level, cx, cy in views_for(ann, rng, a.n_l1, a.n_l2):
            view, (x1, y1, f) = render(img, level, cx, cy); gts = gt_in_view(ann, x1, y1, f)
            cache = {}
            for r in det.detect(view, level):
                if r['label'] in skip or r['confidence'] < a.min_conf:
                    continue
                # keep well clear of every labelled object: a near miss is a localisation matter, not terrain
                ctr = np.array(r['centre'], float)
                if any(iou(r['box'], g[1]) > 0.02 or np.hypot(*(ctr-(g[1][:2]+g[1][2:])/2)) < 0.75*max(g[1][2]-g[1][0], g[1][3]-g[1][1])+8 for g in gts):
                    continue
                red = r['r']
                if red not in cache:
                    small = view if red == 1 else cv2.resize(view, (960//red, 540//red), interpolation=cv2.INTER_AREA)
                    pad = np.zeros((small.shape[0]+WIN, small.shape[1]+WIN, 3), np.uint8)
                    pad[WIN//2:WIN//2+small.shape[0], WIN//2:WIN//2+small.shape[1]] = small; cache[red] = pad
                px, py = int(round(ctr[0]/red)), int(round(ctr[1]/red))
                X.append(cache[red][py:py+WIN, px:px+WIN].copy()); meta.append((level, r['scale'], frame, r['label'], r['confidence']))
    if not X:
        print('nothing mined'); return
    n = len(X); lab = np.full(n, len(OBJECT_CLASSES), np.int16)
    np.savez_compressed(a.out, x=np.stack(X), label=lab, level=np.array([m[0] for m in meta], np.int8),
                        scale=np.array([m[1] for m in meta], np.int8), size_px=np.zeros(n, np.float32),
                        box_w=np.zeros(n, np.float32), box_h=np.zeros(n, np.float32), off=np.zeros((n, 2), np.float32),
                        scene=np.array([a.scene]*n), frame=np.array([m[2] for m in meta], np.int32),
                        track=np.array([f'mined:{m[3]}' for m in meta]), source=np.array(['mined']*n))
    by = {}
    for m in meta:
        by[m[3]] = by.get(m[3], 0)+1
    print(f'mined {n} negative windows from {a.scene} -> {a.out}\n  mistaken for: ' + ', '.join(f'{k} {v}' for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
    sheet = np.vstack([np.hstack([cv2.resize(X[i], (128, 128), interpolation=cv2.INTER_NEAREST) for i in rng.choice(n, 12)]) for _ in range(6)])
    cv2.imwrite(str(Path(a.out).with_suffix('.jpg')), sheet)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['eval', 'mine']); ap.add_argument('--weights', required=True)
    ap.add_argument('--scene', required=True, choices=['helsinki', 'validation'])
    ap.add_argument('--step', type=int, default=1, help='use every k-th frame'); ap.add_argument('--n-l1', type=int, default=1)
    ap.add_argument('--n-l2', type=int, default=3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--propose', type=float, default=0.25); ap.add_argument('--accept', type=float, default=0.35)
    ap.add_argument('--min-conf', type=float, default=0.35); ap.add_argument('--classes', default='', help='comma list the weights were trained on')
    ap.add_argument('--yolo', action='store_true', help='--weights is an Ultralytics checkpoint'); ap.add_argument('--verifier', default='')
    ap.add_argument('--relabel', action='store_true'); ap.add_argument('--imgsz', type=int, default=960); ap.add_argument('--power', type=float, default=1.0)
    ap.add_argument('--out', default=str(HERE/'out'/'neg_mined.npz')); ap.add_argument('--json', default='')
    a = ap.parse_args()
    {'eval': cmd_eval, 'mine': cmd_mine}[a.cmd](a)


if __name__ == '__main__':
    main()
