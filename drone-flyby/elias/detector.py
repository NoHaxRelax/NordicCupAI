"""Runtime detector built on the scale-preserving window classifier (elias/train.py, SPEC.md).

Plugs into Oscar's endpoint through its factory hook:

    DRONE_DETECTOR=elias.detector:build ELIAS_WEIGHTS=elias/out/runs/<tag>.pt python api.py

Per delivered view (960x540 at level L, factor f = 4, 2, 1):
  1. For each window scale s in {f, 2f, 4f} (s <= 8) the view is reduced by r = s / f with INTER_AREA,
     exactly as training windows were made, and zero-padded by half a window.
  2. Proposal pass: the network runs once over the whole reduced view (fully convolutional, stride 16).
     That pass is slightly off-distribution (a training window ends in zeros, a dense cell sees its
     neighbours), so it only nominates cells whose "not background" probability clears a low bar.
  3. Verification pass: around every nominated cell the exact 64x64 windows on a stride-8 grid are cut
     and classified in the mode the network was trained in. Training jitter was +-6 window px, so a
     stride of 8 always puts some window within reach of the object centre.
  4. Size consistency: altitude is fixed, so each class fits exactly one window scale per level
     (SPEC rule: smallest s with longer box side / s <= 48). A class predicted at another scale is a
     partial view of something bigger or a blob of something smaller, and is dropped.
  5. The box is the class's typical organiser box around the window centre (organiser boxes are loose
     footprints, 22 to 41 % object for some classes, so a tight box cannot reach IoU 0.5 anyway).
     Greedy class-agnostic suppression removes the same object found at neighbouring windows.

Rows: {'label', 'box' (delivered pixels xyxy), 'confidence'}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dtos import OBJECT_CLASSES  # noqa: E402
from train import BG, SCALES, WindowNet  # noqa: E402

FACTOR = {0: 4, 1: 2, 2: 1}
WIN = 64


def class_sizes():
    """Median organiser box (w, h) in source px per class, from both labelled scenes."""
    cache = HERE/'out'/'class_sizes.json'
    if cache.exists():
        return json.loads(cache.read_text())
    sizes = {c: [] for c in OBJECT_CLASSES}
    for scene in ('helsinki', 'validation'):
        for f in sorted((ROOT/'src'/scene/'annotations').glob('*.json')):
            for a in json.loads(f.read_text())['annotations']:
                x1, y1, x2, y2 = a['bbox']
                if x1 > 2 and y1 > 2 and x2 < 3837 and y2 < 2157:
                    sizes[a['object_id']].append((x2-x1, y2-y1))
    out = {c: [float(np.median([s[0] for s in v])), float(np.median([s[1] for s in v]))] for c, v in sizes.items() if v}
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(out, indent=1))
    return out


def spec_scale(side_src, f):
    for s in (f, 2*f, 4*f):
        if side_src/s <= 48:
            return s
    return 4*f


class WindowDetector:
    name = 'elias-windows'

    def __init__(self, weights, *, device=None, width=32, propose=0.25, accept=0.35, topk=160, slack=1.25):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = WindowNet(len(OBJECT_CLASSES)+1, width, True).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = self.model.to(memory_format=torch.channels_last)
        self.propose, self.accept, self.topk, self.slack = propose, accept, topk, slack
        self.sizes = class_sizes()
        self.half = self.device != 'cpu'
        self(np.zeros((540, 960, 3), np.uint8), {'view': {'resolution_level': 1}})  # warm up

    def valid_scales(self, label, f):
        """Scales at which this class may legitimately be recognised at this level: the SPEC scale of
        its typical box, plus the neighbour when the box sits within `slack` of the 48 px boundary."""
        w, h = self.sizes.get(label, (60, 60)); side = max(w, h)
        ok = {spec_scale(side, f), spec_scale(side*self.slack, f), spec_scale(side/self.slack, f)}
        return ok

    @torch.no_grad()
    def _logits(self, x, s_idx):
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
            return self.model(x, s_idx).float()

    def __call__(self, image, request):
        rows = self.detect(image, int(request['view']['resolution_level']))
        return [{'label': r['label'], 'box': r['box'], 'confidence': r['confidence']} for r in rows]

    @torch.no_grad()
    def detect(self, image, level, check_scale=True):
        """Rows with the extra keys 'scale' (window scale s), 'r' (reduction of the view) and 'centre'
        (window centre in delivered pixels), which negative mining needs to re-cut the window."""
        f = FACTOR[level]
        rows = []
        for r in (1, 2, 4):
            s = f*r
            if s not in SCALES:
                continue
            img = image if r == 1 else cv2.resize(image, (image.shape[1]//r, image.shape[0]//r), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            pad = np.zeros((h+WIN, w+WIN, 3), np.uint8); pad[WIN//2:WIN//2+h, WIN//2:WIN//2+w] = img
            t = torch.from_numpy(pad).to(self.device).permute(2, 0, 1)[None].float().div_(255.).sub_(0.45).div_(0.25)
            t = t.contiguous(memory_format=torch.channels_last)
            s_idx = torch.tensor([SCALES.index(s)], device=self.device)
            # proposal pass: dense logits, stride 16 over the padded image
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
                m = self.model
                onehot = F.one_hot(s_idx, len(SCALES)).float()[:, :, None, None].expand(-1, -1, t.shape[2], t.shape[3])
                dense = m.head(m.body(m.stem(torch.cat([t, onehot], 1)))).float()[0]      # [17, H/16, W/16]
            obj = 1-dense.softmax(0)[BG]
            ys, xs = torch.nonzero(obj > self.propose, as_tuple=True)
            if len(ys) == 0:
                continue
            order = obj[ys, xs].argsort(descending=True)[:self.topk]
            ys, xs = ys[order], xs[order]
            # cell (y, x) covers padded pixels [16y, 16y+16); candidate window centres on a stride-8 grid around it
            cy = (ys*16+8)[:, None]+torch.tensor([-8, 0, 8], device=self.device)[None]
            cx = (xs*16+8)[:, None]+torch.tensor([-8, 0, 8], device=self.device)[None]
            cyx = torch.stack([cy[:, :, None].expand(-1, 3, 3), cx[:, None, :].expand(-1, 3, 3)], -1).reshape(-1, 2)
            cyx = torch.unique(cyx, dim=0)
            cyx = cyx[(cyx[:, 0] >= WIN//2) & (cyx[:, 0] <= h+WIN//2) & (cyx[:, 1] >= WIN//2) & (cyx[:, 1] <= w+WIN//2)]
            if len(cyx) == 0:
                continue
            top, left = cyx[:, 0]-WIN//2, cyx[:, 1]-WIN//2
            ar = torch.arange(WIN, device=self.device)
            yy = (top[:, None]+ar[None])[:, :, None].expand(-1, WIN, WIN); xx = (left[:, None]+ar[None])[:, None, :].expand(-1, WIN, WIN)
            wins = t[0][:, yy, xx].permute(1, 0, 2, 3).contiguous(memory_format=torch.channels_last)
            prob = torch.cat([self._logits(wins[i:i+512], s_idx.expand(min(512, len(wins)-i))).softmax(1)
                              for i in range(0, len(wins), 512)])
            conf, cls = prob[:, :BG].max(1)
            keep = conf > self.accept
            for c, k, (py, px) in zip(conf[keep].tolist(), cls[keep].tolist(), cyx[keep].tolist()):
                label = OBJECT_CLASSES[k]
                if check_scale and s not in self.valid_scales(label, f):
                    continue
                bw, bh = self.sizes.get(label, (60, 60))
                ctr_x, ctr_y = (px-WIN//2)*r, (py-WIN//2)*r                      # delivered pixels
                rows.append({'label': label, 'confidence': float(c), 'scale': s, 'r': r, 'centre': (ctr_x, ctr_y),
                             'box': [ctr_x-bw/f/2, ctr_y-bh/f/2, ctr_x+bw/f/2, ctr_y+bh/f/2]})
        return suppress(rows, image.shape[1], image.shape[0])


def suppress(rows, width, height, iou_thr=0.3):
    rows = sorted(rows, key=lambda r: -r['confidence']); kept = []
    for r in rows:
        b = r['box']
        if any(_iou(b, k['box']) > iou_thr for k in kept):
            continue
        kept.append(r)
    for r in kept:  # keep boxes inside the delivered image; the tracker treats edge-touching boxes as partial
        x1, y1, x2, y2 = r['box']; r['box'] = [max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)]
    return [r for r in kept if r['box'][2]-r['box'][0] > 2 and r['box'][3]-r['box'][1] > 2]


def _iou(a, b):
    ix = max(0., min(a[2], b[2])-max(a[0], b[0])); iy = max(0., min(a[3], b[3])-max(a[1], b[1]))
    inter = ix*iy; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def build():
    weights = os.environ.get('ELIAS_WEIGHTS')
    if not weights:
        raise ValueError('elias.detector:build needs ELIAS_WEIGHTS (a .pt from elias/train.py, --scale-mode input --heads single)')
    return WindowDetector(weights, device=os.environ.get('DRONE_DEVICE') or None,
                          width=int(os.environ.get('ELIAS_WIDTH', '32')),
                          propose=float(os.environ.get('ELIAS_PROPOSE', '0.25')),
                          accept=float(os.environ.get('ELIAS_ACCEPT', '0.35')))
