"""Train and evaluate the scale-preserving window classifier (SPEC.md) and its ablations.

    python elias/train.py --train elias/out/synth_A.npz --test elias/out/real_validation.npz \
        --scale-mode input --heads single --tag A_to_val

Arms of the experiment matrix:
  --scale-mode input      one model, the window scale s (1,2,4,8) given as constant input channels
               none       one model, blind to scale ("resolution indifferent")
               per-scale  one model per scale value, each trained and tested on its own scale only
  --heads      single     one 17-way softmax (16 classes + background)
               experts    16 separate one-against-rest networks at equal TOTAL width, combined by argmax
                          of their probabilities with background when none exceeds 0.5

The report is accuracy per class versus delivered pixel size on the TEST file, which must come from a
scene whose backgrounds and sprites the training file never saw (SPEC.md hold-out rule).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dtos import OBJECT_CLASSES  # noqa: E402

CLASSES = list(OBJECT_CLASSES) + ['background']
BG = len(OBJECT_CLASSES)
SCALES = [1, 2, 4, 8]
SIZE_BINS = [0, 8, 12, 16, 20, 24, 32, 48, 1e9]


def load(paths):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ('x', 'label', 'scale', 'size_px', 'level')}
    return out


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y+(x if self.skip is None else self.skip(x)), inplace=True)


class WindowNet(nn.Module):
    """Fully convolutional: a 64x64 window gives a 4x4 map that is averaged. Slid over a whole view the
    same weights give a stride-16 heatmap, which is how this becomes a detector."""
    def __init__(self, n_out, width=32, scale_channels=True):
        super().__init__()
        self.scale_channels = scale_channels
        cin = 3+(len(SCALES) if scale_channels else 0)
        w = [width, width*2, width*4, width*6]
        self.stem = nn.Sequential(nn.Conv2d(cin, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        layers, c = [], w[0]
        for wi in w:
            layers += [Block(c, wi, 2), Block(wi, wi, 1)]; c = wi
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, n_out, 1)

    def forward(self, x, scale_idx):
        if self.scale_channels:
            onehot = F.one_hot(scale_idx, len(SCALES)).float()[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, onehot], 1)
        return self.head(self.body(self.stem(x))).mean((2, 3))


class Experts(nn.Module):
    def __init__(self, width, scale_channels):
        super().__init__()
        self.nets = nn.ModuleList([WindowNet(1, max(8, width//4), scale_channels) for _ in range(BG)])

    def forward(self, x, scale_idx):
        return torch.cat([n(x, scale_idx) for n in self.nets], 1)  # 16 logits, one against rest


def to_tensors(d, device):
    x = torch.from_numpy(d['x']).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 BGR, NCHW
    y = torch.from_numpy(d['label'].astype(np.int64)).to(device)
    s = torch.from_numpy(np.searchsorted(SCALES, d['scale']).astype(np.int64)).to(device)
    return x, y, s


def predict(model, heads, x, s, bs=1024, tta=False):
    """tta: average class probabilities over a horizontal flip and three exposures."""
    model.eval(); out = []
    variants = [(g, f) for g in ((0.8, 1.0, 1.25) if tta else (1.0,)) for f in ((False, True) if tta else (False,))]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(x), bs):
            x0 = x[i:i+bs].float().div_(255.)
            if heads == 'single':
                prob = 0
                for g, f in variants:
                    xv = (x0*g).clamp_(0, 1).sub_(0.45).div_(0.25)
                    prob = prob+model(xv.flip(3) if f else xv, s[i:i+bs]).float().softmax(1)
                out.append(prob.argmax(1)); continue
            xb = x0.sub_(0.45).div_(0.25)
            lg = model(xb, s[i:i+bs]).float()
            if heads == 'single':
                out.append(lg.argmax(1))
            else:
                p = lg.sigmoid(); best = p.max(1)
                out.append(torch.where(best.values > 0.5, best.indices, torch.full_like(best.indices, BG)))
    return torch.cat(out)


def train_one(d_train, a, device, scale_channels, only_scale=None):
    x, y, s = to_tensors(d_train, device)
    if only_scale is not None:
        m = s == only_scale; x, y, s = x[m], y[m], s[m]
    model = (WindowNet(len(CLASSES), a.width, scale_channels) if a.heads == 'single' else Experts(a.width, scale_channels)).to(device)
    model = model.to(memory_format=torch.channels_last)
    steps = a.epochs*(len(x)//a.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(1, steps), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    step = 0
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x)-a.batch+1, a.batch):
            idx = perm[i:i+a.batch]
            xb = x[idx].float().div_(255.)
            xb = xb*(1+0.06*(torch.rand(len(idx), 1, 1, 1, device=device)-.5))
            if a.aug:  # colour balance, gamma, saturation and a horizontal flip (never vertical: lean rule)
                n = len(idx); r = lambda *sh: torch.rand(*sh, device=device)
                xb = xb*torch.exp((r(n, 3, 1, 1)-.5)*2*math.log(1.2))
                xb = xb.clamp(1e-4, 1)**torch.exp((r(n, 1, 1, 1)-.5)*2*math.log(1.25))
                gray = xb.mean(1, keepdim=True); xb = (gray+(xb-gray)*(0.5+r(n, 1, 1, 1))).clamp(0, 1)
                flip = r(n) < .5; xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            xb = xb.sub_(0.45).div_(0.25).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                lg = model(xb, s[idx])
                if a.heads == 'single':
                    loss = F.cross_entropy(lg.float(), y[idx], label_smoothing=0.05)
                else:
                    tgt = F.one_hot(y[idx], len(CLASSES))[:, :BG].float()
                    loss = F.binary_cross_entropy_with_logits(lg.float(), tgt, pos_weight=torch.full((BG,), 4.0, device=device))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); step += 1
    return model


def report(pred, d, tag, trained=None):
    y, size = d['label'], d['size_px']; p = pred.cpu().numpy()
    pos = y != BG
    shared = pos & np.isin(y, sorted(trained)) if trained is not None else pos  # classes the training set contains
    rows = {'tag': tag, 'n_test': int(len(y)), 'acc_pos': float((p[pos] == y[pos]).mean()),
            'acc_shared': float((p[shared] == y[shared]).mean()), 'n_shared': int(shared.sum()),
            'macro_acc_shared': float(np.mean([(p[(y == c)] == c).mean() for c in np.unique(y[shared])])),
            'shared_classes': [CLASSES[c] for c in np.unique(y[shared])],
            'bg_false_alarm': float((p[~pos] != BG).mean()) if (~pos).any() else None,
            'objectness_recall': float((p[pos] != BG).mean()), 'by_size': [], 'by_class': {}, 'by_class_size': {}}
    for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]):
        m = shared & (size >= lo) & (size < hi)
        if m.sum():
            rows['by_size'].append({'px': f'{lo:g}-{hi:g}' if hi < 1e8 else f'{lo:g}+', 'n': int(m.sum()),
                                    'acc': float((p[m] == y[m]).mean()), 'noticed': float((p[m] != BG).mean())})
    for c in range(BG):
        m = y == c
        if m.sum():
            rows['by_class'][CLASSES[c]] = {'n': int(m.sum()), 'acc': float((p[m] == c).mean())}
            rows['by_class_size'][CLASSES[c]] = [
                {'px': f'{lo:g}-{hi:g}' if hi < 1e8 else f'{lo:g}+', 'n': int((m & (size >= lo) & (size < hi)).sum()),
                 'acc': float((p[m & (size >= lo) & (size < hi)] == c).mean())}
                for lo, hi in zip(SIZE_BINS[:-1], SIZE_BINS[1:]) if (m & (size >= lo) & (size < hi)).sum()]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', nargs='+', required=True); ap.add_argument('--test', nargs='+', required=True)
    ap.add_argument('--scale-mode', default='input', choices=['input', 'none', 'per-scale'])
    ap.add_argument('--heads', default='single', choices=['single', 'experts'])
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=512); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tag', default='run')
    ap.add_argument('--aug', type=int, default=0, help='1 = GPU colour balance, gamma, saturation, horizontal flip')
    ap.add_argument('--tta', type=int, default=0, help='1 = test-time averaging over flip and exposure')
    ap.add_argument('--out', default=str(HERE/'out'/'runs'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = 'cuda'
    d_train, d_test = load(a.train), load(a.test)
    t0 = time.time()
    xt, yt, st = to_tensors(d_test, device)
    if a.scale_mode == 'per-scale':
        pred = torch.full_like(yt, BG); models = {}
        for k, sv in enumerate(SCALES):
            if (st == k).any() and (torch.from_numpy(d_train['scale']) == sv).any():
                models[sv] = train_one(d_train, a, device, False, only_scale=k)
                pred[st == k] = predict(models[sv], a.heads, xt[st == k], st[st == k], tta=bool(a.tta))
        state = {str(k): m.state_dict() for k, m in models.items()}
    else:
        model = train_one(d_train, a, device, a.scale_mode == 'input')
        pred = predict(model, a.heads, xt, st, tta=bool(a.tta)); state = model.state_dict()
    rows = report(pred, d_test, a.tag, trained=set(np.unique(d_train['label']).tolist())-{BG})
    rows.update({'train': a.train, 'test': a.test, 'scale_mode': a.scale_mode, 'heads': a.heads, 'width': a.width,
                 'epochs': a.epochs, 'aug': a.aug, 'tta': a.tta, 'n_train': int(len(d_train['label'])), 'seconds': round(time.time()-t0, 1)})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f'{a.tag}.json').write_text(json.dumps(rows, indent=1)); torch.save(state, out/f'{a.tag}.pt')
    print(f"{a.tag}: acc on shared classes {rows['acc_shared']:.3f} (macro {rows['macro_acc_shared']:.3f}, {len(rows['shared_classes'])} classes)  background false alarm {rows['bg_false_alarm']}  "
          f"noticed {rows['objectness_recall']:.3f}  ({rows['seconds']} s, train n={rows['n_train']})")
    print('  by delivered size: ' + '  '.join(f"{r['px']}:{r['acc']:.2f}(n{r['n']})" for r in rows['by_size']))


if __name__ == '__main__':
    main()
