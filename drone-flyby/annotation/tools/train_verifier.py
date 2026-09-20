#!/usr/bin/env python3
"""Train one member of the small-class verifier ensemble on 64 px L1 crops.

Sources are crop indices (index.jsonl from make_verifier_synth.py or crop_picker/cut_crops.py), each given as
NAME=ROOT. Rows are mapped to the verifier's classes by their `kind`:
  positive / real -> the crop's class          negative -> background
  other           -> other (any other labelled object)
  test            -> held out entirely; scored at the end (needs --marks to join truth and detector confidence)
Only level-1 crops (64 px) train; L2 crops are reduced to 64 px with INTER_AREA and L0 crops enlarged, so every
input covers 128 source px at the L1 scale (scale-preserving: the object's physical size stays a cue).

Splits come from the rows (train / dev / test); --scene-test makes every row of a scene a test row (the honest
cross-scene number).

    python3 train_verifier.py --source synth=/workspace/verifier/synth-v6 --source picked=/workspace/verifier/picked-v1 \
        --source real=/workspace/verifier/real-v1 --marks /workspace/verifier/marks/test-marks.json \
        --model convnext_tiny --input 128 --epochs 12 --out /workspace/verifier/runs/m1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ['small_launcher', 'medium_launcher', 'ta-ta', 'jammer', 'other', 'background']
CIDX = {c: i for i, c in enumerate(CLASSES)}
SMALL = CLASSES[:4]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to64(img, level):
    """Crop of a 128-source-px window at any level -> 64x64 (the L1 look)."""
    if img.shape[0] != 64:
        interp = cv2.INTER_AREA if img.shape[0] > 64 else cv2.INTER_CUBIC
        img = cv2.resize(img, (64, 64), interpolation=interp)
    return img


def label_of(row):
    kind, cls = row.get('kind'), row.get('class')
    if kind in ('positive', 'real'):
        return CIDX[cls] if cls in CIDX and cls in SMALL else CIDX['other']
    if kind == 'negative':
        return CIDX['background']
    if kind == 'other':
        return CIDX['other']
    return None


def load_sources(specs, marks_path=None, scene_test=None, max_per=None):
    marks = {}
    if marks_path:
        m = json.loads(Path(marks_path).read_text())
        for mk in (m['marks'] if isinstance(m, dict) else m):
            marks[mk['id']] = mk
    rows = []
    for spec in specs:
        name, root = spec.split('=', 1)
        root = Path(root)
        per = Counter()
        for line in open(root / 'index.jsonl'):
            r = json.loads(line)
            if r.get('kind') == 'placement':
                continue
            lvl = int(r.get('level', 1))
            r['_path'] = str(root / r['file']); r['_source'] = name; r['_level'] = lvl
            if r.get('kind') == 'test':
                r['_label'] = None; r['split'] = 'test'
            else:
                r['_label'] = label_of(r)
                if r['_label'] is None:
                    continue
            if scene_test and r.get('scene') == scene_test and r.get('kind') != 'test':
                # a held-out scene: its crops become test rows with truth from their kind
                r['split'] = 'test'
                r['_truth'] = {'real': 'object', 'positive': 'object', 'negative': 'terrain', 'other': 'other_object'}.get(r.get('kind'), 'unknown')
                r['_det_label'] = r.get('class') if r.get('kind') in ('real', 'positive') else None
                r['_det_conf'] = 1.0; r['_split_frame'] = 'scene-heldout'
            mk = marks.get(r.get('mark_id'))
            if mk:
                for k in ('det_conf', 'truth', 'det_label', 'det_level', 'split_frame'):
                    if k in mk:
                        r['_' + k] = mk[k]
            key = (r.get('kind'), r.get('class'), r.get('split'), lvl)
            if max_per and per[key] >= max_per:
                continue
            per[key] += 1
            rows.append(r)
    return rows


def load_images(rows, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    def read(r):
        im = cv2.imread(r['_path'], cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit(f'cannot read {r["_path"]}')
        return to64(im, r['_level'])[:, :, ::-1]  # RGB
    with ThreadPoolExecutor(workers) as ex:
        ims = list(ex.map(read, rows))
    return np.stack(ims).astype(np.uint8)


class SmallCNN(nn.Module):
    """From-scratch member at native 64 px (Elias's window-net idea): no pretrained bias."""
    def __init__(self, n, width=48):
        super().__init__()
        def block(a, b, stride):
            return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1, bias=False), nn.BatchNorm2d(b), nn.GELU(),
                                 nn.Conv2d(b, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.GELU())
        self.body = nn.Sequential(block(3, width, 1), block(width, width * 2, 2), block(width * 2, width * 4, 2),
                                  block(width * 4, width * 8, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(width * 8, n))
        self.input = 64

    def forward(self, x):
        return self.head(self.body(x))


def build_model(name, n, input_size):
    if name == 'smallcnn':
        return SmallCNN(n), 64
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=n)
    return m, input_size


def augment(x, train):
    """x: uint8 tensor [B,64,64,3] on device -> float [B,3,S,S] normalised (S set by caller)."""
    x = x.permute(0, 3, 1, 2).float() / 255.
    if train:
        B = x.shape[0]
        flip_h = torch.rand(B, device=x.device) < .5
        x = torch.where(flip_h[:, None, None, None], x.flip(3), x)
        flip_v = torch.rand(B, device=x.device) < .5
        x = torch.where(flip_v[:, None, None, None], x.flip(2), x)
        rot = torch.rand(B, device=x.device) < .5
        x = torch.where(rot[:, None, None, None], x.transpose(2, 3), x)
        # small translation with edge replication (a detector box is rarely centred on the object)
        dx, dy = random.randint(-4, 4), random.randint(-4, 4)
        if dx or dy:
            x = F.pad(x, (4, 4, 4, 4), mode='replicate')[:, :, 4 + dy:68 + dy, 4 + dx:68 + dx]
        # edge padding: the live hook replicates the view border into windows that cross it (BORDER_REPLICATE);
        # a quarter of the batch gets one side replaced by its own border strip so the streaks carry no evidence
        n_pad = B // 4
        if n_pad:
            k = random.randint(8, 28); side = random.randint(0, 3)
            xp = x[:n_pad].clone()
            if side == 0:
                xp[:, :, :, :k] = xp[:, :, :, k:k + 1]
            elif side == 1:
                xp[:, :, :, -k:] = xp[:, :, :, -k - 1:-k]
            elif side == 2:
                xp[:, :, :k, :] = xp[:, :, k:k + 1, :]
            else:
                xp[:, :, -k:, :] = xp[:, :, -k - 1:-k, :]
            x = torch.cat([xp, x[n_pad:]], 0)
        gain = (1 + (torch.rand(B, 1, 1, 1, device=x.device) - .5) * .3)
        bias = (torch.rand(B, 1, 1, 1, device=x.device) - .5) * .1
        x = (x * gain + bias).clamp(0, 1)
        x = (x + torch.randn_like(x) * .01).clamp(0, 1)
    return x


def prepare(x, size):
    if size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, X, size, device, bs=512, tta=True):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        x = prepare(augment(xb, False), size)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
            logits = model(x).float()
            if tta:
                logits = (logits + model(x.flip(3)).float() + model(x.flip(2)).float() + model(x.transpose(2, 3)).float()) / 4
        out.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def average_precision(scores, hits):
    order = np.argsort(-np.asarray(scores)); h = np.asarray(hits)[order]
    if h.sum() == 0:
        return float('nan')
    tp = np.cumsum(h); prec = tp / (np.arange(len(h)) + 1)
    return float((prec * h).sum() / h.sum())


def evaluate_test(rows, probs):
    """Test rows = detector boxes with truth. Reports AP of the detector alone vs re-ranked, per claimed class."""
    report = {}
    known = ('object', 'other_object', 'terrain')
    for c in SMALL:
        idx = [i for i, r in enumerate(rows) if r.get('_det_label') == c and r.get('_truth') in known]
        if not idx:
            continue
        det = np.array([rows[i]['_det_conf'] for i in idx]); truth = np.array([rows[i]['_truth'] == 'object' for i in idx])
        p_claim = probs[idx, CIDX[c]]; p_obj = 1 - probs[idx, CIDX['background']]
        rep = {'n': len(idx), 'objects': int(truth.sum()),
               'ap_detector': average_precision(det, truth), 'ap_reranked': average_precision(np.sqrt(det * p_claim), truth),
               'ap_verifier_only': average_precision(p_claim, truth), 'ap_pobject_rerank': average_precision(np.sqrt(det * p_obj), truth)}
        for thr in (0.2, 0.3, 0.5):
            keep = p_obj >= thr
            rep[f'gate{thr}_recall'] = float(keep[truth].mean()) if truth.any() else float('nan')
            rep[f'gate{thr}_terrain_removed'] = float((~keep)[~truth].mean()) if (~truth).any() else float('nan')
        report[c] = rep
    idx = [i for i, r in enumerate(rows) if r.get('_truth') in known]
    if idx:
        truth = np.array([rows[i]['_truth'] == 'object' for i in idx]); p_obj = 1 - probs[idx, CIDX['background']]
        try:
            from sklearn.metrics import roc_auc_score
            report['all'] = {'n': len(idx), 'auroc_pobject': float(roc_auc_score(truth, p_obj))}
        except Exception:
            pass
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', action='append', required=True, help='NAME=ROOT with index.jsonl')
    ap.add_argument('--marks', default=None, help='marks JSON to join det_conf / truth onto test rows by mark_id')
    ap.add_argument('--scene-test', default=None)
    ap.add_argument('--model', default='convnext_tiny')
    ap.add_argument('--input', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--real-oversample', type=int, default=6)
    ap.add_argument('--max-per', type=int, default=None, help='cap rows per (kind, class, split, level)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows = load_sources(a.source, a.marks, a.scene_test, a.max_per)
    train = [r for r in rows if r['split'] == 'train' and r['_label'] is not None and r['_level'] == 1]
    dev = [r for r in rows if r['split'] == 'dev' and r['_label'] is not None and r['_level'] == 1]
    test = [r for r in rows if r['split'] == 'test']
    train = train + [r for r in train if r.get('kind') == 'real'] * (a.real_oversample - 1)
    cnt = Counter(CLASSES[r['_label']] for r in train)
    print(f'rows: train {len(train)} {dict(cnt)} | dev {len(dev)} | test {len(test)} ({time.time()-t0:.0f}s)', flush=True)
    Xtr = load_images(train); ytr = np.array([r['_label'] for r in train])
    Xdev = load_images(dev) if dev else None; ydev = np.array([r['_label'] for r in dev]) if dev else None
    Xte = load_images(test) if test else None
    print(f'images loaded ({time.time()-t0:.0f}s)', flush=True)

    device = a.device if torch.cuda.is_available() else 'cpu'
    model, size = build_model(a.model, len(CLASSES), a.input)
    model = model.to(device)
    weights = torch.tensor([1 / math.sqrt(max(cnt[c], 1)) for c in CLASSES], dtype=torch.float32)
    weights = (weights / weights.mean()).to(device)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(Xtr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.startswith('cuda'))
    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
    history = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.; n = 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb = Xtr_t[idx].to(device, non_blocking=True); yb = ytr_t[idx].to(device)
            x = prepare(augment(xb, True), size)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                loss = crit(model(x).float(), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx); n += len(idx)
        rec = {'epoch': ep + 1, 'loss': tot / n}
        if Xdev is not None:
            p = predict(model, Xdev, size, device, tta=False); pred = p.argmax(1)
            rec['dev_acc'] = float((pred == ydev).mean())
            rec['dev_bg_as_object'] = float((pred[ydev == CIDX['background']] != CIDX['background']).mean()) if (ydev == CIDX['background']).any() else None
            rec['dev_object_recall'] = float((pred[ydev < 4] < 4).mean()) if (ydev < 4).any() else None
        history.append(rec); print(json.dumps(rec), flush=True)

    torch.save({'model': a.model, 'input': size, 'classes': CLASSES, 'state_dict': model.state_dict(), 'args': vars(a)}, out / 'model.pt')
    result = {'args': vars(a), 'train_counts': dict(cnt), 'history': history, 'seconds': time.time() - t0}
    if Xte is not None:
        probs = predict(model, Xte, size, device)
        np.save(out / 'test_probs.npy', probs)
        with open(out / 'test_rows.jsonl', 'w') as f:
            for r, p in zip(test, probs):
                f.write(json.dumps({'mark_id': r.get('mark_id'), 'file': r['file'], 'scene': r.get('scene'), 'frame': r.get('frame'),
                                    'det_label': r.get('_det_label'), 'det_conf': r.get('_det_conf'), 'truth': r.get('_truth'),
                                    'split_frame': r.get('_split_frame'), 'probs': [round(float(v), 5) for v in p]}) + '\n')
        result['test'] = evaluate_test(test, probs)
        print(json.dumps(result['test'], indent=1), flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    print(f'done in {time.time()-t0:.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
