"""Verifier stage behind the endpoint's detector hook: re-scores the small-class boxes of the YOLO detector.

    DRONE_DETECTOR=verifier_hook:build DRONE_WEIGHTS=both_m1280.pt DRONE_IMGSZ=1280 DRONE_CONF=0.05 \
    VERIFIER_MODELS=/workspace/verifier/runs/m1-convnext/model.pt,/workspace/verifier/runs/m2-smallcnn/model.pt \
    VERIFIER_GATE='{"small_launcher":0.3,"medium_launcher":0.3,"ta-ta":0.3,"jammer":0.3}' python api.py

For every detector box whose label is one of the verified classes, a 128-source-px window around the box centre is
cut from the delivered view (128 / level factor delivered px, edge-replicated at the border) and brought to the
64 px L1 look, exactly as the training crops were made. The ensemble (mean log-odds over the members, optional
flip/transpose TTA) gives p(claimed class) and p(object) = 1 - p(background).
  re-rank: confidence' = sqrt(confidence * p(claimed class))          (never deletes)
  gate:    if p(object) < gate[label], confidence' = min(confidence', VERIFIER_GATE_FLOOR) so the tracker does not
           give birth to a track from it (below DRONE_BIRTH_CONFIDENCE, above the detector floor)
Other classes pass through untouched. Rows keep their box; a 'verifier' field carries p(object) and p(claimed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

FACTOR = {0: 4, 1: 2, 2: 1}
WINDOW = 128


def _cut(image, cx, cy, level):
    f = FACTOR[int(level)]
    w = WINDOW // f
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - w / 2))
    h, iw = image.shape[:2]
    px0, py0, px1, py1 = max(0, -x0), max(0, -y0), max(0, x0 + w - iw), max(0, y0 + w - h)
    if px0 or py0 or px1 or py1:
        image = cv2.copyMakeBorder(image, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
        x0 += px0; y0 += py0
    crop = image[y0:y0 + w, x0:x0 + w]
    if w != 64:
        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA if w > 64 else cv2.INTER_CUBIC)
    return crop[:, :, ::-1]  # BGR -> RGB


def _padded(image, cx, cy, level):
    """True when the window crosses the delivered view border (the crop then carries replicated stripes)."""
    w = WINDOW // FACTOR[int(level)]; h, iw = image.shape[:2]
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - w / 2))
    return x0 < 0 or y0 < 0 or x0 + w > iw or y0 + w > h


class Verifier:
    def __init__(self, model_paths, device='cuda:0', tta=True):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_verifier import SmallCNN, CLASSES, MEAN, STD  # noqa
        self.torch, self.classes, self.device, self.tta = torch, CLASSES, device, tta
        self.mean, self.std = MEAN.to(device), STD.to(device)
        self.members = []
        for p in model_paths:
            ck = torch.load(p, map_location='cpu')
            if ck['model'] == 'smallcnn':
                model = SmallCNN(len(ck['classes']))
            else:
                import timm
                model = timm.create_model(ck['model'], pretrained=False, num_classes=len(ck['classes']))
            model.load_state_dict(ck['state_dict']); model.to(device).eval()
            self.members.append((model, int(ck['input'])))
        # Fixed batch shapes: crops are padded to a multiple of BATCH so cuDNN never meets a new shape live
        # (a new shape cost 4.6 s once in a replay). Warm every shape up to 4 x BATCH now.
        torch.backends.cudnn.benchmark = False
        for n in (self.BATCH, 2 * self.BATCH, 4 * self.BATCH):
            self(np.zeros((n, 64, 64, 3), np.uint8))

    BATCH = 16

    def __call__(self, crops):
        """crops: uint8 [N,64,64,3] RGB -> probs [N, C] (mean log-odds over members)."""
        torch = self.torch
        n = len(crops)
        if n == 0:
            return np.zeros((0, len(self.classes)))
        pad = (-n) % self.BATCH
        if pad:
            crops = np.concatenate([crops, np.zeros((pad, 64, 64, 3), np.uint8)])
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(crops)).to(self.device).permute(0, 3, 1, 2).float() / 255.
            logit_sum = None
            for model, size in self.members:
                xi = x if size == 64 else torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
                xi = (xi - self.mean) / self.std
                views = [xi] + ([xi.flip(3), xi.flip(2), xi.transpose(2, 3)] if self.tta else [])
                with torch.autocast('cuda', dtype=torch.float16, enabled=str(self.device).startswith('cuda')):
                    p = sum(model(v).float().softmax(1) for v in views) / len(views)
                lo = torch.log(p.clamp(1e-6, 1 - 1e-6)) - torch.log((1 - p).clamp(1e-6, 1))
                logit_sum = lo if logit_sum is None else logit_sum + lo
            p = torch.sigmoid(logit_sum / len(self.members))
            p = p / p.sum(1, keepdim=True)
        return p.cpu().numpy()[:n]


class VerifiedDetector:
    name = 'verified'

    def __init__(self, base, verifier, classes, gate, floor=0.1, tta=True, gate_padded=0.8):
        self.base, self.verifier, self.classes, self.gate, self.floor = base, verifier, set(classes), gate, floor
        self.gate_padded = gate_padded  # stricter p(object) gate for windows that cross the view border
        self.stats = {'boxes': 0, 'verified': 0, 'gated': 0, 'padded': 0}

    def __call__(self, image, request):
        rows = self.base(image, request)
        level = int((request.get('view') or {}).get('resolution_level', 1)) if isinstance(request, dict) else 1
        idx = [i for i, r in enumerate(rows) if r['label'] in self.classes]
        self.stats['boxes'] += len(rows)
        if not idx:
            return rows
        crops = np.stack([_cut(image, (rows[i]['box'][0] + rows[i]['box'][2]) / 2, (rows[i]['box'][1] + rows[i]['box'][3]) / 2, level) for i in idx])
        probs = self.verifier(crops)
        ci = {c: k for k, c in enumerate(self.verifier.classes)}
        for i, p in zip(idx, probs):
            r = rows[i]; conf = float(r['confidence'])
            p_claim = float(p[ci[r['label']]]); p_obj = float(1 - p[ci['background']])
            new = float(np.sqrt(conf * p_claim))
            padded = _padded(image, (r['box'][0] + r['box'][2]) / 2, (r['box'][1] + r['box'][3]) / 2, level)
            gate = max(self.gate.get(r['label'], 0.0), self.gate_padded) if padded else self.gate.get(r['label'], 0.0)
            if padded:
                self.stats['padded'] += 1
            if p_obj < gate:
                new = min(new, self.floor); self.stats['gated'] += 1
            r['verifier'] = {'p_object': round(p_obj, 4), 'p_claimed': round(p_claim, 4), 'original': round(conf, 4), 'padded': padded}
            r['confidence'] = new
            self.stats['verified'] += 1
        return rows


def build():
    """Factory for DRONE_DETECTOR=verifier_hook:build: wraps the endpoint's own ultralytics detector."""
    import detectors
    env = dict(os.environ); env['DRONE_DETECTOR'] = os.environ.get('VERIFIER_BASE', 'ultralytics')
    base = detectors.build_detector(env)
    paths = [p for p in os.environ.get('VERIFIER_MODELS', '').split(',') if p]
    if not paths:
        raise ValueError('verifier_hook:build needs VERIFIER_MODELS=a.pt,b.pt')
    classes = [c for c in os.environ.get('VERIFIER_CLASSES', 'small_launcher,medium_launcher,ta-ta,jammer').split(',') if c]
    gate = json.loads(os.environ.get('VERIFIER_GATE', '{}') or '{}')
    verifier = Verifier(paths, device=os.environ.get('DRONE_DEVICE', 'cuda:0'), tta=os.environ.get('VERIFIER_TTA', '1') == '1')
    return VerifiedDetector(base, verifier, classes, gate, floor=float(os.environ.get('VERIFIER_GATE_FLOOR', '0.1')),
                            gate_padded=float(os.environ.get('VERIFIER_GATE_PADDED', '0.8')))
