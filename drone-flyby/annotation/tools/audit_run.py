#!/usr/bin/env python3
"""Audit a served run: re-score every small-class detector box with the verifier and label it with the truth.

For each frame of the run log, the raw detector boxes of the four small classes are cut from the captured view exactly
as the hook does, run through the ensemble, and matched to the truth (v8 labels + Elias's hidden tracks). Each box is also
matched to the frame's emitted answers (same class, IoU >= 0.5 in source px) to see whether it reached the output.
Writes audit.jsonl and a contact sheet of every TERRAIN box that reached the output (crop, class, confidence, p_object).

    python3 audit_run.py --log <run>/frames/<seq>.jsonl --captures <run>/frames/captures --models m1.pt,m2.pt \
        --v8 labels-v8.json --hidden2 validation_hidden2.json --hidden validation_hidden.json --out audit
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verifier_hook import Verifier, _cut  # noqa: E402

W, H = 3840, 2160
SMALL = ['small_launcher', 'medium_launcher', 'ta-ta', 'jammer']
FACTOR = {0: 4, 1: 2, 2: 1}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_labels(v8_path, hidden2_path):
    labels = defaultdict(list)
    v8 = json.loads(Path(v8_path).read_text())['predictions_by_frame']
    for fr, rows in v8.items():
        for o in rows:
            b = o['bbox']; labels[int(fr)].append((o['object_id'], [b[0] * W, b[1] * H, b[2] * W, b[3] * H], 'v8'))
    h2 = json.loads(Path(hidden2_path).read_text())
    for fr, boxes in h2['zones'].items():
        f = int(fr); active = [t for t in h2['tracks'] if t['kept_out'][0] <= f <= t['kept_out'][1]]
        for t, b in zip(active, boxes) if len(active) == len(boxes) else []:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; w, h = t['size']
            labels[f].append((t['what'] if t['what'] in SMALL or t['what'] in ('small_tower', 'tank') else 'unknown', [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 'hidden2'))
    return labels


def truth_of(cls, sb, lab, hidden1, frame):
    cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
    best = max(((iou(sb, lb), lc) for lc, lb, _ in lab), default=(0.0, None))
    in_zone = [lc for lc, lb, src in lab if src == 'hidden2' and lb[0] - 20 <= cx <= lb[2] + 20 and lb[1] - 20 <= cy <= lb[3] + 20]
    z1 = hidden1.get(str(frame)); in_row = bool(z1) and z1[0] <= cx <= z1[2] and z1[1] <= cy <= z1[3]
    near = any(abs((lb[0] + lb[2]) / 2 - cx) < 64 and abs((lb[1] + lb[3]) / 2 - cy) < 64 for _, lb, _ in lab)
    if best[0] >= 0.3 and best[1] == cls or cls in in_zone:
        return 'object'
    if best[0] >= 0.3 or in_zone or in_row:
        return 'other_object'
    return 'unknown' if near else 'terrain'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', required=True); ap.add_argument('--captures', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--v8', required=True); ap.add_argument('--hidden2', required=True); ap.add_argument('--hidden', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.v8, a.hidden2); hidden1 = json.loads(Path(a.hidden).read_text())['zones']
    ver = Verifier(a.models.split(','), device=a.device)
    ci = {c: k for k, c in enumerate(ver.classes)}
    seq = Path(a.log).stem
    rows = [json.loads(l) for l in open(a.log)]
    audit = []; crops_all = []; metas = []
    for r in rows:
        dets = [d for d in r.get('raw_detections') or [] if d['label'] in SMALL]
        if not dets:
            continue
        frame, level, region = int(r['frame']), int(r['level']), r['region']; f = FACTOR[level]
        img = cv2.imread(str(Path(a.captures) / f'{seq}-{frame:06d}.png'))
        if img is None:
            continue
        lab = labels.get(frame, [])
        answers = [(x['object_id'], [x['bbox'][0] * W, x['bbox'][1] * H, x['bbox'][2] * W, x['bbox'][3] * H], x['confidence']) for x in r['response']]
        for d in dets:
            b = d['box']; sb = [region[0] + b[0] * f, region[1] + b[1] * f, region[0] + b[2] * f, region[1] + b[3] * f]
            crops_all.append(_cut(img, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, level))
            emitted = max(((iou(sb, ab), ac) for al, ab, ac in answers if al == d['label']), default=(0.0, None))
            metas.append(dict(frame=frame, level=level, cls=d['label'], conf=float(d['confidence']), box_source=[round(v, 1) for v in sb],
                              truth=truth_of(d['label'], sb, lab, hidden1, frame), emitted=bool(emitted[0] >= 0.5), emitted_conf=emitted[1]))
    probs = ver(np.stack(crops_all)) if crops_all else np.zeros((0, 6))
    for m, p, c in zip(metas, probs, crops_all):
        m['p_object'] = float(1 - p[ci['background']]); m['p_claimed'] = float(p[ci[m['cls']]]); m['argmax'] = ver.classes[int(np.argmax(p))]
        audit.append(m)
    with open(out / 'audit.jsonl', 'w') as fh:
        for m in audit:
            fh.write(json.dumps(m) + '\n')
    # summary
    print(f'{len(audit)} small-class detector boxes in {len(rows)} frames')
    for c in SMALL:
        rows_c = [m for m in audit if m['cls'] == c]
        if not rows_c:
            continue
        print(f'=== {c}: {len(rows_c)} boxes')
        for truth in ('object', 'terrain', 'other_object', 'unknown'):
            sel = [m for m in rows_c if m['truth'] == truth]
            if not sel:
                continue
            passed = [m for m in sel if m['p_object'] >= 0.5]; em = [m for m in sel if m['emitted']]
            print(f"  {truth:13s} n={len(sel):3d} p_object>=0.5: {len(passed):3d}  emitted: {len(em):3d}  emitted&passed: {sum(1 for m in em if m['p_object']>=0.5):3d}  emitted&gated: {sum(1 for m in em if m['p_object']<0.5):3d}")
    # contact sheet of terrain boxes that reached the output, sorted by emitted confidence
    leaks = [(m, c) for m, c in zip(audit, crops_all) if m['truth'] == 'terrain' and m['emitted']]
    leaks.sort(key=lambda mc: -(mc[0]['emitted_conf'] or 0))
    if leaks:
        tile = 96; cols = 8; rows_n = (len(leaks) + cols - 1) // cols
        sheet = np.zeros((rows_n * (tile + 28), cols * tile, 3), np.uint8)
        for i, (m, c) in enumerate(leaks):
            y, x = divmod(i, cols); im = cv2.resize(c[:, :, ::-1], (tile, tile), interpolation=cv2.INTER_NEAREST)
            sheet[y * (tile + 28):y * (tile + 28) + tile, x * tile:(x + 1) * tile] = im
            cv2.putText(sheet, f"f{m['frame']} {m['cls'][:6]} c{m['emitted_conf']:.2f}", (x * tile + 2, y * (tile + 28) + tile + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"pobj {m['p_object']:.2f} {m['argmax'][:6]}", (x * tile + 2, y * (tile + 28) + tile + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / 'terrain_leaks.png'), sheet)
        print(f'terrain boxes that reached the output: {len(leaks)} -> {out/"terrain_leaks.png"}')


if __name__ == '__main__':
    main()
