#!/usr/bin/env python3
"""Score the boxes a replay EMITTED (after the tracker) for the four small classes, with the test-set truth.

Reads the per-frame jsonl written by the endpoint (DRONE_LOG_DIR) and, for every emitted box of a small class,
decides object / other_object / terrain / unknown exactly like build_marks.py does for detector boxes (v8 labels,
Elias's hidden tracks). Reports per class and per frame block: emitted boxes, how many are objects, AP over the
emitted boxes ranked by confidence, and recall of the labelled instances (label boxes matched by an emitted box of
the same class at IoU 0.3). Compare a baseline replay with a verified one on the same frames.

    python3 score_emitted.py replays/baseline replays/verified --v8 ... --hidden2 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_marks import load_labels, iou, split_of, SMALL, W, H  # noqa: E402


def average_precision(scores, hits):
    order = np.argsort(-np.asarray(scores)); h = np.asarray(hits)[order]
    if h.sum() == 0:
        return float('nan')
    tp = np.cumsum(h); prec = tp / (np.arange(len(h)) + 1)
    return float((prec * h).sum() / h.sum())


def load_replay(folder):
    rows = []
    for f in sorted(Path(folder).glob('*.jsonl')):
        for line in open(f):
            r = json.loads(line)
            if 'response' in r and 'frame' in r:
                rows.append(r)
    return rows


def score(rows, labels, hidden1, blocks):
    out = {}
    for name, keep in blocks.items():
        per = {c: {'scores': [], 'hits': [], 'kinds': Counter()} for c in SMALL}
        matched = defaultdict(set); instances = defaultdict(set)
        for r in rows:
            frame = int(r['frame'])
            if not keep(split_of('validation', frame)):
                continue
            lab = labels.get(('validation', frame), [])
            for k, (lc, lb, src) in enumerate(lab):
                if lc in SMALL:
                    instances[lc].add((frame, k))
            for a in r['response']:
                c = a['object_id']
                if c not in SMALL:
                    continue
                b = a['bbox']; sb = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
                cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
                best = (0.0, None, None)
                for k, (lc, lb, src) in enumerate(lab):
                    v = iou(sb, lb)
                    if v > best[0]:
                        best = (v, lc, k)
                in_zone = [lc for lc, lb, src in lab if src == 'hidden2' and lb[0] - 20 <= cx <= lb[2] + 20 and lb[1] - 20 <= cy <= lb[3] + 20]
                z1 = hidden1.get(str(frame)); in_plane_row = bool(z1) and z1[0] <= cx <= z1[2] and z1[1] <= cy <= z1[3]
                near = any(abs((lb[0] + lb[2]) / 2 - cx) < 64 and abs((lb[1] + lb[3]) / 2 - cy) < 64 for _, lb, _ in lab)
                if best[0] >= 0.3 and best[1] == c or c in in_zone:
                    truth = 'object'
                    if best[0] >= 0.3 and best[1] == c:
                        matched[c].add((frame, best[2]))
                elif best[0] >= 0.3 or in_zone or in_plane_row:
                    truth = 'other_object'
                elif near:
                    truth = 'unknown'
                else:
                    truth = 'terrain'
                per[c]['kinds'][truth] += 1
                if truth != 'unknown':
                    per[c]['scores'].append(float(a['confidence'])); per[c]['hits'].append(truth == 'object')
        rep = {}
        for c in SMALL:
            p = per[c]
            rep[c] = {'emitted': sum(p['kinds'].values()), 'kinds': dict(p['kinds']), 'ap_emitted': average_precision(p['scores'], p['hits']) if p['scores'] else float('nan'),
                      'label_instances': len(instances[c]), 'label_recall': (len(matched[c]) / len(instances[c])) if instances[c] else float('nan')}
        out[name] = rep
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('replays', nargs='+')
    ap.add_argument('--v8', required=True)
    ap.add_argument('--hidden', required=True)
    ap.add_argument('--hidden2', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    labels = load_labels(a.v8, a.hidden2)
    hidden1 = json.loads(Path(a.hidden).read_text())['zones']
    blocks = {'all': lambda s: True, 'unseen(dev+reserved)': lambda s: s in ('dev', 'reserved')}
    result = {}
    for folder in a.replays:
        rows = load_replay(folder)
        result[folder] = score(rows, labels, hidden1, blocks)
        print(f'##### {folder}: {len(rows)} frames')
        for name, rep in result[folder].items():
            print(f'=== {name}')
            for c, v in rep.items():
                print(f"  {c:16s} emitted {v['emitted']:4d} {v['kinds']} AP {v['ap_emitted']:.3f} | labelled instances {v['label_instances']} recall {v['label_recall']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
