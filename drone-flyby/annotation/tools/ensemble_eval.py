#!/usr/bin/env python3
"""Score the verifier ensemble on the test rows written by train_verifier.py.

Members are run directories holding test_rows.jsonl (one row per detector box, with probs). Probabilities are
combined by mean log-odds per class. Reports, per claimed class: AP of the detector alone, AP after re-ranking
(geometric mean of detector confidence and p(claimed class)), and for a sweep of p(object) gate thresholds the
share of true objects kept and of terrain boxes removed, on all frames and on frames outside the training split
(dev + reserved), where the terrain is unseen by the verifier.

    python3 ensemble_eval.py runs/m1-convnext runs/m2-smallcnn [--out ensemble.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ['small_launcher', 'medium_launcher', 'ta-ta', 'jammer', 'other', 'background']
SMALL = CLASSES[:4]
KNOWN = ('object', 'other_object', 'terrain')


def average_precision(scores, hits):
    order = np.argsort(-np.asarray(scores)); h = np.asarray(hits)[order]
    if h.sum() == 0:
        return float('nan')
    tp = np.cumsum(h); prec = tp / (np.arange(len(h)) + 1)
    return float((prec * h).sum() / h.sum())


def combine(prob_sets):
    logits = [np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1)) for p in prob_sets]
    m = np.mean(logits, axis=0)
    p = 1 / (1 + np.exp(-m))
    return p / p.sum(axis=1, keepdims=True)


def report(rows, probs, frames_filter=None):
    out = {}
    sel = [i for i, r in enumerate(rows) if r['truth'] in KNOWN and (frames_filter is None or frames_filter(r))]
    for c in SMALL:
        idx = [i for i in sel if r_det(rows[i]) == c]
        if not idx:
            continue
        det = np.array([rows[i]['det_conf'] for i in idx]); truth = np.array([rows[i]['truth'] == 'object' for i in idx])
        p_claim = probs[idx, CLASSES.index(c)]; p_obj = 1 - probs[idx, CLASSES.index('background')]
        rep = {'n': len(idx), 'objects': int(truth.sum()), 'ap_detector': average_precision(det, truth),
               'ap_reranked': average_precision(np.sqrt(det * p_claim), truth), 'ap_verifier_only': average_precision(p_claim, truth)}
        for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
            keep = p_obj >= thr
            rep[f'gate{thr}'] = {'objects_kept': float(keep[truth].mean()) if truth.any() else None,
                                 'terrain_removed': float((~keep)[~truth].mean()) if (~truth).any() else None}
        out[c] = rep
    if sel:
        truth = np.array([rows[i]['truth'] == 'object' for i in sel]); p_obj = 1 - probs[sel, CLASSES.index('background')]
        try:
            from sklearn.metrics import roc_auc_score
            out['all'] = {'n': len(sel), 'auroc_pobject': float(roc_auc_score(truth, p_obj))}
        except Exception:
            out['all'] = {'n': len(sel)}
    return out


def r_det(r):
    return r['det_label']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('members', nargs='+')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows_by_member = []
    for m in a.members:
        rows_by_member.append([json.loads(l) for l in open(Path(m) / 'test_rows.jsonl')])
    ids = [r['mark_id'] for r in rows_by_member[0]]
    for rows in rows_by_member[1:]:
        assert [r['mark_id'] for r in rows] == ids, 'members must share the test set'
    rows = rows_by_member[0]
    probs = combine([np.array([r['probs'] for r in rows]) for rows in rows_by_member])
    result = {'members': a.members, 'all_frames': report(rows, probs),
              'unseen_frames': report(rows, probs, lambda r: r.get('split_frame') in ('dev', 'reserved'))}
    for name in ('all_frames', 'unseen_frames'):
        print(f'=== {name}')
        for c, v in result[name].items():
            if c == 'all':
                print('  all', v); continue
            g3, g5 = v['gate0.3'], v['gate0.5']
            print(f"  {c:16s} n={v['n']:3d} obj={v['objects']:3d} AP det {v['ap_detector']:.3f} -> rerank {v['ap_reranked']:.3f} "
                  f"| gate .3 keeps {g3['objects_kept']} removes {g3['terrain_removed']} | gate .5 keeps {g5['objects_kept']} removes {g5['terrain_removed']}")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
