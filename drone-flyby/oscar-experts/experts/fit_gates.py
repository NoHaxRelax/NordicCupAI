"""Fit one explainable logistic gate per class over the expert's recorded candidate features.

Training data: candidates from evaluate.py runs on TRAINING tiles. A candidate is positive when its
box overlaps a label of the class (IoU >= .5); candidates on empty tiles and non-overlapping
candidates on labelled tiles are negatives. Tiles are grouped for a 5-fold cross-validation so the
recall/background numbers are out-of-fold. The threshold is chosen at the requested recall on the
out-of-fold positives. Output: per-class standardisation, weights (readable), threshold, and the
out-of-fold background-kept rate. No dev or reserved tiles are read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import iou

EXCLUDE = {'cx', 'cy', 'score', 'confidence', 'proposer_heading', 'angle', 'fitted_scale', 'sprite_chroma', 'component_area', 'partial'}


def features_of(candidate):
    out = {}
    for k, v in candidate.items():
        if k in EXCLUDE or k.endswith('_error'):
            continue
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)) and np.isfinite(v):
            out[k] = float(v)
        elif isinstance(v, dict) and k == 'competitor_scores':
            for name, val in v.items():
                if isinstance(val, (int, float)):
                    out[f'competitor_{name}'] = float(val)
    return out


def collect(report, with_templates=False):
    rows, labels, groups, templates = [], [], [], []
    for crop in report['crops']:
        targets = [t['bbox'] for t in crop['targets']]
        for c in crop['candidates']:
            if c.get('rejected_by') or 'bbox' not in c:
                continue
            best = max((iou(c['bbox'], t) for t in targets), default=0.)
            if .2 < best < .5:
                continue  # near miss: neither a clean positive nor a clean negative
            f = features_of(c); f['zoom'] = float(crop['zoom'])
            rows.append(f); labels.append(1 if best >= .5 else 0); groups.append(crop['id']); templates.append(str(c.get('template_id', '')))
    for e in report.get('empty', []):
        for c in e.get('candidates', []):
            if c.get('rejected_by') or 'bbox' not in c:
                continue
            f = features_of(c); f['zoom'] = float(e['zoom'])
            rows.append(f); labels.append(0); groups.append(e['id']); templates.append(str(c.get('template_id', '')))
    if with_templates:
        return rows, np.array(labels), np.array(groups), np.array(templates)
    return rows, np.array(labels), np.array(groups)


def fit_gate(rows, y, groups, keys, recall, folds):
    """One standardised logistic gate with an out-of-fold threshold at the recall target. Returns (gate dict, oof scores)."""
    X = design(rows, keys)
    mean, std = X.mean(0), np.maximum(X.std(0), 1e-6)
    Xs = (X - mean) / std
    unique = np.unique(groups); rng = np.random.default_rng(1731); rng.shuffle(unique)
    fold_of = {g: i % folds for i, g in enumerate(unique)}
    fold = np.array([fold_of[g] for g in groups])
    oof = np.zeros(len(y))
    for f in range(folds):
        train = fold != f
        if y[train].sum() < 4 or (~train).sum() == 0:
            continue
        w = fit_logistic(Xs[train], y[train])
        oof[~train] = np.hstack([Xs[~train], np.ones((int((~train).sum()), 1))]) @ w
    pos_scores = np.sort(oof[y == 1])
    threshold = float(pos_scores[max(0, int((1 - recall) * len(pos_scores)))])
    kept_bg = float((oof[y == 0] >= threshold).mean()) if (y == 0).any() else 0.
    recall_oof = float((oof[y == 1] >= threshold).mean())
    w = fit_logistic(Xs, y)
    weights = {k: round(float(v), 3) for k, v in zip(keys + ['zoom0', 'zoom1', 'zoom2'], w[:-1])}
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:6]
    gate = dict(keys=keys, mean=mean.tolist(), std=std.tolist(), weights=w.tolist(), threshold=threshold,
                positives=int(y.sum()), negatives=int((y == 0).sum()), oof_recall=recall_oof, oof_background_kept=kept_bg, top_weights=top)
    return gate, oof


def design(rows, keys):
    X = np.zeros((len(rows), len(keys) + 3), np.float32)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            X[i, j] = r.get(k, 0.)
        z = int(r.get('zoom', 1))
        X[i, len(keys) + z] = 1.
    return X


def fit_logistic(X, y, l2=1e-2, iterations=600, lr=.5):
    w = np.zeros(X.shape[1] + 1, np.float64)
    Xb = np.hstack([X, np.ones((len(X), 1), np.float32)]).astype(np.float64)
    pos_weight = max(1., (y == 0).sum() / max(1, (y == 1).sum()))
    sample_weight = np.where(y == 1, pos_weight, 1.)
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ ((p - y) * sample_weight) / len(y) + l2 * np.r_[w[:-1], 0.]
        w -= lr * grad
    return w


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recall', type=float, default=.98)
    p.add_argument('--folds', type=int, default=5)
    a = p.parse_args()
    gates = {}
    for report_path in sorted(a.runs.glob('*/report.json')):
        report = json.loads(report_path.read_text())
        name = report['class_name']
        rows, y, groups, templates = collect(report, with_templates=True)
        if y.sum() < 8:
            print(f'{name}: only {int(y.sum())} positives, no gate'); continue
        keys = sorted({k for r in rows for k in r if k != 'zoom'})
        gate, oof = fit_gate(rows, y, groups, keys, a.recall, a.folds)
        # Per-template sub-gates: with several sprites per class the features of true candidates spread out
        # (each sprite has its own correlation and colour statistics) and one class-wide gate lets much more
        # background through (large launcher: 15% -> 58% with six sprites). A gate per sprite keeps them tight.
        by_template, combined = {}, oof.copy()
        for t in sorted(set(templates)):
            sel = templates == t
            if t and y[sel].sum() >= 8 and (y[sel] == 0).sum() >= 40:
                sub, sub_oof = fit_gate([r for r, m in zip(rows, sel) if m], y[sel], groups[sel], keys, a.recall, a.folds)
                # keep the sprite's gate only where it beats the class gate on this sprite's own out-of-fold rows
                # (a split with few positives can be looser than the class gate: jet plane .11 -> .41 otherwise)
                class_kept = float((oof[sel & (y == 0)] >= gate['threshold']).mean()) if (sel & (y == 0)).any() else 0.
                if sub['oof_background_kept'] < class_kept:
                    by_template[t] = sub
                    combined[sel] = sub_oof - sub['threshold'] + gate['threshold']  # express on the class gate's threshold scale
        if by_template:
            gate['by_template'] = by_template
            kept_bg = float((combined[y == 0] >= gate['threshold']).mean()) if (y == 0).any() else 0.
            gate['oof_background_kept_with_templates'] = kept_bg
            gate['oof_recall_with_templates'] = float((combined[y == 1] >= gate['threshold']).mean())
        gates[name] = gate
        extra = f" | per-template ({len(by_template)}): recall {gate['oof_recall_with_templates']:.3f}, background kept {gate['oof_background_kept_with_templates']:.3f}" if by_template else ''
        print(f"{name:16s} pos {gate['positives']:4d} neg {gate['negatives']:5d} | out-of-fold recall {gate['oof_recall']:.3f}, background kept {gate['oof_background_kept']:.3f}{extra} | top: {gate['top_weights']}")
    a.output.write_text(json.dumps(dict(format='expert-gates-v1', recall_target=a.recall, source=str(a.runs), gates=gates), indent=1))


def apply_gate(gate, candidate, zoom):
    sub = gate.get('by_template', {}).get(str(candidate.get('template_id', '')))
    if sub is not None:
        gate = sub  # the sprite's own gate when it had enough training candidates; class gate otherwise
    f = features_of(candidate); f['zoom'] = float(zoom)
    x = design([f], gate['keys'])[0]
    xs = (x - np.array(gate['mean'])) / np.array(gate['std'])
    logit = float(np.dot(np.r_[xs, 1.], np.array(gate['weights'])))
    return logit, logit >= gate['threshold']


if __name__ == '__main__':
    main()
