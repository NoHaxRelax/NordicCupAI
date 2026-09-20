"""Which transformation of a model's word timestamps lands closest to the
annotated span edges?

For every annotated yes question, take the oracle-selected sentence merge (the
candidate with the best tIoU, as in span_ceiling.py), read its first and last
word, and fit candidate rules that map word times to the gold edge. Every rule
is fitted leave-one-conversation-out (LOCO): the constants for conversation c
are fitted on the other 38, so the reported error is out-of-sample.

Start rules (ws, we, dur = first word start/end/duration; gap_prev = silence
before the first word):
  const_start   gold = ws + a
  const_end     gold = we - a                (committee's rule)
  frac          gold = ws + f * dur
  linear        gold = ws + a + b * dur
  linear_gap    gold = ws + a + b * dur + c * gap_prev
End rules (ws, we, dur = last word; gap_next = silence after it):
  const_end     gold = we + a
  frac          gold = ws + f * dur
  linear        gold = we + a + b * dur
  linear_gap    gold = we + a + b * gap_next
  midgap        gold = we + f * gap_next     (fraction of the following pause)

Reports MAE per rule and the mean tIoU of the oracle merge with that rule
applied to both edges (best start rule x best end rule), so the edge error is
expressed in score terms.

    python bench/asr/fit_edges.py --model large-v3
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
sys.path.insert(0, str(CASE))
from span_ceiling import merges, sentences, tiou  # noqa: E402


def load(model):
    rows = [r for r in csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8'))
            if r['question_type'] == 'positive']
    cache = {}
    items = []
    for r in rows:
        tid = r['transcript_id']
        f = CASE / 'transcripts' / f'conversation_{tid}.{model}.json'
        if not f.exists():
            continue
        if tid not in cache:
            d = json.loads(f.read_text(encoding='utf-8'))
            words = [w for s in d['segments'] for w in s['words']]
            cache[tid] = (d, sentences(d), words)
        d, sents, words = cache[tid]
        g = (float(r['evidence_start']), float(r['evidence_end']))
        best, span = 0.0, None
        for (s0, s1, n) in merges(sents, 4):
            v = tiou(g, (s0, s1))
            if v > best:
                best, span = v, (s0, s1)
        if span is None:
            continue
        # first / last word of the selected merge
        idx0 = next(i for i, w in enumerate(words) if abs(w['start'] - span[0]) < 1e-6)
        idx1 = max(i for i, w in enumerate(words) if abs(w['end'] - span[1]) < 1e-6)
        w0, w1 = words[idx0], words[idx1]
        gap_prev = w0['start'] - words[idx0 - 1]['end'] if idx0 > 0 else 1.0
        gap_next = words[idx1 + 1]['start'] - w1['end'] if idx1 + 1 < len(words) else 1.0
        items.append({
            'conv': tid, 'g0': g[0], 'g1': g[1],
            'ws': w0['start'], 'we': w0['end'], 'dur0': w0['end'] - w0['start'], 'gap_prev': max(0.0, gap_prev),
            'ls': w1['start'], 'le': w1['end'], 'dur1': w1['end'] - w1['start'], 'gap_next': max(0.0, gap_next),
            'oracle': best,
        })
    return items


def fit_predict(items, features, target, base):
    """LOCO linear fit: target - base = X @ beta. Returns out-of-sample predictions."""
    convs = sorted({it['conv'] for it in items})
    pred = np.zeros(len(items))
    y = np.array([it[target] - it[base] for it in items])
    X = np.array([[1.0] + [it[f] for f in features] for it in items])
    conv = np.array([it['conv'] for it in items])
    for c in convs:
        tr, te = conv != c, conv == c
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ beta + np.array([it[base] for it in items])[te]
    beta_all, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pred, beta_all


def fit_frac(items, target, s, e):
    """gold = s + f*(e-s), f fitted LOCO as the median fraction."""
    convs = sorted({it['conv'] for it in items})
    conv = np.array([it['conv'] for it in items])
    frac = np.array([(it[target] - it[s]) / max(1e-3, it[e] - it[s]) for it in items])
    pred = np.zeros(len(items))
    for c in convs:
        f = np.median(frac[conv != c])
        for i in np.where(conv == c)[0]:
            pred[i] = items[i][s] + f * (items[i][e] - items[i][s])
    return pred, float(np.median(frac))


def fit_const(items, target, base):
    convs = sorted({it['conv'] for it in items})
    conv = np.array([it['conv'] for it in items])
    off = np.array([it[target] - it[base] for it in items])
    pred = np.zeros(len(items))
    for c in convs:
        a = np.median(off[conv != c])
        for i in np.where(conv == c)[0]:
            pred[i] = items[i][base] + a
    return pred, float(np.median(off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='large-v3')
    a = ap.parse_args()
    items = load(a.model)
    n = len(items)
    g0 = np.array([it['g0'] for it in items]); g1 = np.array([it['g1'] for it in items])
    print(f'{n} annotated spans, model {a.model}, oracle merge mean tIoU (raw edges) {st.mean(it["oracle"] for it in items):.3f}\n')

    starts = {}
    starts['raw ws'] = (np.array([it['ws'] for it in items]), '')
    p, c = fit_const(items, 'g0', 'ws'); starts['ws + a'] = (p, f'a={c:+.3f}')
    p, c = fit_const(items, 'g0', 'we'); starts['we + a'] = (p, f'a={c:+.3f}')
    p, f = fit_frac(items, 'g0', 'ws', 'we'); starts['ws + f*dur'] = (p, f'f={f:.2f}')
    p, b = fit_predict(items, ['dur0'], 'g0', 'ws'); starts['ws + a + b*dur'] = (p, f'a={b[0]:+.3f} b={b[1]:+.2f}')
    p, b = fit_predict(items, ['dur0', 'gap_prev'], 'g0', 'ws'); starts['ws + a + b*dur + c*gap_prev'] = (p, f'a={b[0]:+.3f} b={b[1]:+.2f} c={b[2]:+.2f}')

    ends = {}
    ends['raw le'] = (np.array([it['le'] for it in items]), '')
    p, c = fit_const(items, 'g1', 'le'); ends['le + a'] = (p, f'a={c:+.3f}')
    p, f = fit_frac(items, 'g1', 'ls', 'le'); ends['ls + f*dur'] = (p, f'f={f:.2f}')
    p, b = fit_predict(items, ['dur1'], 'g1', 'le'); ends['le + a + b*dur'] = (p, f'a={b[0]:+.3f} b={b[1]:+.2f}')
    p, b = fit_predict(items, ['gap_next'], 'g1', 'le'); ends['le + a + b*gap_next'] = (p, f'a={b[0]:+.3f} b={b[1]:+.2f}')
    p, f = fit_frac(items, 'g1', 'le', 'le_plus_gap'.replace('le_plus_gap', 'le')) if False else (None, None)
    # midgap: fraction of the following pause
    gapend = [dict(it, ge=it['le'] + it['gap_next']) for it in items]
    p, f = fit_frac(gapend, 'g1', 'le', 'ge'); ends['le + f*gap_next'] = (p, f'f={f:.2f}')

    def report(name, rules, gold):
        print(f'{name} rules (LOCO), MAE and share within 50 ms / 100 ms:')
        print(f'  {"rule":<34} {"MAE s":>7} {"<=50ms":>7} {"<=100ms":>8}  params')
        out = {}
        for k, (p, desc) in rules.items():
            err = np.abs(p - gold)
            out[k] = err.mean()
            print(f'  {k:<34} {err.mean():7.3f} {np.mean(err <= 0.05):7.0%} {np.mean(err <= 0.10):8.0%}  {desc}')
        print()
        return out

    ms = report('START', starts, g0)
    me = report('END', ends, g1)

    print('Mean tIoU of the oracle merge with a rule applied to both edges:')
    print(f'  {"start rule":<30} {"end rule":<26} {"tIoU":>6}')
    combos = [('raw ws', 'raw le'), ('ws + a', 'le + a')]
    bs = min(ms, key=ms.get); be = min(me, key=me.get)
    combos += [(bs, be), ('we + a', 'le + a'), ('ws + f*dur', 'ls + f*dur'), (bs, 'le + a'), ('ws + a', be)]
    seen = set()
    for s, e in combos:
        if (s, e) in seen:
            continue
        seen.add((s, e))
        ps, pe = starts[s][0], ends[e][0]
        vals = [tiou((it['g0'], it['g1']), (min(ps[i], pe[i] - 0.05), pe[i])) for i, it in enumerate(items)]
        print(f'  {s:<30} {e:<26} {st.mean(vals):6.3f}')


if __name__ == '__main__':
    main()
