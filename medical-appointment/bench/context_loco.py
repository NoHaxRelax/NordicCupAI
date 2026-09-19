"""Leave-one-conversation-out test of edge edits to the 27B's citation.

Entry 63 showed that adding the unit in front of the citation helps on 17 of the 195 gold-yes training
questions and hurts on 162, and that every hand-written trigger loses. This script asks whether a
FITTED trigger does better when it is evaluated honestly: for each of the 39 training conversations,
fit on the other 38, apply to the held-out one, pool the 195 decisions.

Four edits: add-front (one more unit in front), drop-front (remove the first unit of a citation of
two or more), add-back, drop-back. Features come from the citation text, the question and the
neighbouring units only, never from the gold interval. Two fitted policies per edit:
  logistic    P(edit helps); apply when P x mean gain > (1 - P) x mean loss, both means from the fold
  ridge       predicted tIoU change; apply when it is positive
plus a combined policy that picks, per question, the edit with the largest predicted change.
Reported: mean tIoU of the pooled held-out decisions against the served 0.7119, fires / wins / losses,
the pooled held-out AUC of "the edit helps", and a conversation-clustered bootstrap interval.

    python bench/context_loco.py            # prints the tables, writes research/14-context-loco.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import context_pattern as cp  # noqa: E402  loads the served run, rebuilds the units, asserts the spans
from context_pattern import content_words, span_from_ids, tiou  # noqa: E402

OUT = cp.CASE / 'research' / '14-context-loco.md'
ACC = cp.ACC
EDITS = ['add-front', 'drop-front', 'add-back', 'drop-back']
RNG = np.random.default_rng(0)


def load():
    """Rows of context_pattern.main() plus the units and the gold interval of each question."""
    d = cp.json.loads(cp.RESULT.read_text(encoding='utf-8'))
    gold = {q['question_id']: q['gold'] for q in d['questions'] if q['label'] == 1}
    rows = cp.main()
    convs = {}
    for r in rows:
        if r['tid'] not in convs:
            words, duration = cp.load_words(cp.CASE / 'transcripts' / f"conversation_{r['tid']}.large-v3-turbo.json")
            convs[r['tid']] = (cp.make_units(words, 'clause-and'), duration)
        r['units'], r['dur'] = convs[r['tid']]
        r['gold'] = gold[r['id']]
    return rows


def unit_feats(u, qw, other_text):
    """Features of one neighbouring or edge unit relative to the question and the rest of the citation."""
    t = u.text.strip()
    ws = t.split()
    first = re.sub(r"[^a-z']", '', ws[0].lower()) if ws else ''
    cw = content_words(t)
    return [len(ws), float(t.endswith('?')), float(t.endswith(',')), float(len(ws) <= 3),
            len(qw & cw), len((qw & cw) - content_words(other_text)),
            float(first in ('yes', 'no', 'yeah', 'okay', 'exactly')), float(first in ('and', 'but', 'so', 'then')),
            float(first in cp.PRONOUNS)]


def features(r, edit):
    """Feature vector and the edited span, or None when the edit does not apply."""
    units, (c0, c1) = r['units'], r['chosen']
    qw = content_words(r['q'])
    chosen = r['chosen_text']
    ws = chosen.split()
    first = r['first']
    base = [c1 - c0 + 1, len(ws), float(first in cp.PRONOUNS), float(first in ('it', "it's", 'that', 'this', 'they')),
            float(chosen[:1].islower()), len(qw & content_words(chosen)) / max(1, len(qw)), len(qw - content_words(chosen)),
            float(chosen.rstrip().endswith('?')), float(first in ('and', 'but', 'so'))]
    if edit == 'add-front':
        if c0 == 0:
            return None, None
        p = units[c0 - 1]
        f = base + unit_feats(p, qw, chosen) + [units[c0].start - p.end]
        return f, span_from_ids(list(range(c0 - 1, c1 + 1)), units, r['dur'])
    if edit == 'drop-front':
        if c1 == c0:
            return None, None
        rest = cp.utext(units, c0 + 1, c1)
        f = base + unit_feats(units[c0], qw, rest) + [units[c0 + 1].start - units[c0].end]
        return f, span_from_ids(list(range(c0 + 1, c1 + 1)), units, r['dur'])
    if edit == 'add-back':
        if c1 + 1 >= len(units):
            return None, None
        n = units[c1 + 1]
        f = base + unit_feats(n, qw, chosen) + [n.start - units[c1].end]
        return f, span_from_ids(list(range(c0, c1 + 2)), units, r['dur'])
    if edit == 'drop-back':
        if c1 == c0:
            return None, None
        rest = cp.utext(units, c0, c1 - 1)
        f = base + unit_feats(units[c1], qw, rest) + [units[c1].start - units[c1 - 1].end]
        return f, span_from_ids(list(range(c0, c1)), units, r['dur'])
    raise ValueError(edit)


def build(rows):
    """Per edit: applicable row indices, feature matrix, tIoU deltas."""
    out = {}
    for e in EDITS:
        idx, X, delta = [], [], []
        for i, r in enumerate(rows):
            f, sp = features(r, e)
            if f is None:
                continue
            idx.append(i)
            X.append(f)
            delta.append(tiou(r['gold'], sp) - r['tiou'])
        out[e] = (np.array(idx), np.array(X, float), np.array(delta))
    return out


def loco(rows, data):
    """Held-out predictions for every applicable question: P(helps) and predicted delta."""
    tids = np.array([r['tid'] for r in rows])
    preds = {}
    for e, (idx, X, delta) in data.items():
        p_help = np.full(len(idx), np.nan)
        p_delta = np.full(len(idx), np.nan)
        apply_log = np.zeros(len(idx), bool)
        for t in sorted(set(tids)):
            te = tids[idx] == t
            tr = ~te
            if te.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            y = delta[tr] > 1e-9
            if y.sum() >= 2 and (~y).sum() >= 2:
                lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y)
                p = lr.predict_proba(Xte)[:, 1]
                gain = delta[tr][y].mean()
                loss = -delta[tr][~y & (delta[tr] < -1e-9)].mean() if (delta[tr] < -1e-9).any() else 0.0
                apply_log[te] = p * gain > (1 - p) * loss
            else:
                p = np.zeros(te.sum())
            p_help[te] = p
            p_delta[te] = Ridge(alpha=1.0).fit(Xtr, delta[tr]).predict(Xte)
        preds[e] = (p_help, p_delta, apply_log)
    return preds


def policy_score(rows, data, chosen_delta):
    """Mean tIoU after applying, per question, the delta in chosen_delta (0 where nothing is applied)."""
    base = np.array([r['tiou'] for r in rows])
    return float((base + chosen_delta).mean())


def bootstrap_ci(rows, chosen_delta, n=5000):
    tids = np.array([r['tid'] for r in rows])
    uniq = sorted(set(tids))
    by = {t: chosen_delta[tids == t] for t in uniq}
    means = []
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        v = np.concatenate([by[t] for t in pick])
        means.append(v.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows = load()
    base = float(np.mean([r['tiou'] for r in rows]))
    data = build(rows)
    preds = loco(rows, data)
    lines = [f'served mean tIoU {base:.4f}, score {0.4*ACC+0.6*base:.4f}; {len(rows)} gold-yes questions, 39 conversations held out one at a time', '']
    hdr = '| edit | applicable | helps in-sample | policy | fires | wins | losses | LOCO mean tIoU | change | 95 % CI of change | held-out AUC |'
    lines += [hdr, '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    md = []
    best_delta_per_row = np.zeros(len(rows))
    best_pred_per_row = np.full(len(rows), -np.inf)
    for e in EDITS:
        idx, X, delta = data[e]
        p_help, p_delta, apply_log = preds[e]
        helps = delta > 1e-9
        auc = roc_auc_score(helps, p_help) if 0 < helps.sum() < len(helps) else float('nan')
        for name, mask in (('logistic, expected value > 0', apply_log), ('ridge, predicted change > 0', p_delta > 0),
                           ('perfect trigger (not a policy)', helps)):
            chosen = np.zeros(len(rows))
            chosen[idx[mask]] = delta[mask]
            m = policy_score(rows, data, chosen)
            lo, hi = bootstrap_ci(rows, chosen)
            wins, losses = int((delta[mask] > 1e-9).sum()), int((delta[mask] < -1e-9).sum())
            lines.append(f'| {e} | {len(idx)} | {int(helps.sum())} | {name} | {int(mask.sum())} | {wins} | {losses} | {m:.4f} | {m-base:+.4f} | {lo:+.4f} to {hi:+.4f} | {auc:.3f} |')
        # combined policy bookkeeping: the edit with the largest predicted change per question
        better = p_delta > best_pred_per_row[idx]
        best_pred_per_row[idx[better]] = p_delta[better]
        best_delta_per_row[idx[better]] = delta[better]
    chosen = np.where(best_pred_per_row > 0, best_delta_per_row, 0.0)
    m = policy_score(rows, data, chosen)
    lo, hi = bootstrap_ci(rows, chosen)
    fires = int((best_pred_per_row > 0).sum())
    lines.append(f'| any edit | {len(rows)} | | ridge, the edit with the largest predicted change | {fires} | {int((chosen > 1e-9).sum())} | {int((chosen < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | {lo:+.4f} to {hi:+.4f} | |')
    # the crude rule from entry 63 for reference, no fitting
    idx, X, delta = data['add-front']
    pron = np.array([rows[i]['pronoun'] for i in idx])
    chosen = np.zeros(len(rows)); chosen[idx[pron]] = delta[pron]
    m = policy_score(rows, data, chosen)
    lines.append(f'| add-front | {len(idx)} | | hand rule: citation opens on a pronoun | {int(pron.sum())} | {int((delta[pron] > 1e-9).sum())} | {int((delta[pron] < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | | |')
    text = '\n'.join(lines)
    print(text)
    OUT.write_text('# Leave-one-conversation-out test of edge edits to the 27B citation\n\n'
                   'Generated by `bench/context_loco.py` (see its docstring). Follows research/13-context-pattern.md and '
                   'findings-log entry 63. Every policy row is fitted on 38 conversations and applied to the 39th, '
                   'pooled over the 39 folds; "perfect trigger" is the in-sample upper bound and not a policy.\n\n'
                   + text + '\n', encoding='utf-8')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
