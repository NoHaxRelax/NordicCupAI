"""ONE check of the citation edge edits on the VALIDATION conversations. Measurement only.

The validation labels (bench/mine/agent_answers.md binaries, bench/mine/span_state.json recovered
spans) are a held-out yardstick and are never used to tune the pipeline. This script applies the
policies of bench/context_loco.py, fitted on ALL 39 training conversations, to the served 27B's stored
validation answers (bench/results/served/pod3/answers.jsonl, the last block of 19 = the 0.8084 run
with units-fewshot-both) and scores them against the hand labels the way bench/mine/val_probe.py
does. Every policy is scored in the same pass; nothing is selected afterwards.

The served span of each question is mapped back to the run of clause units it was built from (an
exact match of model.span_from_ids over every run of up to 8 units), so the edits operate on units,
exactly as they did on the training set.

    python bench/context_val.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'mine'))
import context_loco as cl  # noqa: E402
import context_pattern as cp  # noqa: E402
import answers_md  # noqa: E402
from context_pattern import content_words, span_from_ids, tiou  # noqa: E402

DUMP = CASE / 'request_dump'
SERVED = HERE / 'results' / 'served' / 'pod3' / 'answers.jsonl'
VAL = {3, 7, 8, 11, 14, 16, 24, 26, 30, 31, 40, 44, 45, 46, 51, 60, 62, 73, 80}
MAX_RUN = 8


def served_blocks():
    """The validation blocks of the served answers file, in the order they were served."""
    recs = [json.loads(l) for l in SERVED.read_text(encoding='utf-8').splitlines() if l.strip()]
    val = [r for r in recs if int(r['file'].split('_')[-1].split('.')[0]) in VAL]
    assert len(val) % 19 == 0, len(val)
    return [val[i:i + 19] for i in range(0, len(val), 19)]


def load_units(stem):
    d = json.loads((DUMP / 'transcripts' / f'{stem}.large-v3-turbo.json').read_text(encoding='utf-8'))
    words = []
    for s in d['segments']:
        for w in s.get('words', []):
            words.append(cp.model.Word(w['w'], float(w['start']), float(w['end'])))
        if s.get('words'):
            words[-1].w += '\x00'
    return cp.make_units(words, 'clause-and'), float(d['duration'])


def find_run(span, units, duration):
    hits = []
    for i in range(len(units)):
        for j in range(i, min(len(units), i + MAX_RUN)):
            sp = span_from_ids(list(range(i, j + 1)), units, duration)
            if [round(x, 2) for x in sp] == [round(x, 2) for x in span]:
                hits.append((i, j))
    return hits


def val_rows(block):
    """Rows in the shape context_loco.features expects, for the gold-yes questions with a hand span."""
    state = json.loads((HERE / 'mine' / 'span_state.json').read_text(encoding='utf-8'))
    hand = answers_md.parse()
    rows, n, correct, unmatched, no_gold = [], 0, 0, 0, 0
    for rec in block:
        stem = rec['file'].replace('.mp3', '')
        units, duration = load_units(stem)
        labels = {r['q']: r for r in hand.get(stem, [])}
        for i, (q, ans, span) in enumerate(zip(rec['questions'], rec['answers'], rec['spans']), 1):
            if i not in labels:
                continue
            n += 1
            correct += (bool(ans) == bool(labels[i]['answer']))
            if not labels[i]['answer']:
                continue
            st = state.get(f'{stem}:{i}')
            if not (st and st.get('stage') == 'done'):
                no_gold += 1
                continue
            gold = [st['g'], st['h']]
            if not span:
                rows.append({'id': f'{stem}:{i}', 'tid': stem, 'q': q, 'units': units, 'dur': duration, 'gold': gold,
                             'chosen': None, 'tiou': 0.0})
                continue
            hits = find_run(span, units, duration)
            if len(hits) != 1:
                unmatched += 1
                rows.append({'id': f'{stem}:{i}', 'tid': stem, 'q': q, 'units': units, 'dur': duration, 'gold': gold,
                             'chosen': None, 'tiou': tiou(gold, span)})
                continue
            c = hits[0]
            text = cp.utext(units, *c)
            first = re.sub(r"[^a-z']", '', text.split()[0].lower()) if text.split() else ''
            rows.append({'id': f'{stem}:{i}', 'tid': stem, 'q': q, 'units': units, 'dur': duration, 'gold': gold,
                         'chosen': c, 'chosen_text': text, 'first': first, 'tiou': tiou(gold, span)})
    return rows, n, correct, unmatched, no_gold


def main():
    blocks = served_blocks()
    print(f'{len(blocks)} validation blocks in {SERVED.name}; scoring each against the hand labels to identify the runs')
    for k, b in enumerate(blocks):
        rows, n, correct, unmatched, no_gold = val_rows(b)
        acc = correct / n
        t = statistics.mean(r['tiou'] for r in rows)
        print(f'  block {k}: questions {n} accuracy {acc:.4f} positives with a hand span {len(rows)} mean tIoU {t:.4f} '
              f'score {0.4*acc+0.6*t:.4f}  (spans not matching one unit run: {unmatched}, gold-yes without a span label: {no_gold})')
    block = blocks[-1]                                  # the last served validation run: units-fewshot-both, portal 0.8084
    rows, n, correct, unmatched, no_gold = val_rows(block)
    acc = correct / n
    base = statistics.mean(r['tiou'] for r in rows)
    print(f'\nlast block = the served configuration: accuracy {acc:.4f}, mean tIoU {base:.4f} over {len(rows)} positives, '
          f'score {0.4*acc+0.6*base:.4f} under the hand labels (portal said 0.8084)')

    train = cl.load()
    tdata = cl.build(train)
    lines = ['| edit | policy (fitted on all 39 training conversations) | applicable | fires | wins | losses | validation mean tIoU | change | validation score |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines.append(f'| none | served | | | | | {base:.4f} | | {0.4*acc+0.6*base:.4f} |')
    best_pred = np.full(len(rows), -np.inf)
    best_delta = np.zeros(len(rows))
    for e in cl.EDITS:
        idx, X, delta = tdata[e]
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        y = delta > 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
        gain = delta[y].mean()
        loss = -delta[delta < -1e-9].mean()
        rg = Ridge(alpha=1.0).fit(Xs, delta)
        vidx, vX, vdelta = [], [], []
        for i, r in enumerate(rows):
            if r['chosen'] is None:
                continue
            f, sp = cl.features(r, e)
            if f is None:
                continue
            vidx.append(i); vX.append(f); vdelta.append(tiou(r['gold'], sp) - r['tiou'])
        vidx, vX, vdelta = np.array(vidx), np.array(vX, float), np.array(vdelta)
        vXs = sc.transform(vX)
        p = lr.predict_proba(vXs)[:, 1]
        pd = rg.predict(vXs)
        for name, mask in (('logistic, expected value > 0', p * gain > (1 - p) * loss), ('ridge, predicted change > 0', pd > 0)):
            chosen = np.zeros(len(rows)); chosen[vidx[mask]] = vdelta[mask]
            m = base + chosen.mean()
            lines.append(f'| {e} | {name} | {len(vidx)} | {int(mask.sum())} | {int((vdelta[mask] > 1e-9).sum())} | '
                         f'{int((vdelta[mask] < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | {0.4*acc+0.6*m:.4f} |')
        better = pd > best_pred[vidx]
        best_pred[vidx[better]] = pd[better]
        best_delta[vidx[better]] = vdelta[better]
        if e == 'add-front':
            pron = np.array([rows[i]['first'] in cp.PRONOUNS for i in vidx])
            chosen = np.zeros(len(rows)); chosen[vidx[pron]] = vdelta[pron]
            m = base + chosen.mean()
            lines.append(f'| {e} | hand rule: citation opens on a pronoun | {len(vidx)} | {int(pron.sum())} | {int((vdelta[pron] > 1e-9).sum())} | '
                         f'{int((vdelta[pron] < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | {0.4*acc+0.6*m:.4f} |')
        if e == 'drop-front':
            qf = np.array([rows[i]['units'][rows[i]['chosen'][0]].text.strip().endswith('?') for i in vidx])
            chosen = np.zeros(len(rows)); chosen[vidx[qf]] = vdelta[qf]
            m = base + chosen.mean()
            lines.append(f'| {e} | hand rule: first unit ends with a question mark | {len(vidx)} | {int(qf.sum())} | {int((vdelta[qf] > 1e-9).sum())} | '
                         f'{int((vdelta[qf] < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | {0.4*acc+0.6*m:.4f} |')
        perfect = np.zeros(len(rows)); perfect[vidx[vdelta > 1e-9]] = vdelta[vdelta > 1e-9]
        m = base + perfect.mean()
        lines.append(f'| {e} | perfect trigger (not a policy) | {len(vidx)} | {int((vdelta > 1e-9).sum())} | {int((vdelta > 1e-9).sum())} | 0 | {m:.4f} | {m-base:+.4f} | {0.4*acc+0.6*m:.4f} |')
    chosen = np.where(best_pred > 0, best_delta, 0.0)
    m = base + chosen.mean()
    lines.append(f'| any edit | ridge, the edit with the largest predicted change | {len(rows)} | {int((best_pred > 0).sum())} | '
                 f'{int((chosen > 1e-9).sum())} | {int((chosen < -1e-9).sum())} | {m:.4f} | {m-base:+.4f} | {0.4*acc+0.6*m:.4f} |')
    text = '\n'.join(lines)
    print()
    print(text)
    out = CASE / 'research' / '14-context-loco.md'
    doc = out.read_text(encoding='utf-8')
    marker = '\n## Validation check'
    doc = doc.split(marker)[0].rstrip() + '\n' + marker + (
        f'\n\nOne pass over the 19 validation conversations (`bench/context_val.py`): the served 27B answers of the last '
        f'validation run (portal 0.8084), scored against the hand labels ({n} questions, accuracy {acc:.4f}, {len(rows)} gold-yes '
        f'questions with a recovered span; {unmatched} served spans did not map to exactly one unit run and were left as served), '
        f'with every policy fitted on all 39 training conversations. Nothing was selected after seeing these numbers.\n\n' + text + '\n')
    out.write_text(doc, encoding='utf-8')
    print(f'appended the validation section to {out}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
