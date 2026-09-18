r"""Is a committee of extractive QA models worth it?

Reads the held-out predictions each member dumped during training and asks three
questions:

  1. How good is each member alone?
  2. How good is the best possible pick per question (oracle best-of-N)? That is
     the hard ceiling for *any* combination rule. If it barely beats the best
     single member, the members fail on the same questions and no voting scheme
     can rescue it.
  3. How much of that ceiling do real, implementable rules actually capture?

    .\run.cmd python scripts/committee_eval.py --runs runs/committee

Spans cannot be majority-voted directly, so three fusion rules are tried:
timeline bin-voting (each member votes per 50 ms, keep the longest run with a
majority), medoid (the member span most agreeing with the others), and the mean
of the edges.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACCURACY_WEIGHT, TIOU_WEIGHT = 0.4, 0.6
BIN = 0.05

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK_SOFT, GRID = '#0b0b0b', '#52514e', '#dedcd5'


def tiou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# span fusion rules
# --------------------------------------------------------------------------- #

def fuse_bins(spans, min_votes=None):
    """Each member votes per 50 ms bin; keep the longest run with a majority."""
    if not spans:
        return None
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    n = max(1, int(math.ceil((hi - lo) / BIN)))
    votes = [0] * n
    for s in spans:
        a = max(0, int((s[0] - lo) / BIN))
        b = min(n, int(math.ceil((s[1] - lo) / BIN)))
        for k in range(a, b):
            votes[k] += 1
    need = min_votes if min_votes else len(spans) // 2 + 1
    best = cur = None
    for k, v in enumerate(votes):
        if v >= need:
            cur = (k, k + 1) if cur is None else (cur[0], k + 1)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    if best is None:
        return None
    return lo + best[0] * BIN, lo + best[1] * BIN


def fuse_medoid(spans):
    """The member span that agrees most with the others."""
    if len(spans) == 1:
        return spans[0]
    return max(spans, key=lambda s: sum(tiou(s, o) for o in spans if o is not s))


def fuse_mean(spans):
    return float(np.mean([s[0] for s in spans])), float(np.mean([s[1] for s in spans]))


# --------------------------------------------------------------------------- #

def load(runs: Path):
    by_model = {}
    for f in sorted(runs.glob('*/preds.jsonl')):
        rows = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        if rows:
            by_model[f.parent.name] = rows
    return by_model


def shared_questions(by_model):
    """Reject ambiguous or incompatible held-out rows before comparing models."""
    reference, id_sets = {}, []
    required = {'id', 'transcript_id', 'fold', 'gold_yes', 'pred_yes',
                'time_start', 'time_end', 'evidence_start', 'evidence_end'}
    metadata = ('transcript_id', 'fold', 'gold_yes', 'evidence_start', 'evidence_end')
    for model, rows in by_model.items():
        seen = set()
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(f'{model}: prediction row missing {sorted(missing)}')
            qid = row['id']
            if qid in seen:
                raise ValueError(f'{model}: duplicate question ID {qid!r}')
            seen.add(qid)
            if qid in reference:
                other_model, other = reference[qid]
                mismatch = [key for key in metadata if row[key] != other[key]]
                if mismatch:
                    raise ValueError(f'{model} and {other_model}: incompatible '
                                     f'{qid!r} fields {mismatch}; use identical held-out folds and gold data')
            else:
                reference[qid] = (model, row)
        id_sets.append(seen)
    ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if not ids:
        raise ValueError('members have no held-out question IDs in common')
    return ids


def summarize(rows_by_id, ids, gold):
    """rows_by_id: id -> (pred_yes, span or None). Returns the case metrics."""
    correct = sum(int(rows_by_id[i][0] == gold[i]['gold_yes']) for i in ids)
    tious = []
    for i in ids:
        if not gold[i]['gold_yes']:
            continue
        yes, span = rows_by_id[i]
        t = 0.0
        if yes and span:
            t = tiou((gold[i]['evidence_start'], gold[i]['evidence_end']), span)
        tious.append(t)
    acc = correct / len(ids)
    mt = float(np.mean(tious)) if tious else 0.0
    return {'accuracy': acc, 'mean_tiou': mt,
            'score': ACCURACY_WEIGHT * acc + TIOU_WEIGHT * mt}


def line(name, m, width=26):
    return (f'  {name:<{width}} acc {m["accuracy"]:.3f}   tIoU {m["mean_tiou"]:.3f}   '
            f'score {m["score"]:.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, default=ROOT / 'runs' / 'committee')
    p.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    by_model = load(a.runs)
    if len(by_model) < 2:
        raise SystemExit(f'need at least 2 members with preds.jsonl under {a.runs}; found {list(by_model)}')

    models = sorted(by_model)
    # questions every member covered (identical folds, so this is all of them)
    try:
        ids = shared_questions(by_model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    gold = {r['id']: r for r in by_model[models[0]]}
    n_pos = sum(1 for i in ids if gold[i]['gold_yes'])

    print(f'{len(models)} members, {len(ids)} held-out questions '
          f'({n_pos} gold positives) over {len({gold[i]["transcript_id"] for i in ids})} conversations')
    for m in models:
        folds = sorted({r['fold'] for r in by_model[m]})
        print(f'    {m:<20} folds {folds}')
    print()

    # per-member predictions, indexed
    pred = {}
    for m in models:
        pred[m] = {}
        for r in by_model[m]:
            span = (r['time_start'], r['time_end']) if r['time_start'] is not None else None
            pred[m][r['id']] = (r['pred_yes'], span)

    print('Individual members')
    singles = {}
    for m in models:
        singles[m] = summarize(pred[m], ids, gold)
        print(line(m, singles[m]))
    best_single = max(singles, key=lambda m: singles[m]['score'])
    print(f'  -> best single: {best_single} at {singles[best_single]["score"]:.3f}')
    print()

    # ---- ceiling -----------------------------------------------------------
    oracle = {}
    for i in ids:
        yes_any = any(pred[m][i][0] == gold[i]['gold_yes'] for m in models)
        best_span, best_t = None, -1.0
        for m in models:
            yes, span = pred[m][i]
            if yes and span and gold[i]['gold_yes']:
                t = tiou((gold[i]['evidence_start'], gold[i]['evidence_end']), span)
                if t > best_t:
                    best_t, best_span = t, span
        oracle[i] = (gold[i]['gold_yes'] if yes_any else not gold[i]['gold_yes'], best_span)
    orc = summarize(oracle, ids, gold)
    print('Ceiling  (oracle: the best member per question, not implementable)')
    print(line('oracle best-of-N', orc))
    gap = orc['score'] - singles[best_single]['score']
    gap_t = orc['mean_tiou'] - singles[best_single]['mean_tiou']
    print(f'  -> headroom over best single: score +{gap:.3f}   tIoU +{gap_t:.3f}')
    print()

    # ---- real combination rules -------------------------------------------
    print('Implementable rules')
    combos = {}
    for label, fuse in (('vote + bin-fusion', fuse_bins),
                        ('vote + medoid', fuse_medoid),
                        ('vote + mean edges', fuse_mean)):
        out = {}
        for i in ids:
            yeses = [m for m in models if pred[m][i][0]]
            say_yes = len(yeses) > len(models) / 2
            span = None
            if say_yes:
                spans = [pred[m][i][1] for m in yeses if pred[m][i][1]]
                span = fuse(spans) if spans else None
            out[i] = (say_yes, span)
        combos[label] = summarize(out, ids, gold)
        print(line(label, combos[label]))
    best_combo = max(combos, key=lambda k: combos[k]['score'])
    print()

    # ---- do the members fail together? ------------------------------------
    print('Error overlap on gold positives (tIoU < 0.5 counts as a failure)')
    fails = {}
    for m in models:
        f = set()
        for i in ids:
            if not gold[i]['gold_yes']:
                continue
            yes, span = pred[m][i]
            t = tiou((gold[i]['evidence_start'], gold[i]['evidence_end']), span) if yes and span else 0.0
            if t < 0.5:
                f.add(i)
        fails[m] = f
        print(f'  {m:<26} {len(f):>3} / {n_pos} fail')
    shared = set.intersection(*fails.values())
    union = set.union(*fails.values())
    print(f'  {"shared by ALL members":<26} {len(shared):>3}'
          f'   ({len(shared) / max(1, len(union)):.0%} of the {len(union)} distinct failures)')
    for x, y in itertools.combinations(models, 2):
        inter = len(fails[x] & fails[y])
        un = len(fails[x] | fails[y])
        print(f'    {x} n {y}: {inter} shared, Jaccard {inter / max(1, un):.2f}')
    print()

    # ---- verdict -----------------------------------------------------------
    delta = combos[best_combo]['score'] - singles[best_single]['score']
    print('=' * 72)
    print(f'best single      {singles[best_single]["score"]:.3f}   ({best_single})')
    print(f'best committee   {combos[best_combo]["score"]:.3f}   ({best_combo})')
    print(f'oracle ceiling   {orc["score"]:.3f}')
    if delta > 0 and gap > 1e-9:
        print(f'committee gains  {delta:+.3f}  =  {delta / gap:.0%} of the {gap:.3f} available headroom')
    else:
        print(f'committee gains  {delta:+.3f}   (headroom was {gap:.3f}; no rule captured any of it)')
    print()
    if delta < 0.005:
        print('VERDICT: not worth it. The members fail on the same questions;')
        print('three models cost 3x inference for nothing. Spend the effort on the')
        print('span representation (clause-level units) instead.')
    elif delta < 0.02:
        print('VERDICT: marginal. Real but small; weigh it against 3x inference')
        print('inside the 60 s per-conversation budget.')
    else:
        print('VERDICT: worth it. The members make genuinely different errors and')
        print('a simple rule captures a useful part of the gap.')
    print('=' * 72)

    out = {'members': singles, 'oracle': orc, 'combos': combos,
           'n_questions': len(ids), 'n_positives': n_pos,
           'failures': {m: sorted(f) for m, f in fails.items()},
           'shared_failures': sorted(shared)}
    (a.runs / 'committee.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\nwritten -> {a.runs / "committee.json"}')

    if a.plot:
        plot(singles, combos, orc, a.runs)


def plot(singles, combos, orc, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # One global rank by score, color kept by category (member / rule / ceiling)
    # so a top-to-bottom read is never contradicted by a color grouping.
    entries = [(m, v['score'], PALETTE[0]) for m, v in singles.items()]
    entries += [(k, v['score'], PALETTE[2]) for k, v in combos.items()]
    entries += [('oracle best-of-N', orc['score'], INK_SOFT)]
    entries.sort(key=lambda e: e[1])
    names = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    colors = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(9, 0.52 * len(names) + 1.8), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.annotate(f'{v:.3f}', (v, yi), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel('case score  (0.4×accuracy + 0.6×tIoU)', color=INK_SOFT, fontsize=9)
    ax.set_title('Individual members (blue), committee rules (green), ceiling (grey)',
                 loc='left', color=INK, fontsize=11, pad=10)
    ax.grid(True, axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    fig.tight_layout()
    fig.savefig(out_dir / 'committee.png', dpi=150, facecolor='#fcfcfb')
    print(f'plot    -> {out_dir / "committee.png"}')


if __name__ == '__main__':
    raise SystemExit(main())
