"""Yes-threshold sweep on a bench result that carries p_yes per question.

Scoring is the live rule as measured on the service (research/07 entries 15
and 20): accuracy over all questions, tIoU over annotated yes questions with a
span counted only when we answered yes. For a threshold t we answer yes when
p_yes >= t and keep the run's span for that question.

Reports the score per threshold on the whole set and leave-one-conversation-out
(the threshold chosen on the other 38 conversations, applied to the held-out
one), so the recommended threshold is out of sample.

    python bench/llm/threshold_sweep.py bench/results/llm/qwen3-4b.units-pyes.large-v3-turbo.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

CASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE))
from span_ceiling import tiou  # noqa: E402

THRESHOLDS = [x / 100 for x in range(5, 96, 5)]


def score(recs, t):
    correct = 0; tious = []
    for r in recs:
        p = r.get('p_yes')
        yes = (p >= t) if p is not None else bool(r.get('answer'))
        correct += int(yes) == r['label']
        if r['label'] == 1 and r.get('gold'):
            g = tuple(r['gold']); s = r.get('span')
            tious.append(tiou(g, tuple(s)) if (yes and s) else 0.0)
    acc = correct / len(recs)
    mt = st.mean(tious) if tious else 0.0
    return 0.4 * acc + 0.6 * mt, acc, mt, sum(1 for r in recs if (r.get('p_yes') if r.get('p_yes') is not None else r.get('answer')) and ((r.get('p_yes') or 0) >= t if r.get('p_yes') is not None else True)) / len(recs)


def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text(encoding='utf-8'))
    recs = [r for r in (d.get('records') or d.get('questions') or []) if r.get('answer') is not None]
    have = sum(1 for r in recs if r.get('p_yes') is not None)
    print(f'{len(recs)} questions, {have} with p_yes')
    print(f'{"t":>5} {"score":>7} {"acc":>6} {"tIoU":>6} {"yes-rate":>8}')
    for t in THRESHOLDS:
        s, a, m, yr = score(recs, t)
        print(f'{t:5.2f} {s:7.3f} {a:6.3f} {m:6.3f} {yr:8.3f}')
    base = score(recs, 0.5)
    print(f'\nas answered by the model (t=0.5 equivalent): score {base[0]:.3f}')
    # leave-one-conversation-out choice of the threshold
    by_conv = defaultdict(list)
    for r in recs:
        by_conv[r['transcript_id']].append(r)
    loco_scores, chosen = [], []
    for held, rows in by_conv.items():
        train = [r for c, rs in by_conv.items() if c != held for r in rs]
        best_t = max(THRESHOLDS, key=lambda t: score(train, t)[0])
        chosen.append(best_t)
        loco_scores.append((score(rows, best_t)[0], len(rows)))
    n = sum(k for _, k in loco_scores)
    loco = sum(s * k for s, k in loco_scores) / n
    print(f'LOCO: chosen thresholds median {st.median(chosen):.2f} (min {min(chosen):.2f}, max {max(chosen):.2f}); out-of-sample score {loco:.3f} vs {base[0]:.3f} as answered')


if __name__ == '__main__':
    main()
