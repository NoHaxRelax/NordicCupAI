"""Held-out diagnosis of the served pipeline on the validation set.

Uses request_dump/answers.jsonl (what the endpoint answered in the last real validation
run: booleans and spans) against the hand binaries (agent_answers.md) and the recovered
gold spans (span_state.json). Prints the accuracy, the mean tIoU under the live scorer's
policy (a positive answered no scores zero), the reconstructed score, and where the span
losses sit: missed positive, wrong place (no overlap), too short / spill (overlap but
IoU < 0.9), exact (IoU >= 0.9). Diagnosis only: the pipeline is never tuned on this set.

    python bench/mine/val_diag.py
    python bench/mine/val_diag.py --list        # one line per positive
    python bench/mine/val_diag.py --dump DIR   # answers.jsonl written by bench/mine/send_dump.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
import answers_md  # noqa: E402


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dump', default=str(CASE / 'request_dump'), help='dump dir with answers.jsonl (default request_dump/)')
    a = ap.parse_args()
    rows = answers_md.parse()
    state = json.loads((HERE / 'span_state.json').read_text(encoding='utf-8')) if (HERE / 'span_state.json').exists() else {}
    served = {}
    for line in (Path(a.dump) / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        d = json.loads(line); served[Path(d['file']).stem] = d
    n = correct = 0
    ious, cats, starts, ends = [], {'missed': 0, 'wrong place': 0, 'edges': 0, 'exact': 0, 'no gold yet': 0}, [], []
    lines = []
    for stem in sorted(rows, key=lambda s: int(re.sub(r'\D', '', s))):
        d = served[stem]
        for r in rows[stem]:
            i = r['q'] - 1
            n += 1
            correct += (bool(d['answers'][i]) == r['answer'])
            if not r['answer']:
                continue
            q = state.get(f"{stem}:{r['q']}")
            gold = [q['g'], q['h']] if q and q.get('stage') == 'done' else None
            sp = d['spans'][i] if d.get('spans') else None
            if gold is None:
                cats['no gold yet'] += 1; continue
            if not d['answers'][i]:
                cats['missed'] += 1; v = 0.0
            else:
                v = iou(sp, gold)
                cats['exact' if v >= 0.9 else 'edges' if v > 0 else 'wrong place'] += 1
            ious.append(v)
            if sp is not None and d['answers'][i]:
                starts.append(sp[0] - gold[0]); ends.append(sp[1] - gold[1])
            lines.append(f"{stem.replace('conversation_sample_', 's'):>4} q{r['q']:<2} {'yes' if d['answers'][i] else 'NO '} "
                         f"gold {gold[0]:7.2f}-{gold[1]:7.2f} ({gold[1]-gold[0]:4.1f}s)  served {sp[0] if sp else '   -  ':>7}-{sp[1] if sp else '   -  ':<7} iou {v:.2f}  "
                         f"seed {r['start']:7.2f}-{r['end']:7.2f} iou {iou([r['start'], r['end']], gold):.2f}")
    if a.list:
        print('\n'.join(lines))
    acc = correct / n
    m = statistics.mean(ious) if ious else 0.0
    print(f'binaries: {correct}/{n} = {acc:.3f}   positives with gold: {len(ious)} of 95   mean tIoU (nulls on no): {m:.3f}   '
          f'reconstructed score: {0.4 * acc + 0.6 * m:.4f}')
    print('span outcome on positives:', ', '.join(f'{k} {v}' for k, v in cats.items()))
    if starts:
        print(f'served minus gold, answered-yes positives: start median {statistics.median(starts):+.2f} s, end median {statistics.median(ends):+.2f} s')
    seed_ious = [iou([r['start'], r['end']], [state[f"{s}:{r['q']}"]['g'], state[f"{s}:{r['q']}"]['h']])
                 for s in rows for r in rows[s] if r['answer'] and state.get(f"{s}:{r['q']}", {}).get('stage') == 'done']
    if seed_ious:
        print(f'hand seeds (turbo segments) vs gold: mean IoU {statistics.mean(seed_ious):.3f}, exact (>=0.9) {sum(v >= 0.9 for v in seed_ious)}, no overlap {sum(v == 0 for v in seed_ious)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
