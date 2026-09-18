"""Parse bench/mine/agent_answers.md (the hand answers with evidence times) into

  bench/mine/agent_labels/<stem>.json     same shape as the /label page files
  bench/mine/current_answers.json         the probe endpoint's table

    python bench/mine/answers_md.py                      # rebuild both, binaries only (null spans)
    python bench/mine/answers_md.py --spans              # table carries the evidence spans too
    python bench/mine/answers_md.py --flip 3,7           # invert every answer of samples 3 and 7 (count query)
    python bench/mine/answers_md.py --flip 8:3,8:6       # invert single questions (sample 8, q3 and q6)
    python bench/mine/answers_md.py --diff               # only print where the served model (run F) disagrees

The md is the source of truth; edit it and re-run. A row is
| q | yes/no | segs | start | end | note |  under a "## conversation_sample_N" heading.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
MD = HERE / 'agent_answers.md'
LABELS = HERE / 'agent_labels'
TABLE = HERE / 'current_answers.json'
DUMP = CASE / 'request_dump'


def parse(md: Path = MD) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    stem = None
    for line in md.read_text(encoding='utf-8').splitlines():
        m = re.match(r'^##\s+(conversation_sample_\d+)\s*$', line)
        if m:
            stem = m.group(1); out[stem] = []; continue
        if stem is None or not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        q = int(cells[0]); ans = cells[1].lower()
        if ans not in ('yes', 'no'):
            raise SystemExit(f'{stem} q{q}: answer must be yes or no, got {cells[1]!r}')
        start = float(cells[3]) if cells[3] else None
        end = float(cells[4]) if cells[4] else None
        if ans == 'yes' and (start is None or end is None):
            raise SystemExit(f'{stem} q{q}: yes without a span')
        if start is not None and end is not None and end <= start:
            raise SystemExit(f'{stem} q{q}: end <= start')
        out[stem].append({'q': q, 'answer': ans == 'yes', 'segs': cells[2], 'start': start, 'end': end,
                          'note': ' | '.join(cells[5:]), 'check': 'CHECK' in line})
    for stem, rows in out.items():
        if [r['q'] for r in rows] != list(range(1, 11)):
            raise SystemExit(f'{stem}: expected q1..q10 in order, got {[r["q"] for r in rows]}')
    return out


def questions() -> dict[str, dict]:
    d = {}
    for line in (DUMP / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line); d[Path(r['file']).stem] = r
    return d


def flips(spec: str | None) -> tuple[set[str], set[tuple[str, int]]]:
    whole, single = set(), set()
    for tok in (spec or '').split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            s, q = tok.split(':'); single.add((f'conversation_sample_{int(s)}', int(q)))
        else:
            whole.add(f'conversation_sample_{int(tok)}')
    return whole, single


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--spans', action='store_true', help='put the evidence spans in the table (default: null spans)')
    ap.add_argument('--flip', default='', help='comma list: N (whole sample) or N:q (one question) to invert')
    ap.add_argument('--diff', action='store_true', help='print disagreements with the served model and exit')
    ap.add_argument('--gold', action='store_true', help='with --spans: use the spans recovered by span_probe.py (span_state.json) where available')
    ap.add_argument('--out', default=str(TABLE))
    a = ap.parse_args()
    rows = parse()
    if a.gold:
        state = json.loads((HERE / 'span_state.json').read_text(encoding='utf-8'))
        n_gold = 0
        for stem, items in rows.items():
            for r in items:
                q = state.get(f"{stem}:{r['q']}")
                if q and q.get('stage') == 'done':
                    r['start'], r['end'] = q['g'], q['h']; n_gold += 1
        print(f'{n_gold} spans taken from span_state.json')
    qs = questions()
    missing = set(qs) - set(rows)
    if missing:
        raise SystemExit(f'no answers for {sorted(missing)}')
    whole, single = flips(a.flip)
    n_yes = n_dis = 0
    table = {}
    LABELS.mkdir(exist_ok=True)
    for stem in sorted(rows, key=lambda s: int(re.sub(r'\D', '', s))):
        d = qs[stem]
        items, row = [], []
        for r in rows[stem]:
            i = r['q'] - 1
            ans = r['answer']
            n_yes += ans
            if d['answers'][i] != ans:
                n_dis += 1
                if a.diff:
                    print(f"{stem} q{r['q']}: model {'yes' if d['answers'][i] else 'no'}, mine {'yes' if ans else 'no'}  {d['questions'][i]}")
            items.append({'question': d['questions'][i], 'answer': ans, 'start': r['start'] if ans else None,
                          'end': r['end'] if ans else None, 'segs': r['segs'], 'note': r['note'], 'check': r['check']})
            if stem in whole or (stem, r['q']) in single:
                ans = not ans
            with_span = a.spans and ans and r['start'] is not None
            row.append({'answer': ans, 'start': r['start'] if with_span else None, 'end': r['end'] if with_span else None})
        table[d['file']] = row
        (LABELS / f'{stem}.json').write_text(json.dumps({'stem': stem, 'items': items}, indent=1, ensure_ascii=False), encoding='utf-8')
    if a.diff:
        print(f'{n_dis} disagreements with the served model; {n_yes} yes of {10 * len(rows)}')
        return 0
    Path(a.out).write_text(json.dumps(table, indent=1), encoding='utf-8')
    fl = f', flipped {sorted(whole)} {sorted(single)}' if (whole or single) else ''
    print(f'{len(table)} files, {n_yes} yes of {10 * len(rows)}, {n_dis} disagree with the model; spans {"on" if a.spans else "off"}{fl}; wrote {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
