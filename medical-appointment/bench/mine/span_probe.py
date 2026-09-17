"""Recover the annotated evidence spans of the validation set, one question per
validation run, seeded by the hand spans in agent_answers.md.

Arithmetic. With every binary right (bench/mine/agent_answers.md, verified
190/190) and a span on exactly one positive question i, the score is
    S = 0.4 + 0.6/95 * IoU_i        (95 gold positives on the validation set)
so one run measures IoU_i = (S - 0.4) * 95 / 0.6 to the portal's precision.

Per question, with gold [g, h], length L = h - g, and seed [s, e]:
  stage len     guess [A, B] = [s - W, e + W], wide enough to contain the gold:
                IoU = L / (B - A)  ->  L
  stage start   guess [A, m] with A <= g and m inside the gold:
                IoU = (m - g) / (m - A)  ->  g = m - IoU * (m - A), h = g + L.
                IoU = 0 means m <= g (move m right); IoU * (m - A) == L means
                m >= h (move m left). Bisect m until it lands inside.
  stage verify  guess [g, h]: IoU must be ~1 (>= 0.97 allows the 20 ms grid).
                Otherwise the containment assumption failed: widen W and redo.
Typically three runs per question. State persists in bench/mine/span_state.json
so the loop can stop and resume; every run also appends to count_runs.jsonl.

    python bench/mine/span_probe.py --url https://xxx.trycloudflare.com --only 44:2,51:6   # pilot
    python bench/mine/span_probe.py --url https://xxx.trycloudflare.com                    # everything
    python bench/mine/span_probe.py --report                                               # no runs, print the state

Only the validation queue is used (portal_status.queue_validation).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
sys.path.insert(0, str(CASE / 'bench'))
sys.path.insert(0, str(HERE))
import portal_status  # noqa: E402
import answers_md  # noqa: E402

STATE = HERE / 'span_state.json'
TABLE = HERE / 'current_answers.json'
LOG = HERE / 'count_runs.jsonl'
N_POS = 95
BASE = 0.4
W0 = 1.5          # initial widening in seconds for the len stage
OK_IOU = 0.97
GRID = 0.02


def durations() -> dict[str, float]:
    out = {}
    for f in (CASE / 'request_dump' / 'transcripts').glob('*.json'):
        d = json.loads(f.read_text(encoding='utf-8'))
        out[f.name.split('.')[0]] = float(d['segments'][-1]['end']) if d.get('segments') else 1e9
    return out


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(st, indent=1), encoding='utf-8')
    tmp.replace(STATE)


def init_state(st: dict) -> dict:
    rows = answers_md.parse()
    for stem, items in rows.items():
        for r in items:
            if not r['answer']:
                continue
            k = f"{stem}:{r['q']}"
            st.setdefault(k, {'seed': [r['start'], r['end']], 'W': W0, 'runs': [], 'stage': 'len',
                              'L': None, 'g': None, 'h': None, 'm_lo': None, 'm_hi': None})
    return st


def base_table() -> dict:
    subprocess.run([sys.executable, str(HERE / 'answers_md.py')], check=True, capture_output=True, text=True)
    return json.loads(TABLE.read_text(encoding='utf-8'))


def one_run(url: str, table: dict, key: str, guess: list[float], poll: float) -> float:
    TABLE.write_text(json.dumps(table, indent=1), encoding='utf-8')
    before = {v.get('submitted_at') for v in portal_status.fetch().get('validations', [])}
    for attempt in range(3):
        q = portal_status.queue_validation(url.rstrip('/') + '/predict')
        t0 = time.time()
        while True:
            time.sleep(poll)
            vals = [v for v in portal_status.fetch().get('validations', []) if v.get('submitted_at') not in before]
            newest = max(vals, key=lambda v: v.get('submitted_at') or '', default=None)
            if newest and newest.get('finished_at'):
                break
            if time.time() - t0 > 600:
                raise RuntimeError('validation did not finish in 10 minutes')
        score = newest.get('score')
        errs = newest.get('errors') or []
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'t': time.time(), 'label': f'span {key} {guess}', 'score': score, 'errors': errs,
                                'elapsed': round(time.time() - t0, 1)}) + '\n')
        if isinstance(score, (int, float)) and not errs:
            return (score - BASE) * N_POS / 0.6
        before.add(newest.get('submitted_at'))
        print(f'    run failed ({errs[:1]}), retrying', flush=True)
        time.sleep(10)
    raise RuntimeError('three failed runs')


def step(key: str, q: dict, dur: float) -> tuple[list[float], str]:
    """Return the next guess and its stage for one question."""
    s, e = q['seed']
    if q['stage'] == 'len':
        A, B = max(0.0, s - q['W']), min(dur, e + q['W'])
        return [A, B], 'len'
    if q['stage'] == 'start':
        A = max(0.0, s - q['W'])
        lo = q['m_lo'] if q['m_lo'] is not None else A
        hi = q['m_hi'] if q['m_hi'] is not None else min(dur, e + q['W'])
        m = (s + e) / 2 if not q.get('m_tried') else (lo + hi) / 2
        return [A, round(m, 3)], 'start'
    if q['stage'] == 'verify':
        return [q['g'], q['h']], 'verify'
    raise ValueError(q['stage'])


def absorb(q: dict, guess: list[float], stage: str, iou: float, dur: float) -> None:
    q['runs'].append({'stage': stage, 'guess': guess, 'iou': round(iou, 5)})
    A, B = guess
    if stage == 'len':
        q['L'] = iou * (B - A)
        if q['L'] < GRID:
            q['stage'] = 'stuck'; q['why'] = 'no overlap with the widened seed'; return
        q['stage'] = 'start'; q['m_tried'] = False; q['m_lo'] = None; q['m_hi'] = None
    elif stage == 'start':
        m = B
        q['m_tried'] = True
        if iou <= 1e-9:                      # m left of the gold
            q['m_lo'] = m
        elif abs(iou * (m - A) - q['L']) < 1.5 * GRID:   # m right of the gold: whole gold inside [A, m]
            q['m_hi'] = m
        else:
            g = m - iou * (m - A)
            q['g'] = round(g, 3); q['h'] = round(g + q['L'], 3); q['stage'] = 'verify'; return
        if q['m_lo'] is not None and q['m_hi'] is not None and q['m_hi'] - q['m_lo'] < GRID:
            q['stage'] = 'stuck'; q['why'] = 'bisection collapsed'
        elif len([r for r in q['runs'] if r['stage'] == 'start']) > 6:
            q['stage'] = 'stuck'; q['why'] = 'too many start probes'
    elif stage == 'verify':
        if iou >= OK_IOU:
            q['stage'] = 'done'
        else:
            q['W'] = q['W'] * 2
            if q['W'] > 12:
                q['stage'] = 'stuck'; q['why'] = f'verify iou {iou:.3f} even with W {q["W"]/2}'
            else:
                q['stage'] = 'len'; q['note'] = f'verify iou {iou:.3f}, widening to W {q["W"]}'


def report(st: dict) -> None:
    n = {'done': 0, 'stuck': 0, 'todo': 0}
    runs = 0
    for k, q in sorted(st.items(), key=lambda kv: (int(re.sub(r'\D', '', kv[0].split(':')[0])), int(kv[0].split(':')[1]))):
        runs += len(q['runs'])
        n['done' if q['stage'] == 'done' else 'stuck' if q['stage'] == 'stuck' else 'todo'] += 1
        if q['stage'] in ('done', 'stuck') or q['runs']:
            print(f"{k:>24} {q['stage']:>7} seed {q['seed'][0]:7.2f}-{q['seed'][1]:7.2f}  gold {q['g'] if q['g'] is not None else '-':>7} - {q['h'] if q['h'] is not None else '-':>7}  L {q['L'] if q['L'] is not None else '-'}  runs {len(q['runs'])}  {q.get('why', '')}")
    print(f"{n['done']} done, {n['stuck']} stuck, {n['todo']} to do; {runs} runs so far")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url')
    ap.add_argument('--only', default='', help='comma list of stem:q, e.g. 44:2,51:6')
    ap.add_argument('--max-runs', type=int, default=10_000)
    ap.add_argument('--poll', type=float, default=4.0)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    st = init_state(load_state())
    save_state(st)
    if a.report:
        report(st); return 0
    if not a.url:
        raise SystemExit('--url required')
    only = {f'conversation_sample_{t.split(":")[0]}:{t.split(":")[1]}' for t in a.only.split(',') if t.strip()}
    dur = durations()
    base = base_table()
    keys = [k for k in st if (not only or k in only) and st[k]['stage'] not in ('done', 'stuck')]
    print(f'{len(keys)} questions to solve', flush=True)
    n_runs = 0
    for k in keys:
        q = st[k]
        stem, qi = k.split(':'); qi = int(qi)
        fname = f'{stem}.mp3'
        while q['stage'] not in ('done', 'stuck') and n_runs < a.max_runs:
            guess, stage = step(k, q, dur.get(stem, 1e9))
            table = json.loads(json.dumps(base))
            table[fname][qi - 1] = {'answer': True, 'start': guess[0], 'end': guess[1]}
            iou = one_run(a.url, table, k, guess, a.poll)
            n_runs += 1
            absorb(q, guess, stage, iou, dur.get(stem, 1e9))
            print(f"{k:>24} {stage:>6} [{guess[0]:7.2f}, {guess[1]:7.2f}] iou {iou:.4f} -> {q['stage']}"
                  f"{'  gold ' + str(q['g']) + '-' + str(q['h']) if q['g'] is not None else ''}", flush=True)
            save_state(st)
        if n_runs >= a.max_runs:
            print('max runs reached'); break
    TABLE.write_text(json.dumps(base, indent=1), encoding='utf-8')   # leave the table clean
    report(st)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
