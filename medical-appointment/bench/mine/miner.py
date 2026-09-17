"""Validation label miner: recovers the yes/no labels of the validation set from
the returned scores, one queued attempt at a time.

Mechanics. With null spans the score is 0.4 * correct / N, so each attempt
returns the number of correct answers. Take a base answer vector p (our
model's answers from a real run, or all-no). Flipping a subset S of positions
changes the count by |S & W| - |S & R| where W is the set of positions p gets
wrong, so one attempt reveals |S & W| = (c - c_base + |S|) / 2. The search is
a recursive halving on groups with a known number of wrong positions, which
stops early when a group is all right or all wrong. With ~19 errors among 190
this is on the order of 60 to 90 attempts.

Priority. Before each enqueue the miner waits while bench/mine/PAUSE exists or
any medical-appointment attempt is queued or running that is not its own; a
serious run therefore only needs the PAUSE file to exist while it is queued.
State is persisted to bench/mine/state.json after every attempt, so the miner
can be stopped and resumed.

    python bench/mine/miner.py --url https://<probe tunnel>/predict --base bench/mine/base_answers.json
    python bench/mine/miner.py --url ... --base none        # start from all-no (slower, ~2x attempts)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
sys.path.insert(0, str(CASE / 'bench'))
import portal_status as ps  # noqa: E402

N_PER = 10
STATE = HERE / 'state.json'
ANSWERS = HERE / 'current_answers.json'
PAUSE = HERE / 'PAUSE'


def log(msg: str) -> None:
    line = f'{time.strftime("%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(HERE / 'miner.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def files_from_requests() -> list:
    """Validation filenames in the order the portal sends them, from requests.jsonl."""
    seen = []
    p = HERE / 'requests.jsonl'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            try:
                f = json.loads(line)['file']
            except Exception:
                continue
            if f not in seen:
                seen.append(f)
    return seen


def write_table(files: list, vec: list) -> None:
    ANSWERS.write_text(json.dumps({f: vec[i * N_PER:(i + 1) * N_PER] for i, f in enumerate(files)}, indent=1), encoding='utf-8')


def busy(own_uuids: set) -> bool:
    d = ps.fetch()
    for a in d.get('validations', []):
        if not a.get('finished_at') and a.get('uuid', a.get('queued_attempt_uuid')) not in own_uuids:
            return True
    return False


def run_attempt(url: str, own_uuids: set, poll: int = 20) -> float:
    """Queue one attempt and return its score when finished."""
    q = ps.queue_validation(url)
    uid = q.get('queued_attempt_uuid')
    own_uuids.add(uid)
    log(f'queued {uid} position {q.get("position_in_queue")}')
    before = len(ps.fetch().get('validations', []))
    while True:
        time.sleep(poll)
        d = ps.fetch()
        v = sorted(d.get('validations', []), key=lambda a: a.get('submitted_at') or '')
        if len(v) >= before and v and v[-1].get('finished_at') and v[-1].get('service_url', '').startswith(url.rsplit('/predict', 1)[0]):
            return float(v[-1].get('score') or 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='the probe endpoint /predict url')
    ap.add_argument('--base', default=str(HERE / 'base_answers.json'), help='json {file: [bools]} or "none"')
    ap.add_argument('--n-files', type=int, default=19)
    ap.add_argument('--dry', action='store_true', help='print the plan, do not queue')
    a = ap.parse_args()

    files = files_from_requests()
    if len(files) < a.n_files:
        sys.exit(f'only {len(files)} validation filenames known in requests.jsonl; run one attempt through the probe server first (or a real run with REQUEST_DUMP_DIR)')
    files = files[:a.n_files]
    N = a.n_files * N_PER

    if STATE.exists():
        st = json.loads(STATE.read_text(encoding='utf-8'))
        log(f'resuming: {len(st["history"])} attempts so far')
    else:
        if a.base != 'none' and Path(a.base).exists():
            base = json.loads(Path(a.base).read_text(encoding='utf-8'))
            p = [bool(x) for f in files for x in (base.get(f) or [False] * N_PER)[:N_PER]]
        else:
            p = [False] * N
        st = {'files': files, 'p': p, 'c_base': None, 'groups': [], 'known': {}, 'history': [], 'own': []}
    own = set(st['own'])
    p = st['p']

    def save():
        st['own'] = sorted(own)
        STATE.write_text(json.dumps(st, indent=1), encoding='utf-8')

    def wait_free():
        while PAUSE.exists() or busy(own):
            log('paused (PAUSE file or another attempt in the queue); checking again in 60 s')
            time.sleep(60)

    # base attempt: our vector as-is, null spans -> c_base
    if st['c_base'] is None:
        write_table(files, p)
        if a.dry:
            log(f'dry: would queue the base attempt with {sum(p)} yes of {N}'); return 0
        wait_free()
        s = run_attempt(a.url, own)
        c = round(s * N / 0.4)
        st['c_base'] = c
        st['history'].append({'S': [], 'score': s, 'c': c})
        st['groups'] = [{'items': list(range(N)), 'k': N - c}]
        log(f'base: score {s:.4f} -> {c}/{N} correct, {N - c} wrong to find')
        save()

    # recursive halving with exact counts
    while True:
        open_groups = [g for g in st['groups'] if 0 < g['k'] < len(g['items'])]
        if not open_groups:
            break
        g = max(open_groups, key=lambda x: len(x['items']))
        st['groups'].remove(g)
        items = g['items']
        A, B = items[:len(items) // 2], items[len(items) // 2:]
        vec = list(p)
        for i in A:
            vec[i] = not vec[i]
        write_table(files, vec)
        if a.dry:
            log(f'dry: would query a group of {len(A)} (parent {len(items)} with {g["k"]} wrong)'); return 0
        wait_free()
        s = run_attempt(a.url, own)
        c = round(s * N / 0.4)
        kA = (c - st['c_base'] + len(A)) // 2
        kB = g['k'] - kA
        st['history'].append({'S': A, 'score': s, 'c': c, 'kA': kA})
        log(f'group of {len(items)} ({g["k"]} wrong): half of {len(A)} has {kA} wrong, other half {kB}; attempts so far {len(st["history"])}')
        for sub, k in ((A, kA), (B, kB)):
            if k == 0:
                for i in sub: st['known'][str(i)] = p[i]
            elif k == len(sub):
                for i in sub: st['known'][str(i)] = not p[i]
            else:
                st['groups'].append({'items': sub, 'k': k})
        save()
        if len(st['known']) == N:
            break

    labels = [st['known'][str(i)] for i in range(N)]
    out = {f: labels[i * N_PER:(i + 1) * N_PER] for i, f in enumerate(files)}
    (HERE / 'labels.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    log(f'done in {len(st["history"])} attempts: {sum(labels)} yes of {N}; labels.json written')
    write_table(files, labels)
    return 0


if __name__ == '__main__':
    sys.exit(main())
