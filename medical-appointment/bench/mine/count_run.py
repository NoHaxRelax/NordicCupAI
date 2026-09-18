"""One count query against the validation set: build the probe table from the hand
answers (optionally with some answers flipped), queue ONE validation attempt at the
probe endpoint, wait for it, and turn the score into the number of correct binaries.

With null spans the score is 0.4 * correct / 190, so correct = score * 190 / 0.4
exactly (one question is 0.0021, the portal reports four decimals). Flipping a set S
of answers changes the count by |S| - 2 * (correct in S), which gives the number of
correct answers inside S from one run.

    python bench/mine/count_run.py --url https://xxx.trycloudflare.com          # all hand answers
    python bench/mine/count_run.py --url ... --flip 3                            # sample 3 inverted
    python bench/mine/count_run.py --url ... --flip 8:3,8:6 --label "s8 q3 q6"

Only the validation queue is touched (via bench/portal_status.py). Results append to
bench/mine/count_runs.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
sys.path.insert(0, str(CASE / 'bench'))
import portal_status  # noqa: E402

LOG = HERE / 'count_runs.jsonl'
N_Q = 190


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True, help='public base url of the probe endpoint (without /predict)')
    ap.add_argument('--flip', default='')
    ap.add_argument('--spans', action='store_true')
    ap.add_argument('--label', default='')
    ap.add_argument('--poll', type=int, default=20)
    a = ap.parse_args()
    cmd = [sys.executable, str(HERE / 'answers_md.py')] + (['--flip', a.flip] if a.flip else []) + (['--spans'] if a.spans else [])
    print(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
    service = a.url.rstrip('/') + '/predict'
    before = {v.get('submitted_at') for v in portal_status.fetch().get('validations', [])}
    q = portal_status.queue_validation(service)
    print(f"queued {q.get('queued_attempt_uuid')} status {q.get('status')} position {q.get('position_in_queue')}", flush=True)
    t0 = time.time()
    while True:
        time.sleep(a.poll)
        vals = [v for v in portal_status.fetch().get('validations', []) if v.get('submitted_at') not in before]
        newest = max(vals, key=lambda v: v.get('submitted_at') or '', default=None)
        if newest and newest.get('finished_at'):
            break
        print(f'  waiting {time.time() - t0:5.0f}s', flush=True)
    score = newest.get('score')
    errs = newest.get('errors') or []
    rec = {'t': time.time(), 'label': a.label, 'flip': a.flip, 'spans': a.spans, 'score': score, 'errors': errs,
           'elapsed': round(time.time() - t0)}
    if isinstance(score, (int, float)) and not a.spans:
        c = score * N_Q / 0.4
        rec['correct'] = round(c)
        print(f'score {score:.4f} -> {c:.2f} correct of {N_Q} (rounded {round(c)}), {rec["elapsed"]}s')
    else:
        print(f'score {score} errors {errs}')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
