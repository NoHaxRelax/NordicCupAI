"""Queue N VALIDATION runs back to back against one /predict URL and record every score.

Validation only: this uses bench/portal_status.queue_validation, which knows the validation queue and
nothing else. One run at a time: the next is queued only after the previous one has finished, and the
loop stops at the first run that reports an error, so a sick endpoint is never hammered.

    python bench/val_loop.py https://<POD_ID>-9054.proxy.runpod.net/predict --runs 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portal_status as ps  # noqa: E402


def mine(d, url):
    rows = [a for a in d.get('validations', []) if (a.get('service_url') or a.get('url') or '').rstrip('/') == url.rstrip('/')]
    return sorted(rows, key=lambda a: a.get('submitted_at') or '')       # the portal's list order is not guaranteed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('url')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', default=str(HERE / 'results' / 'served' / 'val_loop.json'))
    a = ap.parse_args()
    if not a.url.endswith('/predict'):
        sys.exit('give the full URL ending in /predict')
    results = []
    for k in range(1, a.runs + 1):
        d = ps.fetch()
        while any(not v.get('finished_at') for v in d.get('validations', [])):
            print('  a validation is still in flight: waiting'); time.sleep(20); d = ps.fetch()
        before = len(mine(d, a.url))
        q = ps.queue_validation(a.url)
        t0 = time.time()
        print(f"run {k}/{a.runs}: queued {q.get('queued_attempt_uuid')} position {q.get('position_in_queue')}", flush=True)
        while True:
            time.sleep(15)
            m = mine(ps.fetch(), a.url)
            if len(m) > before and m[-1].get('finished_at'):
                v = m[-1]; break
            if time.time() - t0 > 1500:
                sys.exit('no result after 25 minutes: stopping')
        err = v.get('errors') or v.get('error')
        if isinstance(err, str) and err.strip() in ('', '[]'):
            err = None
        v['score'] = float(v['score']) if v.get('score') is not None else None
        results.append({'run': k, 'score': v.get('score'), 'submitted_at': v.get('submitted_at'), 'started_at': v.get('started_at'),
                        'finished_at': v.get('finished_at'), 'errors': err, 'wall_s': round(time.time() - t0)})
        print(f"run {k}/{a.runs}: score {v.get('score')}  wall {round(time.time() - t0)} s  errors {err}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({'url': a.url, 'runs': results}, indent=1), encoding='utf-8')
        if err:
            print('a run reported errors: stopping the loop'); return 1
    scores = [r['score'] for r in results]
    print(f"done: {len(scores)} runs, scores {scores}, min {min(scores)}, max {max(scores)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
