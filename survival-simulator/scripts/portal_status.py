"""Read our team's survival-simulator attempts, and queue validation runs.

Reads the team API key from the gitignored file .claude/nordic-api-key at the
repo root (one line), or from the NORDIC_API_KEY environment variable.

This script knows the verify, validate and status endpoints. It does not know
the evaluation endpoint and never will: the team has one evaluation attempt and
only a human queues it, from the portal.

    python scripts/portal_status.py                       # attempts so far, newest last
    python scripts/portal_status.py --verify URL          # format check, costs nothing
    python scripts/portal_status.py --queue URL           # queue one validation attempt
    python scripts/portal_status.py --watch               # poll until the newest attempt finishes

URL is the /predict endpoint exactly as the portal will call it, for example
http://203.0.113.7:9052/predict -- the path is used as given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = ROOT / '.claude' / 'nordic-api-key'
API = 'https://cases.nordicaicup.com/api/v1/usecases/survival-simulator'
POLL_SECONDS = 30


def key() -> str:
    k = os.environ.get('NORDIC_API_KEY', '').strip()
    for cand in (KEY_FILE, KEY_FILE.with_suffix('.txt')):
        if not k and cand.exists():
            k = cand.read_text(encoding='utf-8').strip().splitlines()[0].strip()
    if not k:
        sys.exit(f'no key: put it on one line in {KEY_FILE} (gitignored) or set NORDIC_API_KEY')
    return k


def checked(r: requests.Response) -> dict:
    """The portal answers a wrong key with 422 "Invalid token", not 401, and a
    missing header with 403. Report either as itself rather than a traceback."""
    if r.status_code in (401, 403, 422):
        body = r.json() if r.headers.get('content-type', '').startswith('application/json') else {}
        sys.exit(f'portal rejected the request ({r.status_code}): '
                 f'{body.get("message", r.text[:200])} -- check the key in {KEY_FILE}')
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    return checked(requests.get(f'{API}{path}', headers={'x-token': key()}, timeout=30))


def post(path: str, url: str) -> dict:
    return checked(requests.post(f'{API}{path}', headers={'x-token': key()},
                                 json={'url': url}, timeout=60))


def fmt_time(ts) -> str:
    if not ts:
        return '-'
    try:
        return dt.datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone().strftime('%H:%M:%S')
    except ValueError:
        return str(ts)


def attempts(d: dict) -> list:
    return sorted(d.get('validations', []) or [], key=lambda a: a.get('submitted_at') or '')


def show(d: dict) -> None:
    print(f"team {d.get('team_name')}   validations {d.get('n_validations')}"
          f"   final attempts used {d.get('n_evaluations')}")
    print(f"  {'#':>2} {'submitted':>9} {'started':>9} {'finished':>9} {'score':>9}  url / errors")
    for i, a in enumerate(attempts(d), 1):
        score = a.get('score')
        score = f'{score:.3f}' if isinstance(score, (int, float)) else '-'
        errors = '; '.join(a.get('errors') or [])[:80]
        url = (a.get('service_url') or '')[:60]
        print(f"  {i:>2} {fmt_time(a.get('submitted_at')):>9} {fmt_time(a.get('started_at')):>9}"
              f" {fmt_time(a.get('finished_at')):>9} {score:>9}  {url}"
              f"{('  ERR ' + errors) if errors else ''}")


def newest(d: dict):
    rows = attempts(d)
    return rows[-1] if rows else None


def watch() -> None:
    """Poll the status endpoint until the newest attempt reports a finish time."""
    while True:
        d = get('/status')
        show(d)
        latest = newest(d)
        if latest is not None and latest.get('finished_at'):
            return
        time.sleep(POLL_SECONDS)
        print('---')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify', metavar='URL', help='send the portal format check to this /predict url')
    ap.add_argument('--queue', metavar='URL', help='queue one validation attempt against this /predict url')
    ap.add_argument('--watch', action='store_true', help='poll until the newest attempt finishes')
    args = ap.parse_args()

    if args.verify:
        print(post('/verify', args.verify))

    if args.queue:
        q = post('/validate/queue', args.queue)
        print(f"queued validation {q.get('queued_attempt_uuid')}"
              f"  status {q.get('status')}  position {q.get('position_in_queue')}")
        args.watch = True
        time.sleep(5)

    if args.watch:
        watch()
    else:
        show(get('/status'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
