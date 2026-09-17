"""Read our team's attempts and scores from the competition portal.

Reads the team API key from the gitignored file .claude/nordic-api-key at the
repo root (one line), or from the NORDIC_API_KEY environment variable, and
calls the status endpoint only. This script never queues anything.

    python bench/portal_status.py            # all attempts, newest last
    python bench/portal_status.py --watch    # poll every 30 s until the newest attempt finishes
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
STATUS_URL = 'https://cases.nordicaicup.com/api/v1/usecases/medical-appointment/status'


def key() -> str:
    k = os.environ.get('NORDIC_API_KEY', '').strip()
    for cand in (KEY_FILE, KEY_FILE.with_suffix('.txt')):
        if not k and cand.exists():
            k = cand.read_text(encoding='utf-8').strip().splitlines()[0].strip()
    if not k:
        sys.exit(f'no key: put it on one line in {KEY_FILE} (gitignored) or set NORDIC_API_KEY')
    return k


def fetch() -> dict:
    r = requests.get(STATUS_URL, headers={'x-token': key()}, timeout=30)
    r.raise_for_status()
    return r.json()


def fmt(ts):
    if not ts:
        return '-'
    try:
        return dt.datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone().strftime('%H:%M:%S')
    except Exception:
        return str(ts)


def show(d: dict) -> None:
    print(f"team {d.get('team_name')}   validations {d.get('n_validations')}   final attempts used {d.get('n_evaluations')}")
    rows = sorted(d.get('validations', []), key=lambda a: a.get('submitted_at') or '')
    print(f"  {'#':>2} {'submitted':>9} {'started':>9} {'finished':>9} {'score':>7}  url / errors")
    for i, a in enumerate(rows, 1):
        sc = a.get('score')
        sc = f'{sc:.4f}' if isinstance(sc, (int, float)) else '-'
        err = '; '.join(a.get('errors') or [])[:80]
        url = (a.get('service_url') or '')[:60]
        print(f"  {i:>2} {fmt(a.get('submitted_at')):>9} {fmt(a.get('started_at')):>9} {fmt(a.get('finished_at')):>9} {sc:>7}  {url}{('  ERR ' + err) if err else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    a = ap.parse_args()
    d = fetch()
    show(d)
    if not a.watch:
        return 0
    n = len(d.get('validations', []))
    newest = max(d.get('validations', []), key=lambda x: x.get('submitted_at') or '', default=None)
    while newest is None or not newest.get('finished_at'):
        time.sleep(30)
        d = fetch()
        newest = max(d.get('validations', []), key=lambda x: x.get('submitted_at') or '', default=None)
        if len(d.get('validations', [])) != n or (newest and newest.get('finished_at')):
            print('---'); show(d)
            if newest and newest.get('finished_at'):
                break
    return 0


if __name__ == '__main__':
    sys.exit(main())
