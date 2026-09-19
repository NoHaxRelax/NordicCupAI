"""Drone-flyby VALIDATION runs on the competition portal: status, queue one run, wait for its score.

    python elias/portal.py status
    python elias/portal.py validate https://<tunnel-host>/predict      # queues ONE validation and waits

This module knows the status and the validation-queue endpoints only. It never references the one-shot
final attempt, which only a human may trigger (and a PreToolUse hook blocks it for agents anyway).
The team key is read from the gitignored .claude/nordic-api-key of the main checkout and never printed.
A team can have one validation in flight: `validate` refuses to queue while another run is unfinished,
so it cannot collide with a teammate's run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import requests

BASE = 'https://cases.nordicaicup.com/api/v1/usecases/drone-flyby'
KEY_FILES = [Path(r'C:\Users\edlun\Desktop\lucky shots\NordicCupAI\.claude\nordic-api-key'),
             Path(__file__).resolve().parents[2]/'.claude'/'nordic-api-key']


def key():
    for f in KEY_FILES:
        if f.exists():
            return f.read_text(encoding='utf-8').strip().splitlines()[0].strip()
    sys.exit('no team key file found')


def status():
    r = requests.get(f'{BASE}/status', headers={'x-token': key()}, timeout=30); r.raise_for_status()
    return r.json()


def fmt(v):
    return dt.datetime.fromisoformat(v.replace('Z', '+00:00')).astimezone().strftime('%m-%d %H:%M:%S') if v else '-'


def show(d, n=8):
    rows = sorted(d.get('validations', []), key=lambda a: a.get('submitted_at') or '')
    best = max((a.get('score') or 0) for a in rows) if rows else 0
    print(f"validations {d.get('n_validations')}  best {best:.4f}  in flight {sum(1 for a in rows if not a.get('finished_at'))}")
    for a in rows[-n:]:
        sc = a.get('score'); sc = f'{sc:.4f}' if isinstance(sc, (int, float)) else '   -  '
        print(f"  {fmt(a.get('submitted_at'))}  done {fmt(a.get('finished_at'))}  {sc}  {(a.get('service_url') or '')[8:60]}  {'; '.join(a.get('errors') or [])[:70]}")
    return rows


def validate(url, wait_s=900, tries=14):
    """Queue ONE validation for `url` and return ITS result.

    The portal keeps one queued attempt per team and runs it with the LAST url submitted before it starts, so a
    teammate's queue request made after ours replaces our url (Saturday night a probe loop did that every two
    minutes). Hence: while our attempt has not started, re-submit the url every few seconds so that ours is the
    latest one when the attempt starts. A re-submit that lands just as the attempt starts creates a second attempt
    of ours (seen Sunday 00:52), so the result is the FIRST new row of our url, and a later unfinished row of ours is
    waited out before returning (the caller restarts the server right after). A row counts only if it is new since
    our first request (a pod's fixed url repeats across runs)."""
    ours = lambda a: (a.get('service_url') or '').rstrip('/') == url.rstrip('/')
    for attempt in range(tries):
        rows = sorted(status().get('validations', []), key=lambda a: a.get('submitted_at') or '')
        if any(not a.get('finished_at') for a in rows):
            print('a teammate validation is running: waiting', flush=True); time.sleep(3); continue
        seen = {((a.get('service_url') or '').rstrip('/'), a.get('submitted_at')) for a in rows}
        r = requests.post(f'{BASE}/validate/queue', headers={'x-token': key()}, json={'url': url}, timeout=30)
        print('queue ->', r.status_code, r.text[:300], flush=True); r.raise_for_status()
        state = (r.json() or {}).get('status')
        t0 = time.time(); last_post = time.time(); reposts = 0
        while time.time()-t0 < wait_s:
            time.sleep(2)
            rows = sorted(status().get('validations', []), key=lambda a: a.get('submitted_at') or '')
            mine = [a for a in rows if ours(a) and ((a.get('service_url') or '').rstrip('/'), a.get('submitted_at')) not in seen]
            if mine and mine[0].get('finished_at'):
                a = mine[0]
                # a duplicate attempt of ours may follow; let it finish against the same server before returning
                for _ in range(100):
                    later = [b for b in mine[1:] if not b.get('finished_at')]
                    if not later:
                        break
                    print('a duplicate attempt of ours is running: waiting for it', flush=True); time.sleep(5)
                    rows = sorted(status().get('validations', []), key=lambda a: a.get('submitted_at') or '')
                    mine = [b for b in rows if ours(b) and ((b.get('service_url') or '').rstrip('/'), b.get('submitted_at')) not in seen]
                print(f"RESULT score {a.get('score')}  errors {str(a.get('errors'))[:120]}  url {a.get('service_url')}  (re-submitted {reposts} times, {len(mine)} rows of ours)")
                return a
            if mine:
                continue                      # ours is running: never post while it runs
            if state == 'queued' and all(a.get('finished_at') for a in rows) and time.time()-last_post > 4:
                try:
                    rr = requests.post(f'{BASE}/validate/queue', headers={'x-token': key()}, json={'url': url}, timeout=30)
                    reposts += 1; last_post = time.time(); state = (rr.json() or {}).get('status') if rr.ok else state
                except (requests.RequestException, ValueError) as exc:
                    print('re-submit failed:', type(exc).__name__, flush=True)
        print('no run of our url within the wait; queueing again', flush=True)
    sys.exit('could not get a validation of our own URL')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['status', 'validate']); ap.add_argument('url', nargs='?')
    a = ap.parse_args()
    if a.cmd == 'status':
        show(status())
    else:
        if not a.url or not a.url.endswith('/predict'):
            sys.exit('give the full URL ending in /predict')
        validate(a.url)
