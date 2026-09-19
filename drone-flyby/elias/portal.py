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


def validate(url, wait_s=900):
    d = status(); rows = show(d, 3)
    if any(not a.get('finished_at') for a in rows):
        sys.exit('another validation is still running: not queueing')
    r = requests.post(f'{BASE}/validate/queue', headers={'x-token': key()}, json={'url': url}, timeout=30)
    print('queue ->', r.status_code, r.text[:300]); r.raise_for_status()
    t0 = time.time(); n0 = len(rows)
    while time.time()-t0 < wait_s:
        time.sleep(15)
        rows = sorted(status().get('validations', []), key=lambda a: a.get('submitted_at') or '')
        if len(rows) > n0 and rows[-1].get('finished_at'):
            a = rows[-1]; print(f"RESULT score {a.get('score')}  errors {a.get('errors')}  url {a.get('service_url')}"); return a
    sys.exit('timed out waiting for the validation to finish')


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
