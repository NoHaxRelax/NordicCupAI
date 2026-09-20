"""Queue THE one evaluation attempt of the medical-appointment use case. Run once, by explicit order of Elias.

Written on 2026-09-20 after Elias created .claude/EVAL_UNLOCK himself and told the agent to go ahead.
bench/portal_status.py deliberately knows nothing about this endpoint; this file is the only place
that does. Shape taken from the portal's own OpenAPI document (cases.nordicaicup.com/openapi.json):
POST /api/v1/usecases/medical-appointment/evaluate/queue, header x-token, body {"url": ...}. The portal
returns an already queued attempt instead of adding a second one, and refuses a new attempt once a
finished one exists, so this call cannot double-submit.

    python bench/eval_queue_once.py queue https://<POD_ID>-9054.proxy.runpod.net/predict
    python bench/eval_queue_once.py watch            # read-only: polls the portal status until the attempt has a score

`queue` refuses to call unless: zero evaluation attempts exist, no validation is in flight, and the
endpoint's GET /api reports the validated configuration with the breaker closed. It makes ONE request
and never retries by itself; on any transport error it prints the portal state and stops.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import portal_status as ps  # noqa: E402

EVAL_URL = 'https://cases.nordicaicup.com/api/v1/usecases/medical-appointment/evaluate/queue'
LOG = HERE / 'results' / 'served' / 'evaluation_attempt.json'


def state():
    d = ps.fetch()
    return d, d.get('n_evaluations'), d.get('evaluations') or [], [v for v in d.get('validations', []) if not v.get('finished_at')]


def queue(url: str) -> int:
    if not (url.startswith('https://') and url.endswith('/predict')):
        sys.exit('the URL must be https and end in /predict')
    d, n_eval, evals, inflight = state()
    print(f"portal: team {d.get('team_name')!r}  evaluations used {n_eval}  listed {len(evals)}  validations in flight {len(inflight)}")
    if n_eval or evals:
        sys.exit('REFUSED: an evaluation attempt already exists for this team; nothing sent')
    if inflight:
        sys.exit('REFUSED: a validation is still in flight; nothing sent')
    api = requests.get(url[:-len('/predict')] + '/api', timeout=20).json()
    m = api.get('model', {})
    want = {'llm_variant': 'units-fewshot-both', 'unit_split': 'clause-and', 'llm_backend': 'vllm',
            'llm_model': 'Qwen/Qwen3.8-27B', 'asr_model': 'large-v3-turbo', 'breaker_open': False}
    bad = {k: m.get(k) for k, v in want.items() if m.get(k) != v}
    print(f"endpoint: uptime {api.get('uptime')}  timed out {api.get('timed_out_conversations')}  counts {m.get('counts')}")
    if bad:
        sys.exit(f'REFUSED: the endpoint does not report the validated configuration: {bad}; nothing sent')
    print(f'sending ONE evaluation request for {url}')
    try:
        r = requests.post(EVAL_URL, headers={'x-token': ps.key()}, json={'url': url}, timeout=60)
    except Exception as exc:                      # never retry blindly: look at the portal first
        print(f'TRANSPORT ERROR: {exc!r}. Not retrying. Portal state now:')
        d, n_eval, evals, _ = state()
        print(f'  evaluations used {n_eval}  listed {evals}')
        return 2
    print(f'HTTP {r.status_code}  {r.text[:600]}')
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({'url': url, 'sent_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'http': r.status_code,
                               'response': r.text[:2000]}, indent=1), encoding='utf-8')
    return 0 if r.status_code == 200 else 3


def watch() -> int:
    t0 = time.time()
    last = None
    while time.time() - t0 < 3600:
        d, n_eval, evals, _ = state()
        cur = json.dumps(evals, sort_keys=True)
        if cur != last:
            print(time.strftime('%H:%M:%S'), 'evaluations used', n_eval, '|', evals, flush=True)
            last = cur
        if evals and all(e.get('finished_at') for e in evals):
            print('FINISHED'); return 0
        time.sleep(20)
    print('watch timed out after an hour; the attempt may still be queued')
    return 1


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'queue':
        sys.exit(queue(sys.argv[2]))
    if len(sys.argv) >= 2 and sys.argv[1] == 'watch':
        sys.exit(watch())
    sys.exit(__doc__)
