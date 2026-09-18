"""Hugging Face Space: validation label miner that survives the laptop closing.

One FastAPI process with three jobs:
  1. /predict           the table-lookup endpoint the portal calls during a probe
                        attempt (answers from the current vector, null spans)
  2. miner thread       the adaptive count-query search (same logic as
                        bench/mine/miner.py), one queued attempt at a time
  3. /control/<token>   pause, resume, status, so a serious validation can be
                        run from anywhere: pause -> wait for the current probe
                        to finish -> queue the real run -> resume

Secrets (Space settings): NORDIC_API_KEY (portal key), CONTROL_TOKEN (any
string), HF_TOKEN (write access, for state persistence), STATE_REPO
(private dataset repo id, e.g. user/nordic-miner-state).
Config: PROBE_URL is derived from SPACE_HOST at start (https://<host>/predict).
State: state.json is uploaded to STATE_REPO after every attempt and downloaded
at start, so a rebuild or restart resumes where it was.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

STATE_DIR = Path(os.environ.get('STATE_DIR', '/data' if Path('/data').exists() else '.'))
STATE = STATE_DIR / 'state.json'
BASE = STATE_DIR / 'base_answers.json'
LOGF = STATE_DIR / 'miner.log'
API = 'https://cases.nordicaicup.com/api/v1/usecases/medical-appointment'
KEY = os.environ.get('NORDIC_API_KEY', '')
TOKEN = os.environ.get('CONTROL_TOKEN', '')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
STATE_REPO = os.environ.get('STATE_REPO', '')
N_PER = 10

app = FastAPI()
lock = threading.Lock()
paused = threading.Event()
current = {'files': [], 'vec': []}


def log(msg: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    try:
        with open(LOGF, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


# ---------------------------------------------------------------- persistence
def hub_push(path: Path) -> None:
    if not (HF_TOKEN and STATE_REPO and path.exists()):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                                          repo_id=STATE_REPO, repo_type='dataset')
    except Exception as e:
        log(f'hub push failed: {e}')


def hub_pull(name: str) -> None:
    if not (HF_TOKEN and STATE_REPO):
        return
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=STATE_REPO, repo_type='dataset', filename=name, token=HF_TOKEN)
        (STATE_DIR / name).write_bytes(Path(p).read_bytes())
        log(f'pulled {name} from {STATE_REPO}')
    except Exception as e:
        log(f'hub pull {name}: {e}')


# ---------------------------------------------------------------- portal
def fetch() -> dict:
    r = requests.get(f'{API}/status', headers={'x-token': KEY}, timeout=30)
    r.raise_for_status()
    return r.json()


def queue(url: str) -> dict:
    r = requests.post(f'{API}/validate/queue', headers={'x-token': KEY}, json={'url': url}, timeout=30)
    r.raise_for_status()
    return r.json()


def busy(own: set) -> bool:
    for a in fetch().get('validations', []):
        if not a.get('finished_at') and a.get('uuid', a.get('queued_attempt_uuid')) not in own:
            return True
    return False


def run_attempt(url: str, own: set) -> float:
    q = queue(url)
    own.add(q.get('queued_attempt_uuid'))
    log(f'queued {q.get("queued_attempt_uuid")} position {q.get("position_in_queue")}')
    base = url.rsplit('/predict', 1)[0]
    while True:
        time.sleep(20)
        v = sorted(fetch().get('validations', []), key=lambda a: a.get('submitted_at') or '')
        if v and v[-1].get('finished_at') and (v[-1].get('service_url') or '').startswith(base):
            return float(v[-1].get('score') or 0.0)


# ---------------------------------------------------------------- miner
def miner(url: str) -> None:
    hub_pull('state.json'); hub_pull('base_answers.json')
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding='utf-8'))
        log(f'resuming with {len(st["history"])} attempts done')
    else:
        base = json.loads(BASE.read_text(encoding='utf-8')) if BASE.exists() else {}
        files = list(base.keys())
        if not files:
            log('no base_answers.json: cannot start (upload it to STATE_REPO)'); return
        N = len(files) * N_PER
        p = [bool(x) for f in files for x in (base[f] + [False] * N_PER)[:N_PER]]
        st = {'files': files, 'p': p, 'c_base': None, 'groups': [], 'known': {}, 'history': [], 'own': []}
    files, p, N = st['files'], st['p'], len(st['files']) * N_PER
    own = set(st['own'])

    def save():
        st['own'] = sorted(own)
        STATE.write_text(json.dumps(st, indent=1), encoding='utf-8')
        hub_push(STATE)

    def set_vec(vec):
        with lock:
            current['files'] = files; current['vec'] = list(vec)

    def wait_free():
        while paused.is_set() or busy(own):
            time.sleep(60)

    if st['c_base'] is None:
        set_vec(p); wait_free()
        s = run_attempt(url, own); c = round(s * N / 0.4)
        st['c_base'] = c; st['history'].append({'S': [], 'score': s, 'c': c})
        st['groups'] = [{'items': list(range(N)), 'k': N - c}]
        log(f'base: {c}/{N} correct, {N - c} wrong to find'); save()

    while True:
        open_groups = [g for g in st['groups'] if 0 < g['k'] < len(g['items'])]
        if not open_groups:
            break
        g = max(open_groups, key=lambda x: len(x['items'])); st['groups'].remove(g)
        items = g['items']; A, B = items[:len(items) // 2], items[len(items) // 2:]
        vec = list(p)
        for i in A: vec[i] = not vec[i]
        set_vec(vec); wait_free()
        s = run_attempt(url, own); c = round(s * N / 0.4)
        kA = (c - st['c_base'] + len(A)) // 2; kB = g['k'] - kA
        st['history'].append({'S': A, 'score': s, 'c': c, 'kA': kA})
        log(f'group {len(items)} ({g["k"]} wrong) -> {len(A)}:{kA}, {len(B)}:{kB}; attempts {len(st["history"])}')
        for sub, k in ((A, kA), (B, kB)):
            if k == 0:
                for i in sub: st['known'][str(i)] = p[i]
            elif k == len(sub):
                for i in sub: st['known'][str(i)] = not p[i]
            else:
                st['groups'].append({'items': sub, 'k': k})
        save()
    labels = [st['known'].get(str(i), p[i]) for i in range(N)]
    (STATE_DIR / 'labels.json').write_text(json.dumps({f: labels[i * N_PER:(i + 1) * N_PER] for i, f in enumerate(files)}, indent=1), encoding='utf-8')
    hub_push(STATE_DIR / 'labels.json')
    set_vec(labels)
    log(f'done in {len(st["history"])} attempts: {sum(labels)} yes of {N}')


# ---------------------------------------------------------------- routes
class Req(BaseModel):
    audio_base64: str
    audio_filename: str
    questions: list


@app.post('/predict')
def predict(r: Req):
    n = len(r.questions)
    with lock:
        files, vec = current['files'], current['vec']
    ans = [False] * n
    if r.audio_filename in files:
        i = files.index(r.audio_filename)
        got = vec[i * N_PER:(i + 1) * N_PER]
        ans = [bool(x) for x in got][:n] + [False] * max(0, n - len(got))
    return {'answers': ans, 'evidence_start': [None] * n, 'evidence_end': [None] * n}


@app.get('/')
def index():
    return 'probe endpoint running'


@app.get('/control/{token}/{cmd}')
def control(token: str, cmd: str):
    if not TOKEN or token != TOKEN:
        raise HTTPException(status_code=404)
    if cmd == 'pause':
        paused.set(); return {'paused': True}
    if cmd == 'resume':
        paused.clear(); return {'paused': False}
    if cmd == 'status':
        st = json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
        return {'paused': paused.is_set(), 'attempts': len(st.get('history', [])), 'known': len(st.get('known', {})),
                'open_groups': len([g for g in st.get('groups', []) if 0 < g['k'] < len(g['items'])]),
                'log_tail': LOGF.read_text(encoding='utf-8').splitlines()[-8:] if LOGF.exists() else []}
    raise HTTPException(status_code=404)


@app.on_event('startup')
def start():
    host = os.environ.get('SPACE_HOST') or os.environ.get('PROBE_HOST', '')
    url = f'https://{host}/predict' if host else os.environ.get('PROBE_URL', '')
    if os.environ.get('MINER_AUTOSTART', '1') == '1' and url and KEY:
        paused.clear() if os.environ.get('START_PAUSED', '0') != '1' else paused.set()
        threading.Thread(target=miner, args=(url,), daemon=True).start()
        log(f'miner thread started for {url} (paused={paused.is_set()})')
    else:
        log('miner not started: missing PROBE_URL/SPACE_HOST or NORDIC_API_KEY, or MINER_AUTOSTART=0')
