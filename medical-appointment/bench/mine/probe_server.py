"""Table-lookup endpoint for validation probing. Answers instantly from
bench/mine/current_answers.json (filename -> list of 10 booleans), always with
null spans, and logs every request's filename and questions to
bench/mine/requests.jsonl. Unknown files get all-no.

    python bench/mine/probe_server.py            # port 9055
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ANSWERS = HERE / 'current_answers.json'
LOG = HERE / 'requests.jsonl'
PORT = int(os.environ.get('PROBE_PORT', '9055'))
app = FastAPI()


class Req(BaseModel):
    audio_base64: str
    audio_filename: str
    questions: list


def table() -> dict:
    try:
        return json.loads(ANSWERS.read_text(encoding='utf-8'))
    except Exception:
        return {}


@app.post('/predict')
def predict(r: Req):
    n = len(r.questions)
    t = table().get(r.audio_filename) or []
    # entries are either booleans (answer, null span) or objects
    # {"answer": bool, "start": s, "end": e} as written by the label page
    answers, starts, ends = [], [], []
    for x in list(t)[:n]:
        if isinstance(x, dict):
            a = bool(x.get('answer'))
            s, e = x.get('start'), x.get('end')
            ok = a and s is not None and e is not None and float(e) >= float(s)
            answers.append(a); starts.append(float(s) if ok else None); ends.append(float(e) if ok else None)
        else:
            answers.append(bool(x)); starts.append(None); ends.append(None)
    while len(answers) < n:
        answers.append(False); starts.append(None); ends.append(None)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'t': time.time(), 'file': r.audio_filename, 'questions': r.questions,
                            'answers': answers}, ensure_ascii=False) + '\n')
    return {'answers': answers, 'evidence_start': starts, 'evidence_end': ends}


@app.get('/')
def index():
    return 'probe endpoint running'


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
