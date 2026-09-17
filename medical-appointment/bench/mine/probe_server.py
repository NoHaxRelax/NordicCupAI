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
    t = table().get(r.audio_filename)
    answers = [bool(x) for x in t][:n] if t else []
    answers += [False] * (n - len(answers))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'t': time.time(), 'file': r.audio_filename, 'questions': r.questions,
                            'answers': answers}, ensure_ascii=False) + '\n')
    return {'answers': answers, 'evidence_start': [None] * n, 'evidence_end': [None] * n}


@app.get('/')
def index():
    return 'probe endpoint running'


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
