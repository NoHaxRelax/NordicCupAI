"""Descriptive statistics of what the 39 training consultations are about.

Asks the local Ollama model to read each cached transcript and list the
conditions discussed, the medications, and the reason for the visit, as JSON.
Then aggregates: distinct conditions, how many per conversation, the most
frequent ones. Output: bench/results/topics.json and a printed table.

    python bench/describe_topics.py                 # qwen3:4b on localhost:11434
    LLM_MODEL=qwen3.5:4b python bench/describe_topics.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

import requests

CASE = Path(__file__).resolve().parent.parent
URL = os.environ.get('LLM_URL', 'http://localhost:11434').rstrip('/')
MODEL = os.environ.get('LLM_MODEL', 'qwen3:4b')
TAG = os.environ.get('ASR_TAG', 'large-v3')

SCHEMA = {
    'type': 'object',
    'properties': {
        'reason_for_visit': {'type': 'string'},
        'conditions': {'type': 'array', 'items': {'type': 'string'}},
        'medications': {'type': 'array', 'items': {'type': 'string'}},
        'specialty': {'type': 'string'},
    },
    'required': ['reason_for_visit', 'conditions', 'medications', 'specialty'],
}
SYSTEM = ('You read a transcript of a doctor-patient consultation and extract, as JSON: '
          'reason_for_visit (one short phrase), conditions (every medical condition, diagnosis or '
          'complaint discussed, as short canonical lowercase names, e.g. "asthma", "type 2 diabetes", '
          '"urinary tract infection"; no medications here), medications (drug names mentioned, lowercase), '
          'specialty (general practice, or the specialty if obvious). Only what is in the transcript.')


def canon(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\b(the|a|an|patient\'?s?|his|her)\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    files = sorted((CASE / 'transcripts').glob(f'*.{TAG}.json'), key=lambda p: int(re.sub(r'\D', '', p.stem.split('.')[0])))
    out = {}
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        text = ' '.join(s['text'].strip() for s in d['segments'])
        body = {'model': MODEL, 'messages': [{'role': 'system', 'content': SYSTEM},
                                             {'role': 'user', 'content': text}],
                'stream': False, 'think': False, 'format': SCHEMA,
                'options': {'temperature': 0.0, 'num_ctx': 3072, 'num_predict': 300}, 'keep_alive': -1}
        r = requests.post(f'{URL}/api/chat', json=body, timeout=120)
        r.raise_for_status()
        res = json.loads(r.json()['message']['content'])
        res['conditions'] = sorted({canon(c) for c in res['conditions'] if canon(c)})
        res['medications'] = sorted({canon(c) for c in res['medications'] if canon(c)})
        tid = f.stem.split('.')[0].replace('conversation_', '')
        out[tid] = res
        print(f'{tid:<10} {res["reason_for_visit"][:50]:<50} | {", ".join(res["conditions"])}', flush=True)
    (CASE / 'bench' / 'results').mkdir(exist_ok=True)
    (CASE / 'bench' / 'results' / 'topics.json').write_text(json.dumps(out, indent=1), encoding='utf-8')

    cond = collections.Counter(c for r in out.values() for c in r['conditions'])
    meds = collections.Counter(m for r in out.values() for m in r['medications'])
    per = [len(r['conditions']) for r in out.values()]
    spec = collections.Counter(r['specialty'].lower() for r in out.values())
    print(f'\nconversations {len(out)}; distinct conditions {len(cond)}; conditions per conversation '
          f'mean {sum(per)/len(per):.1f}, min {min(per)}, max {max(per)}; distinct medications {len(meds)}')
    print('most frequent conditions:', ', '.join(f'{k} ({v})' for k, v in cond.most_common(15)))
    print('conditions seen once:', sum(1 for v in cond.values() if v == 1))
    print('most frequent medications:', ', '.join(f'{k} ({v})' for k, v in meds.most_common(12)))
    print('specialty:', dict(spec))


if __name__ == '__main__':
    sys.exit(main())
