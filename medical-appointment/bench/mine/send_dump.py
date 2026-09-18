"""Replay dumped requests against a /predict endpoint and record the answers.

request_dump/ holds the validation conversations the portal sent us (audio +
questions, written by example.py with REQUEST_DUMP_DIR). This sends each of them
to an endpoint exactly as the evaluator would (one POST per conversation, 60 s
timeout), writes <out>/answers.jsonl in the format example.py dumps, and prints
the wall time per conversation. bench/mine/val_diag.py --dump <out> then scores
the answers against the hand labels. Nothing here touches the portal.

    python bench/mine/send_dump.py --url http://localhost:9055/predict --out bench/results/served/27b
    python bench/mine/send_dump.py --url ... --out ... --only conversation_sample_11 conversation_sample_14
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:9054/predict')
    ap.add_argument('--dump', default=str(CASE / 'request_dump'))
    ap.add_argument('--out', required=True, help='directory for answers.jsonl (created)')
    ap.add_argument('--only', nargs='*', default=None, help='stems to send (default all)')
    ap.add_argument('--timeout', type=float, default=60.0)
    a = ap.parse_args()
    dump, out = Path(a.dump), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted((p.stem for p in dump.glob('*.mp3')), key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0))
    if a.only:
        stems = [s for s in stems if s in set(a.only)]
    lines, walls, failed = [], [], []
    for stem in stems:
        audio = (dump / f'{stem}.mp3').read_bytes()
        questions = json.loads((dump / f'{stem}.questions.json').read_text(encoding='utf-8'))
        payload = {'audio_filename': f'{stem}.mp3', 'audio_base64': base64.b64encode(audio).decode('ascii'),
                   'questions': questions}
        t0 = time.time()
        try:
            r = requests.post(a.url, json=payload, timeout=a.timeout)
            r.raise_for_status()
            d = r.json()
            spans = [[s, e] if s is not None and e is not None else None
                     for s, e in zip(d['evidence_start'], d['evidence_end'])]
            rec = {'file': f'{stem}.mp3', 'questions': questions, 'answers': [bool(x) for x in d['answers']], 'spans': spans}
        except Exception as exc:
            failed.append(stem)
            rec = {'file': f'{stem}.mp3', 'questions': questions, 'answers': [True] * len(questions),
                   'spans': [None] * len(questions), 'error': f'{type(exc).__name__}: {exc}'}
        wall = time.time() - t0
        walls.append(wall)
        lines.append(json.dumps(rec, ensure_ascii=False))
        print(f'{stem:<24} {wall:5.1f} s  yes={sum(rec["answers"])}/{len(questions)}'
              + (f'  FAILED {rec["error"][:80]}' if 'error' in rec else ''), flush=True)
    (out / 'answers.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(stems)} conversations, wall mean {sum(walls) / max(1, len(walls)):.1f} s, worst {max(walls, default=0):.1f} s'
          f' (budget {a.timeout:.0f} s), failed {len(failed)} -> {out / "answers.jsonl"}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
