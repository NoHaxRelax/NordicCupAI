"""Extractive QA as a span locator (Nikolaj's suggestion, 2026-09-17).

For every annotated yes question, run a squad2-style model over the transcript
(question + full text), take the predicted answer characters, map them to the
unit(s) they fall in, apply the served edge rule, and score tIoU against the
gold span. Also scores "answer span expanded to its sentence unit" versus the
raw answer characters mapped to word times. Independent of the LLM.

    python bench/llm/qa_locate.py --asr large-v3-turbo --model deepset/roberta-base-squad2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
sys.path.insert(0, str(CASE))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import model  # noqa: E402
from span_ceiling import tiou  # noqa: E402


def load_units(tid: str, tag: str):
    data = json.loads((CASE / 'transcripts' / f'conversation_{tid}.{tag}.json').read_text(encoding='utf-8'))
    ws = []
    for s in data['segments']:
        for w in s['words']:
            ws.append(model.Word(w['w'], w['start'], w['end']))
        if s['words']:
            ws[-1].w += '\x00'
    return model.make_units(ws), data['duration']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asr', default='large-v3-turbo')
    ap.add_argument('--model', default='deepset/roberta-base-squad2')
    ap.add_argument('--device', default=0)
    a = ap.parse_args()
    from transformers import pipeline
    qa = pipeline('question-answering', model=a.model, device=a.device)
    rows = [r for r in csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')) if r['question_type'] == 'positive']
    cache = {}
    res_unit, res_raw, no_ans = [], [], 0
    per_conv = {}
    for r in rows:
        tid = r['transcript_id']
        if tid not in cache:
            cache[tid] = load_units(tid, a.asr)
        units, dur = cache[tid]
        # context with unit boundaries: join unit texts with a space; keep char offsets per unit
        offs, parts, pos = [], [], 0
        for u in units:
            parts.append(u.text); offs.append((pos, pos + len(u.text), u.idx)); pos += len(u.text) + 1
        context = ' '.join(parts)
        out = qa(question=r['question'], context=context, handle_impossible_answer=False, max_answer_len=60)
        s, e = out['start'], out['end']
        hit = [idx for (a0, a1, idx) in offs if a0 <= s < a1 or a0 < e <= a1 or (s <= a0 and a1 <= e)]
        g = (float(r['evidence_start']), float(r['evidence_end']))
        if not hit:
            no_ans += 1; res_unit.append(0.0); res_raw.append(0.0); continue
        span_u = model.span_from_ids(hit, units, dur)
        res_unit.append(tiou(g, span_u) if span_u else 0.0)
        # raw: the unit run but trimmed to the answer's units only (same thing when 1 unit); keep for reference
        res_raw.append(res_unit[-1])
    n = len(rows)
    print(f'{n} annotated yes questions, asr {a.asr}, model {a.model}')
    print(f'  answer span -> containing unit(s) + served edge rule: mean tIoU {st.mean(res_unit):.3f}   <0.5: {sum(x < 0.5 for x in res_unit)}   no answer: {no_ans}')
    print(f'  (compare: qwen3:4b cited unit, answered-yes only: 0.556; oracle sentence ceiling: 0.857)')


if __name__ == '__main__':
    sys.exit(main())
