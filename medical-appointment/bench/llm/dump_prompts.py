"""Write the exact prompts of a variant to files, one conversation per file, and score
answer files written by any model back through the variant's post-processing.

Used for the Claude-tier probe (models driven through the Claude Code Agent tool, which
cannot be pointed at the bench's HTTP client): dump, let an agent answer each prompt file
with the JSON only, then score.

    python bench/llm/dump_prompts.py dump --variant units-joint-demo --asr large-v3-turbo --out bench/results/probe/prompts
    python bench/llm/dump_prompts.py score --variant units-joint-demo --asr large-v3-turbo \
        --answers bench/results/probe/haiku --out bench/results/llm/claude-haiku.units-joint-demo.large-v3-turbo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CASE))
from bench import load_words, parse_json, TRANSCRIPTS  # noqa: E402
from model import make_units  # noqa: E402
from prompts import VARIANTS  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402
from local_evaluator import Statistics  # noqa: E402


def conversations(asr: str):
    for fn, rows in group_questions_by_conversation():
        stem = Path(fn).stem
        tf = TRANSCRIPTS / f'{stem}.{asr}.json'
        if not tf.exists():
            continue
        words, duration, _ = load_words(tf)
        yield stem, rows, words, duration, make_units(words)


def dump(a):
    v = VARIANTS[a.variant]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, rows, words, duration, units in conversations(a.asr):
        if hasattr(v, 'set_conversation'):
            v.set_conversation(stem, a.asr)
        p = v.build_all([r['question'] for r in rows], units)
        msgs = [{'role': 'system', 'content': p.system}]
        for du, da in (p.demos or []):
            msgs += [{'role': 'user', 'content': du}, {'role': 'assistant', 'content': da}]
        msgs.append({'role': 'user', 'content': p.user})
        (out / f'{stem}.json').write_text(json.dumps({'stem': stem, 'messages': msgs, 'schema': p.schema}, ensure_ascii=False, indent=1), encoding='utf-8')
        # a flat text rendering for agents that read files
        txt = [f'### SYSTEM\n{p.system}']
        for du, da in (p.demos or []):
            txt.append(f'### EXAMPLE INPUT\n{du}\n### EXAMPLE OUTPUT\n{da}')
        txt.append(f'### INPUT\n{p.user}')
        (out / f'{stem}.txt').write_text('\n\n'.join(txt), encoding='utf-8')
        n += 1
    print(f'{n} prompts -> {out}')


def score(a):
    v = VARIANTS[a.variant]
    stats = Statistics(); stats.nulls_on_no = Statistics()
    records = []
    missing = 0
    for stem, rows, words, duration, units in conversations(a.asr):
        f = Path(a.answers) / f'{stem}.json'
        if not f.exists():
            missing += 1; continue
        if hasattr(v, 'set_conversation'):
            v.set_conversation(stem, a.asr)
        p = v.build_all([r['question'] for r in rows], units)
        try:
            out = parse_json(f.read_text(encoding='utf-8'))
            per = v.split(out, len(rows))
        except Exception as exc:
            per = [{'quote': '', 'answer': 'no', 'segments': []}] * len(rows)
            print(f'{stem}: unparseable answers ({exc})')
        for row, item in zip(rows, per):
            gold = gold_evidence(row)
            try:
                yes, span = p.postprocess(item, units, words, duration)
            except Exception:
                yes, span = False, None
            pred = int(bool(yes)); span_t = tuple(span) if span else None
            stats.record(row['question_type'], int(row['label']), pred, gold, span_t)
            stats.nulls_on_no.record(row['question_type'], int(row['label']), pred, gold, span_t if pred == 1 else None)
            records.append({'question_id': row['question_id'], 'transcript_id': row['transcript_id'], 'question': row['question'],
                            'question_type': row['question_type'], 'label': int(row['label']), 'raw': item, 'answer': bool(yes),
                            'span': list(span) if span else None, 'gold': list(gold) if gold else None, 'error': None,
                            'prediction': pred, 'correct': pred == int(row['label']),
                            'tiou': temporal_iou(gold, span_t) if (gold and span_t) else (0.0 if gold else None), 'latency_ms': 0, 'usage': None})
    n = len(records)
    acc = sum(r['correct'] for r in records) / n if n else 0
    res = {'config': {'model': a.model, 'variant': a.variant, 'asr': a.asr, 'source': 'dump_prompts.py score'},
           'summary': {'partial': False, 'questions': n, 'accuracy': acc, 'mean_tiou': stats.mean_tiou,
                       'nulls_on_no': {'mean_tiou': stats.nulls_on_no.mean_tiou, 'score': stats.nulls_on_no.final_score}},
           'questions': records}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'{a.model}: {n} questions ({missing} conversations without answers), accuracy {acc:.3f}, '
          f'tIoU nulls-on-no {stats.nulls_on_no.mean_tiou:.3f}, score {stats.nulls_on_no.final_score:.3f} -> {a.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['dump', 'score'])
    ap.add_argument('--variant', default='units-joint-demo')
    ap.add_argument('--asr', default='large-v3-turbo')
    ap.add_argument('--out', required=True)
    ap.add_argument('--answers', default='')
    ap.add_argument('--model', default='claude')
    a = ap.parse_args()
    (dump if a.cmd == 'dump' else score)(a)


if __name__ == '__main__':
    main()
