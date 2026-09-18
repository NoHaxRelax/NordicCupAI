"""Build bench/ref/timeline.html: per conversation, the gold evidence spans and the
pipeline's spans on one time axis, over the turbo sentence units.

Training set: gold from data/question_train.csv, predictions from a bench result
(default: the newest qwen3-4b units run on turbo transcripts in bench/results/llm).
Validation set: gold from bench/mine/span_state.json (recovered spans), predictions
from request_dump/answers.jsonl (the last real validation run), questions and
answers from bench/mine/agent_answers.md.

    python bench/timeline.py                                  # newest result
    python bench/timeline.py --result bench/results/llm/qwen3-4b.units.large-v3-turbo.json

Open bench/ref/timeline.html directly, or http://localhost:9060/timeline.html under the
oracle server, where clicking a bar plays that stretch of audio.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(HERE / 'mine'))
import answers_md  # noqa: E402

TEMPLATE = HERE / 'ref' / 'timeline_template.html'
OUT = HERE / 'ref' / 'timeline.html'


def segments(path: Path) -> tuple[list[dict], float, list[dict]]:
    d = json.loads(path.read_text(encoding='utf-8'))
    segs = [{'s': round(s['start'], 2), 'e': round(s['end'], 2), 't': s['text'].strip()} for s in d['segments'] if s['text'].strip()]
    words = [{'s': w['start'], 'e': w['end'], 'w': w['w'].strip()} for s in d['segments'] for w in (s.get('words') or [])]
    return segs, (segs[-1]['e'] if segs else 0.0), words


def text_in(words: list[dict], span) -> str:
    """The turbo words that overlap the interval by at least a third of their own
    duration. (A midpoint rule dropped the first word of most gold spans: gold
    starts sit a median 0.29 s after Whisper's first word start on the training
    set, so that word's midpoint falls just before the gold begins.)"""
    if not span or not words:
        return ''
    a, b = span
    out = []
    for w in words:
        d = max(1e-3, w['e'] - w['s'])
        if min(b, w['e']) - max(a, w['s']) >= d / 3:
            out.append(w['w'])
    return ' '.join(out)


def add_texts(qs: list[dict], words: list[dict]) -> None:
    for q in qs:
        q['gtext'] = text_in(words, q.get('gold'))
        q['ptext'] = text_in(words, q.get('span')) if q.get('pred') and q.get('span') else ''


def training(result: Path) -> dict:
    res = json.loads(result.read_text(encoding='utf-8'))
    by = {}
    for q in res['questions']:
        by.setdefault(q['transcript_id'], []).append(q)
    gold_rows = list(csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')))
    order = {}
    for r in gold_rows:
        order.setdefault(r['transcript_id'], []).append(r['question_id'])
    convs = []
    for tid in sorted(by, key=lambda t: int(re.sub(r'\D', '', t))):
        stem = f'conversation_{tid}'
        tf = CASE / 'transcripts' / f'{stem}.large-v3-turbo.json'
        segs, dur, words = segments(tf) if tf.exists() else ([], 0.0, [])
        qs = []
        pos = {qid: i for i, qid in enumerate(order.get(tid, []))}
        for q in sorted(by[tid], key=lambda x: pos.get(x['question_id'], 99)):
            qs.append({'i': pos.get(q['question_id'], 0) + 1, 'text': q['question'], 'type': q['question_type'],
                       'gold': q.get('gold'), 'yes': bool(q['label']), 'pred': bool(q['answer']) if q.get('answer') is not None else None,
                       'span': q.get('span'), 'quote': (q.get('raw') or {}).get('quote')})
        add_texts(qs, words)
        convs.append({'stem': stem, 'dur': dur, 'segs': segs, 'qs': qs})
    cfg = res.get('config', {})
    return {'name': 'training', 'source': f"{cfg.get('model')} {cfg.get('variant')} on {cfg.get('asr')} ({cfg.get('date', '')[:16]}), {result.name}",
            'convs': convs}


def validation() -> dict | None:
    dump = CASE / 'request_dump'
    if not (dump / 'answers.jsonl').exists():
        return None
    state = json.loads((HERE / 'mine' / 'span_state.json').read_text(encoding='utf-8')) if (HERE / 'mine' / 'span_state.json').exists() else {}
    rows = answers_md.parse()
    served = {}
    for line in (dump / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
        d = json.loads(line); served[Path(d['file']).stem] = d
    convs = []
    for stem in sorted(rows, key=lambda s: int(re.sub(r'\D', '', s))):
        tf = next((dump / 'transcripts').glob(f'{stem}.*.json'), None)
        segs, dur, words = segments(tf) if tf else ([], 0.0, [])
        d = served.get(stem, {})
        qs = []
        for r in rows[stem]:
            i = r['q'] - 1
            st = state.get(f"{stem}:{r['q']}")
            gold = [st['g'], st['h']] if st and st.get('stage') == 'done' else None
            qs.append({'i': r['q'], 'text': d['questions'][i] if d else '', 'type': 'positive' if r['answer'] else 'negative',
                       'gold': gold, 'yes': r['answer'], 'pred': bool(d['answers'][i]) if d else None,
                       'span': (d.get('spans') or [None] * 10)[i] if d else None, 'quote': None,
                       'seed': [r['start'], r['end']] if r['answer'] else None})
        add_texts(qs, words)
        convs.append({'stem': stem, 'dur': dur, 'segs': segs, 'qs': qs})
    n_gold = sum(1 for c in convs for q in c['qs'] if q['gold'])
    return {'name': 'validation', 'source': f'served run F (0.6759) vs recovered gold spans ({n_gold} of 95 recovered so far)', 'convs': convs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', help='bench result json with per-question spans')
    a = ap.parse_args()
    if a.result:
        result = Path(a.result)
    else:
        cands = sorted((HERE / 'results' / 'llm').glob('qwen3-4b.units*.large-v3-turbo.json'), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit('no bench result found; pass --result')
        result = cands[-1]
    sets = [training(result)]
    v = validation()
    if v:
        sets.append(v)
    clips_file = HERE / 'ref' / 'clips.json'
    clips = clips_file.read_text(encoding='utf-8') if clips_file.exists() else '{}'
    html = (TEMPLATE.read_text(encoding='utf-8').replace('/*DATA*/null', json.dumps(sets, ensure_ascii=False))
            .replace('/*CLIPS*/null', clips))
    OUT.write_text(html, encoding='utf-8')
    print(f'{result.name}: {len(sets[0]["convs"])} training conversations' + (f', {len(v["convs"])} validation' if v else '')
          + f', clips {"embedded" if clips != "{}" else "none (run bench/clips.py)"} -> {OUT} ({OUT.stat().st_size // 1024} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
