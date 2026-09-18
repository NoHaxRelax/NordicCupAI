"""Build bench/ref/timeline.html: per conversation, the gold evidence spans and the
pipeline's spans on one time axis, over the turbo sentence units.

Training set: gold from data/question_train.csv; one selectable set per bench result
under bench/results/llm (every complete 390-question run whose transcripts exist
locally), re-scored from its raw answers with the right edge offsets by
bench/llm/replay.py, labelled "model · prompt · transcripts · notes" with its
spans-on-no score. The menu is sorted by score; the best run opens first.
Validation set: gold from bench/mine/span_state.json (recovered spans), predictions
from request_dump/answers.jsonl (the last real validation run), questions and
answers from bench/mine/agent_answers.md.

    python bench/timeline.py                                  # every run
    python bench/timeline.py bench/results/llm/qwen3.8-27b.*.json   # a subset

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
sys.path.insert(0, str(HERE / 'llm'))
import answers_md  # noqa: E402
import replay  # noqa: E402  bench/llm/replay.py: re-scores stored runs with the right offsets
from utils import group_questions_by_conversation  # noqa: E402

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


VARIANT_NOTE = {
    'units': 'one question per request', 'units-claim': 'claim rewrite, one per request',
    'units-nooffset': 'one per request, no edge offsets', 'units-pyes': 'one per request, yes-probability',
    'units-v2': 'one per request, v2 prompt', 'words': 'word-level phrases', 'words-fewshot': 'word-level + few-shot lines',
    'units-fewshot': 'few-shot lines (12 nearest examples)', 'units-joint': 'joint: 10 questions per request',
    'units-joint-demo': 'joint + 2 worked conversations', 'units-joint-demo-fewshot': 'joint + 2 worked + few-shot lines',
    'units-joint-demo-all': 'joint + all 38 other worked conversations',
    'units-joint-demo-x1': "joint + 2 worked + the 'confirmed or acted upon' line",
}


def run_label(path: Path, cfg: dict) -> tuple[str, str]:
    """(short model name, notes) for the menu."""
    model = (cfg.get('model') or '?').split('/')[-1]
    notes = [VARIANT_NOTE.get(cfg.get('variant'), '')]
    if (cfg.get('unit_split') or 'sentence') != 'sentence':
        notes.append(f"units cut: {cfg['unit_split']}")
    if cfg.get('source', '').startswith('dump_prompts'):
        if cfg.get('variant') == 'units-joint-demo-all' or '.clean' in path.name:
            notes.append('1 agent per conversation')
        else:
            notes.append('5 prompts per agent (batched)')
        notes.append('Claude probe, not servable')
    else:
        host = cfg.get('url', '')
        notes.append('Ollama laptop' if '11434' in host else 'vLLM')
        if cfg.get('workers'):
            notes.append(f"workers {cfg['workers']}")
        if cfg.get('max_tokens'):
            notes.append(f"max_tokens {cfg['max_tokens']}")
        if cfg.get('date'):
            notes.append(cfg['date'][:10])
    return model, ', '.join(n for n in notes if n)


def runs(files: list[Path]) -> tuple[list[dict], dict, dict]:
    """One set per result file, spans re-scored by replay.py; transcripts and the
    set-independent question fields are shared through TRANS / QBASE keyed by
    "stem|asr" so fifty runs do not embed the transcripts fifty times."""
    gold_rows = list(csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')))
    order: dict[str, list[str]] = {}
    for r in gold_rows:
        order.setdefault(r['transcript_id'], []).append(r['question_id'])
    by_tid = {}
    for fn, rows in group_questions_by_conversation():
        by_tid[rows[0]['transcript_id']] = (Path(fn).stem, rows)
    trans: dict[str, dict] = {}
    qbase: dict[str, list[dict]] = {}
    words_cache: dict[str, list[dict]] = {}
    sets = []
    for f in files:
        try:
            r = replay.replay_file(f, None, by_tid)
        except Exception as exc:
            print(f'  skip {f.name}: {type(exc).__name__}: {exc}', file=sys.stderr)
            continue
        if r['partial'] or r['questions'] != 390:
            print(f'  skip {f.name}: partial or {r["questions"]} questions', file=sys.stderr)
            continue
        res = json.loads(f.read_text(encoding='utf-8'))
        raw_by = {q['question_id']: q for q in res['questions']}
        asr = r['asr']
        by: dict[str, list[dict]] = {}
        for q in r['per_question']:
            by.setdefault(q['transcript_id'], []).append(q)
        convs = []
        for tid in sorted(by, key=lambda t: int(re.sub(r'\D', '', t))):
            stem = f'conversation_{tid}'
            tk = f'{stem}|{asr}'
            if tk not in trans:
                tf = CASE / 'transcripts' / f'{stem}.{asr}.json'
                segs, dur, words = segments(tf) if tf.exists() else ([], 0.0, [])
                trans[tk] = {'segs': segs, 'dur': dur}
                words_cache[tk] = words
            words = words_cache[tk]
            pos = {qid: i for i, qid in enumerate(order.get(tid, []))}
            ordered = sorted(by[tid], key=lambda x: pos.get(x['question_id'], 99))
            if tk not in qbase:
                base = []
                for q in ordered:
                    raw = raw_by[q['question_id']]
                    b = {'i': pos.get(q['question_id'], 0) + 1, 'text': raw['question'], 'type': raw['question_type'],
                         'gold': q.get('gold'), 'yes': bool(q['label'])}
                    b['gtext'] = text_in(words, b['gold'])
                    base.append(b)
                qbase[tk] = base
            qs = []
            for q in ordered:
                raw = raw_by[q['question_id']]
                pred = None if q['prediction'] < 0 else bool(q['prediction'])
                item = {'pred': pred, 'span': q.get('span'), 'quote': (raw.get('raw') or {}).get('quote')}
                item['ptext'] = text_in(words, q['span']) if pred and q.get('span') else ''
                qs.append(item)
            convs.append({'stem': stem, 'tk': tk, 'qs': qs})
        cfg = res.get('config', {})
        model, notes = run_label(f, cfg)
        name = f"{model} · {cfg.get('variant')} · {asr}" + (f' · {notes}' if notes else '')
        sets.append({'name': name, 'score': r['spans']['score'], 'acc': r['accuracy'], 'tiou': r['spans']['mean_tiou'],
                     'source': f"{name} — score {r['spans']['score']:.4f} (accuracy {r['accuracy']:.3f}, mean tIoU {r['spans']['mean_tiou']:.3f}, "
                               f"spans on every question, offsets {r['offsets'][0]:+.2f}/{r['offsets'][1]:+.2f}) · {f.name}",
                     'convs': convs})
    sets.sort(key=lambda x: -x['score'])
    return sets, trans, qbase


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
    ap.add_argument('files', nargs='*', help='result JSONs (default: every complete run under bench/results/llm)')
    a = ap.parse_args()
    files = [Path(f) for f in a.files] if a.files else sorted(p for p in (HERE / 'results' / 'llm').glob('*.json')
                                                              if not p.name.endswith('.partial.json'))
    sets, trans, qbase = runs(files)
    if not sets:
        raise SystemExit('no usable bench result found')
    v = validation()
    if v:
        sets.append(v)
    clips_file = HERE / 'ref' / 'clips.json'
    clips = clips_file.read_text(encoding='utf-8') if clips_file.exists() else '{}'
    html = (TEMPLATE.read_text(encoding='utf-8').replace('/*DATA*/null', json.dumps(sets, ensure_ascii=False))
            .replace('/*TRANS*/null', json.dumps(trans, ensure_ascii=False))
            .replace('/*QBASE*/null', json.dumps(qbase, ensure_ascii=False))
            .replace('/*CLIPS*/null', clips))
    OUT.write_text(html, encoding='utf-8')
    print(f'{len(sets) - (1 if v else 0)} training runs' + (f', {len(v["convs"])} validation conversations' if v else '')
          + f', clips {"embedded" if clips != "{}" else "none (run bench/clips.py)"} -> {OUT} ({OUT.stat().st_size // 1024} KB)')
    for st in sets:
        if st.get('score') is not None:
            print(f"  {st['score']:.4f}  {st['name']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
