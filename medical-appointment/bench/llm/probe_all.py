"""Many-shot leave-one-out probe over ALL 58 labelled conversations: 39 training + 19 validation.

For every conversation the prompt holds the other 57 as worked examples (transcript, ten questions,
the annotators' answers with unit ids) and asks the ten questions of the held-out one; the held-out
conversation never appears among its own demos (checked at dump time). Answers are written by agents
driven through the Claude Code Workflow tool (a hosted model cannot be pointed at the bench's HTTP
client and can never be served), one tool-less agent per conversation so nothing leaks between them.

    ASR_MODEL=large-v3-turbo UNIT_SPLIT=clause-and python bench/llm/probe_all.py dump \\
        --variant units-joint-demo-all-both-val --out bench/results/probe/prompts-fable-all
    ... one agent per conversation writes bench/results/probe/fable-all/<stem>.json (JSON only) ...
    ASR_MODEL=large-v3-turbo UNIT_SPLIT=clause-and python bench/llm/probe_all.py score \\
        --variant units-joint-demo-all-both-val --answers bench/results/probe/fable-all \\
        --model claude-fable-5.1 --out bench/results/llm/claude-fable-5.1.units-joint-demo-all-both-val.large-v3-turbo.clause-and.json

Scoring: training conversations against data/question_train.csv, validation conversations against
the hand labels (bench/mine/agent_answers.md binaries, bench/mine/span_state.json recovered spans,
which reproduce the portal scores to four decimals, entry 65). Spans-on-every-question policy, the
served offsets, the same tIoU as local_evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
for p in (str(HERE), str(CASE), str(CASE / 'bench' / 'mine')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('ASR_MODEL', 'large-v3-turbo')
os.environ.setdefault('UNIT_SPLIT', 'clause-and')
import model  # noqa: E402
from model import make_units, render_transcript  # noqa: E402
from prompts import VARIANTS  # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402
import answers_md  # noqa: E402

DUMP = CASE / 'request_dump'
TRANSCRIPTS = CASE / 'transcripts'


def load_words(path: Path):
    d = json.loads(path.read_text(encoding='utf-8'))
    words = []
    for s in d['segments']:
        for w in s.get('words', []):
            words.append(model.Word(w['w'], float(w['start']), float(w['end'])))
        if s.get('words'):
            words[-1].w += '\x00'
    return words, float(d['duration'])


def conversations(asr: str):
    """(stem, set, rows, words, duration, units); rows carry question, label, gold, question_id."""
    for fn, rows in group_questions_by_conversation():
        stem = Path(fn).stem
        tf = TRANSCRIPTS / f'{stem}.{asr}.json'
        if not tf.exists():
            continue
        words, duration = load_words(tf)
        out = [{'question_id': r['question_id'], 'transcript_id': r['transcript_id'], 'question': r['question'],
                'question_type': r['question_type'], 'label': int(r['label']), 'gold': gold_evidence(r)} for r in rows]
        yield stem, 'train', out, words, duration, make_units(words)
    hand = answers_md.parse()
    state = json.loads((CASE / 'bench' / 'mine' / 'span_state.json').read_text(encoding='utf-8'))
    for stem in sorted(hand, key=lambda s: int(s.split('_')[-1])):
        tf, qf = DUMP / 'transcripts' / f'{stem}.{asr}.json', DUMP / f'{stem}.questions.json'
        if not (tf.exists() and qf.exists()):
            continue
        words, duration = load_words(tf)
        qs = json.loads(qf.read_text(encoding='utf-8'))
        out = []
        for r in hand[stem]:
            st = state.get(f"{stem}:{r['q']}")
            gold = (float(st['g']), float(st['h'])) if (r['answer'] and st and st.get('stage') == 'done') else None
            out.append({'question_id': f"{stem}_q{r['q']}", 'transcript_id': stem.replace('conversation_', ''),
                        'question': qs[r['q'] - 1].strip(), 'question_type': 'validation', 'label': int(r['answer']), 'gold': gold})
        yield stem, 'validation', out, words, duration, make_units(words)


def dump(a):
    v = VARIANTS[a.variant]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest, leaks, chars = [], 0, []
    for stem, subset, rows, words, duration, units in conversations(a.asr):
        v.set_conversation(stem, a.asr)
        p = v.build_all([r['question'] for r in rows], units)
        own = render_transcript(units)
        if any(own in du or stem in du for du, _ in (p.demos or [])):
            leaks += 1
            print(f'LEAK: {stem} appears among its own demos')
        msgs = [{'role': 'system', 'content': p.system}]
        for du, da in (p.demos or []):
            msgs += [{'role': 'user', 'content': du}, {'role': 'assistant', 'content': da}]
        msgs.append({'role': 'user', 'content': p.user})
        (out / f'{stem}.json').write_text(json.dumps({'stem': stem, 'set': subset, 'messages': msgs, 'schema': p.schema},
                                                     ensure_ascii=False, indent=1), encoding='utf-8')
        txt = [f'### SYSTEM\n{p.system}']
        for du, da in (p.demos or []):
            txt.append(f'### EXAMPLE INPUT\n{du}\n### EXAMPLE OUTPUT\n{da}')
        txt.append(f'### INPUT\n{p.user}')
        text = '\n\n'.join(txt)
        (out / f'{stem}.txt').write_text(text, encoding='utf-8')
        chars.append(len(text))
        manifest.append({'stem': stem, 'set': subset, 'demos': len(p.demos or []), 'chars': len(text), 'lines': text.count('\n') + 1})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'{len(manifest)} prompts -> {out}; demos per prompt {min(m["demos"] for m in manifest)} to {max(m["demos"] for m in manifest)}; '
          f'{statistics.mean(chars)/1000:.0f}k chars mean ({max(chars)/1000:.0f}k max, about {statistics.mean(chars)/4/1000:.0f}k tokens); '
          f'leaks {leaks}')


def score(a):
    v = VARIANTS[a.variant]
    records, missing = [], []
    for stem, subset, rows, words, duration, units in conversations(a.asr):
        f = Path(a.answers) / f'{stem}.json'
        if not f.exists():
            missing.append(stem); continue
        v.set_conversation(stem, a.asr)
        p = v.build_all([r['question'] for r in rows], units)
        try:
            out = json.loads(f.read_text(encoding='utf-8'))
            per = v.split(out, len(rows))
        except Exception as exc:
            per = [{'quote': '', 'answer': 'no', 'segments': []}] * len(rows)
            print(f'{stem}: unparseable answers ({exc})')
        for row, item in zip(rows, per):
            try:
                yes, span = p.postprocess(item, units, words, duration)
            except Exception:
                yes, span = False, None
            pred = int(bool(yes)); span_t = tuple(span) if span else None; gold = row['gold']
            records.append({**{k: row[k] for k in ('question_id', 'transcript_id', 'question', 'question_type', 'label')},
                            'set': subset, 'raw': item, 'answer': bool(yes), 'span': list(span) if span else None,
                            'gold': list(gold) if gold else None, 'prediction': pred, 'correct': pred == row['label'],
                            'tiou': (temporal_iou(gold, span_t) if span_t else 0.0) if gold else None})
    summary = {}
    for name, sel in (('train', lambda r: r['set'] == 'train'), ('validation', lambda r: r['set'] == 'validation'), ('all', lambda r: True)):
        rs = [r for r in records if sel(r)]
        if not rs:
            continue
        acc = sum(r['correct'] for r in rs) / len(rs)
        ious = [r['tiou'] for r in rs if r['gold']]
        t = statistics.mean(ious) if ious else 0.0
        summary[name] = {'questions': len(rs), 'accuracy': acc, 'wrong': sum(not r['correct'] for r in rs),
                         'positives': len(ious), 'mean_tiou': t, 'score': 0.4 * acc + 0.6 * t}
        print(f"{name:<10} questions {len(rs):>3}  accuracy {acc:.4f} ({sum(not r['correct'] for r in rs)} wrong)  "
              f"positives {len(ious):>3}  mean tIoU {t:.4f}  score {0.4*acc+0.6*t:.4f}")
    if missing:
        print(f'{len(missing)} conversations without answers: {missing}')
    res = {'config': {'model': a.model, 'variant': a.variant, 'asr': a.asr, 'unit_split': model.UNIT_SPLIT,
                      'start_offset': model.START_OFFSET, 'end_offset': model.END_OFFSET, 'source': 'probe_all.py score'},
           'summary': {'partial': bool(missing), **summary}, 'questions': records}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'-> {a.out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['dump', 'score'])
    ap.add_argument('--variant', default='units-joint-demo-all-both-val')
    ap.add_argument('--asr', default='large-v3-turbo')
    ap.add_argument('--out', required=True)
    ap.add_argument('--answers', default='')
    ap.add_argument('--model', default='claude')
    a = ap.parse_args()
    if model.ASR_MODEL != a.asr:
        raise SystemExit(f'model.ASR_MODEL={model.ASR_MODEL} but --asr {a.asr}: set ASR_MODEL before running')
    (dump if a.cmd == 'dump' else score)(a)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
