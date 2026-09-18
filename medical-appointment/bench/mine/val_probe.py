"""Offline probe of any answering model on the VALIDATION conversations.

Measurement only. The validation labels (bench/mine/agent_answers.md binaries and
span_state.json recovered spans) are a held-out yardstick and are never used to tune the
pipeline; this script exists to answer "what would a different model score on the same
served setup", which the portal cannot answer for a model we are not allowed to serve
(a hosted API may not sit in the request path).

    python bench/mine/val_probe.py dump  --variant units-fewshot --out bench/results/probe/val-sonnet
    ... an agent writes <stem>.json = {"<question_id>": {"quote":..,"answer":"yes|no","segments":[..]}, ...}
    python bench/mine/val_probe.py score --variant units-fewshot --out bench/results/probe/val-sonnet

The prompts are built exactly as the endpoint builds them: same units (UNIT_SPLIT), same
variant from bench/llm/prompts.py, same few-shot pool. No leave-one-out is needed because
the pool holds only training conversations, which is also the real serving condition.
Set ASR_MODEL=large-v3-turbo (and UNIT_SPLIT) before running, as the bench does.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
for p in (str(CASE), str(CASE / 'bench' / 'llm'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model  # noqa: E402
from prompts import VARIANTS  # noqa: E402
import answers_md  # noqa: E402

DUMP = CASE / 'request_dump'


def conversations(asr: str):
    """(stem, questions, words, duration, units) for each validation conversation."""
    for qf in sorted(DUMP.glob('*.questions.json'), key=lambda p: int(re.sub(r'\D', '', p.stem) or 0)):
        stem = qf.name.split('.questions.json')[0]
        tf = DUMP / 'transcripts' / f'{stem}.{asr}.json'
        if not tf.exists():
            print(f'  skip {stem}: no {tf.name}', file=sys.stderr)
            continue
        d = json.loads(tf.read_text(encoding='utf-8'))
        words = []
        for s in d['segments']:
            for w in s.get('words', []):
                words.append(model.Word(w['w'], float(w['start']), float(w['end'])))
            if s.get('words'):
                words[-1].w += '\x00'
        yield (stem, json.loads(qf.read_text(encoding='utf-8')), words, float(d['duration']),
               model.make_units(words))


def dump(a) -> int:
    out = Path(a.out); (out / 'prompts').mkdir(parents=True, exist_ok=True)
    variant = VARIANTS[a.variant]
    n = 0
    for stem, questions, words, duration, units in conversations(a.asr):
        if hasattr(variant, 'set_conversation'):
            variant.set_conversation(stem, a.asr)      # stem is not in the pool: every training example is usable
        for i, q in enumerate(questions, 1):
            p = variant(q, units)
            (out / 'prompts' / f'{stem}.q{i}.txt').write_text(
                '--- SYSTEM ---\n' + p.system + '\n--- USER ---\n' + p.user, encoding='utf-8')
            n += 1
    print(f'{n} prompts for {len(list(DUMP.glob("*.questions.json")))} conversations -> {out / "prompts"}')
    print(f'units {model.UNIT_SPLIT}, variant {a.variant}, asr {a.asr}, offsets '
          f'{model.START_OFFSET:+.2f}/{model.END_OFFSET:+.2f}')
    return 0


def score(a) -> int:
    out = Path(a.out)
    variant = VARIANTS[a.variant]
    state = json.loads((HERE / 'span_state.json').read_text(encoding='utf-8'))
    hand = answers_md.parse()                       # {stem: [{q, answer, start, end}, ...]}
    n = correct = 0
    ious, missing = [], 0
    for stem, questions, words, duration, units in conversations(a.asr):
        f = out / f'{stem}.json'
        got = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
        rows = {r['q']: r for r in hand.get(stem, [])}
        for i, q in enumerate(questions, 1):
            if i not in rows:
                continue
            n += 1
            item = got.get(f'q{i}') or got.get(str(i)) or {}
            if not item:
                missing += 1
            p = variant(q, units)
            try:
                yes, span = p.postprocess(item, units, words, duration)
            except Exception:
                yes, span = True, None
            correct += (bool(yes) == bool(rows[i]['answer']))
            if rows[i]['answer']:                   # only gold-yes questions carry evidence
                st = state.get(f'{stem}:{i}')
                gold = (st['g'], st['h']) if st and st.get('stage') == 'done' else None
                if gold:
                    if span:
                        inter = max(0.0, min(gold[1], span[1]) - max(gold[0], span[0]))
                        union = max(gold[1], span[1]) - min(gold[0], span[0])
                        ious.append(inter / union if union > 0 else 0.0)
                    else:
                        ious.append(0.0)
    acc = correct / max(1, n)
    tiou = statistics.mean(ious) if ious else 0.0
    print(f'{a.model}  variant {a.variant}  units {model.UNIT_SPLIT}')
    print(f'  questions {n}  accuracy {acc:.4f} ({correct}/{n})  positives {len(ious)}  mean tIoU {tiou:.4f}')
    print(f'  SCORE {0.4 * acc + 0.6 * tiou:.4f}' + (f'   ({missing} answers missing)' if missing else ''))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['dump', 'score'])
    ap.add_argument('--variant', default='units-fewshot')
    ap.add_argument('--asr', default='large-v3-turbo')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='probe')
    a = ap.parse_args()
    if model.ASR_MODEL != a.asr:
        raise SystemExit(f'model.ASR_MODEL={model.ASR_MODEL} but --asr {a.asr}: set ASR_MODEL before running')
    return dump(a) if a.cmd == 'dump' else score(a)


if __name__ == '__main__':
    raise SystemExit(main())
