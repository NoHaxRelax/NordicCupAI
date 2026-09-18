"""How much of the sentence-to-word oracle gap do clause-level units close?

Offline, training set only, no network and no LLM. For every annotated training
span (`utils.gold_evidence` over `data/question_train.csv`) this searches the
best contiguous run of up to `--max-units` units, scored through the served
`model.span_from_ids` (START_RULE=first-word-end and the turbo edge offsets),
and reports that unit-level oracle for the three `UNIT_SPLIT` modes of
`model.make_units`:

    sentence    terminal punctuation, segment ends, pauses (what serves today)
    clause      + a cut at a clause comma (a conjunction follows, or a
                subject-verb clause of at least four words), pieces >= 3 words
    clause-all  + a cut at every comma or semicolon with three words on each side

Context (research/07-findings-log.md 40, 44, 45): the annotators mark a clause
inside one of our sentence units for about 15 % of the golds, which costs the
27B few-shot run 37 spans and +0.048 of score. The sentence oracle is 0.861 and
the word-level oracle 0.946; this measures where the clause modes land between
them, and what the finer cut costs in units per transcript.

The oracle is an upper bound for a perfect selector over these units, not a
prediction: more units also means more ways for the model to pick the wrong one.

    PYTHONIOENCODING=utf-8 python bench/units/clause_oracle.py
    python bench/units/clause_oracle.py --asr large-v3 --max-units 4
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent            # bench/units
CASE = HERE.parent.parent                         # medical-appointment
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))


def _argv_opt(name: str, default: str) -> str:
    args = sys.argv[1:]
    for i, s in enumerate(args):
        if s == f'--{name}' and i + 1 < len(args):
            return args[i + 1]
        if s.startswith(f'--{name}='):
            return s.split('=', 1)[1]
    return default


# model.py keys its fitted edge offsets on ASR_MODEL at import time, so this has
# to happen before the import (bench/llm/bench.py, findings log entry 38).
ASR = _argv_opt('asr', 'large-v3-turbo')
if os.environ.get('ASR_MODEL', ASR) != ASR:
    raise SystemExit(f'ASR_MODEL={os.environ["ASR_MODEL"]} contradicts --asr {ASR}: unset one of them')
os.environ['ASR_MODEL'] = ASR
os.environ.setdefault('START_RULE', 'first-word-end')
os.environ.pop('UNIT_SPLIT', None)                # every mode is passed explicitly

import model                                      # noqa: E402  serving code
from model import Unit, Word, make_units, span_from_ids               # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402

TRANSCRIPTS = CASE / 'transcripts'
MODES = ['sentence', 'clause', 'clause-all', 'clause-and']
MAX_UNITS = int(_argv_opt('max-units', '6'))
WORST = int(_argv_opt('worst', '15'))
GAP = float(_argv_opt('gap', '0.7'))
ACCURACY_WEIGHT, TIOU_WEIGHT = 0.4, 0.6           # local_evaluator.py
Span = Tuple[float, float]


def load_words(path: Path) -> Tuple[List[Word], float]:
    """Word stream + duration from a cached transcript, exactly as
    bench/llm/bench.py and model._cached_transcript build it (segment ends
    marked with \\x00)."""
    d = json.loads(path.read_text(encoding='utf-8'))
    words: List[Word] = []
    for s in d['segments']:
        for w in s.get('words', []):
            words.append(Word(w['w'], float(w['start']), float(w['end'])))
        if s.get('words'):
            words[-1].w += '\x00'
    return words, float(d['duration'])


def runs(units: List[Unit], duration: float) -> List[Tuple[int, int, Span]]:
    """Every contiguous run of 1..MAX_UNITS units, as the served span."""
    out = []
    for i in range(len(units)):
        for j in range(i, min(len(units), i + MAX_UNITS)):
            sp = span_from_ids(list(range(i, j + 1)), units, duration)
            if sp is not None:
                out.append((i, j, sp))
    return out


def oracle(candidates, gold: Span) -> Tuple[float, Optional[Tuple[int, int]]]:
    best, best_run = 0.0, None
    for i, j, sp in candidates:
        v = temporal_iou(gold, sp)
        if v > best:
            best, best_run = v, (i, j)
    return best, best_run


def words_in(words: List[Word], gold: Span, frac: float = 1.0 / 3.0) -> str:
    """The transcript inside the gold interval: every word overlapping it by at
    least a third of its own duration (the rule the timeline page uses)."""
    keep = []
    for w in words:
        dur = max(1e-6, w.end - w.start)
        ov = max(0.0, min(gold[1], w.end) - max(gold[0], w.start))
        if ov / dur >= frac:
            keep.append(w.w.replace('\x00', ''))
    return ''.join(keep).strip()


def main() -> int:
    print(f'asr {ASR}; start rule {model.START_RULE}; offsets start {model.START_OFFSET:+.2f} '
          f'end {model.END_OFFSET:+.2f}; runs of up to {MAX_UNITS} units')

    per_mode: Dict[str, List[float]] = {m: [] for m in MODES}
    per_gold: Dict[str, Dict[str, float]] = {m: {} for m in MODES}
    n_units: Dict[str, List[int]] = {m: [] for m in MODES}
    unit_words: Dict[str, List[int]] = {m: [] for m in MODES}
    detail: List[dict] = []                       # one row per gold, clause mode
    missing: List[str] = []

    for fn, rows in group_questions_by_conversation():
        stem = Path(fn).stem
        path = TRANSCRIPTS / f'{stem}.{ASR}.json'
        if not path.exists():
            missing.append(stem)
            continue
        words, duration = load_words(path)
        golds = [(r, gold_evidence(r)) for r in rows]
        golds = [(r, g) for r, g in golds if g is not None]
        for mode in MODES:
            units = make_units(words, mode=mode)
            n_units[mode].append(len(units))
            unit_words[mode].extend(len(u.text.split()) for u in units)
            cands = runs(units, duration)
            for r, g in golds:
                v, run = oracle(cands, g)
                per_mode[mode].append(v)
                per_gold[mode][r['question_id']] = v
                if mode == 'clause':
                    detail.append({
                        'qid': r['question_id'], 'stem': stem, 'question': r['question'],
                        'gold': g, 'tiou': v, 'run': run,
                        'gold_text': words_in(words, g),
                        'units': [u.text for u in units[run[0]:run[1] + 1]] if run else [],
                        'near': [u.text for u in units
                                 if min(g[1], u.end) - max(g[0], u.start) > 0.05],
                    })

    n = len(per_mode['sentence'])
    if missing:
        print(f'({len(missing)} conversations without a {ASR} transcript, skipped: {", ".join(missing)})')
    print(f'{n} annotated training golds over {len(n_units["sentence"])} conversations\n')

    base = per_gold['sentence']
    print('| unit mode | units/conv | words/unit | mean oracle tIoU | oracle >= 0.9 | '
          'score ceiling | gained >= 0.1 | lost >= 0.05 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for mode in MODES:
        v = per_mode[mode]
        gained = sum(1 for q, x in per_gold[mode].items() if x - base[q] >= 0.1)
        lost = sum(1 for q, x in per_gold[mode].items() if base[q] - x >= 0.05)
        ceiling = ACCURACY_WEIGHT * 1.0 + TIOU_WEIGHT * st.mean(v)
        g_cell = '-' if mode == 'sentence' else str(gained)
        l_cell = '-' if mode == 'sentence' else str(lost)
        print(f'| {mode} | {st.mean(n_units[mode]):.1f} | {st.mean(unit_words[mode]):.1f} | '
              f'{st.mean(v):.3f} | {sum(x >= 0.9 for x in v)} of {n} | {ceiling:.3f} | '
              f'{g_cell} | {l_cell} |')

    print('\nDistribution of the oracle (count of golds):\n')
    print('| unit mode | < 0.5 | 0.5-0.7 | 0.7-0.9 | >= 0.9 | median |')
    print('|---|---:|---:|---:|---:|---:|')
    for mode in MODES:
        v = per_mode[mode]
        print(f'| {mode} | {sum(x < 0.5 for x in v)} | {sum(0.5 <= x < 0.7 for x in v)} | '
              f'{sum(0.7 <= x < 0.9 for x in v)} | {sum(x >= 0.9 for x in v)} | {st.median(v):.3f} |')

    for mode in MODES[1:]:
        d = [per_gold[mode][q] - base[q] for q in base]
        print(f'\n{mode} against sentence: mean {st.mean(d):+.3f}, '
              f'better {sum(x > 0.005 for x in d)}, worse {sum(x < -0.005 for x in d)}, '
              f'unchanged {sum(abs(x) <= 0.005 for x in d)}; '
              f'largest single gain {max(d):+.3f}, largest single loss {min(d):+.3f}')

    worst = sorted((d for d in detail if d['tiou'] < GAP), key=lambda d: d['tiou'])[:WORST]
    print(f'\n\n## The {len(worst)} worst golds under clause units (oracle < {GAP})\n')
    for d in worst:
        g0, g1 = d['gold']
        print(f'--- {d["qid"]}  oracle {d["tiou"]:.3f}  gold [{g0:.2f}-{g1:.2f}] ({g1 - g0:.2f}s)')
        print(f'    Q      {d["question"]}')
        print(f'    GOLD   {d["gold_text"]}')
        print(f'    BEST   {" || ".join(d["units"]) or "(none)"}')
        for t in d['near']:
            print(f'    NEAR   {t}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
