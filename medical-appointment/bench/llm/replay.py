"""Re-score stored bench runs from their raw model answers, without any inference.

Every result JSON under bench/results/llm/ keeps the parsed model output of each
question (`raw`). This script rebuilds the units from the same transcripts, runs
the variant's post-processing again with the edge offsets that belong to the
transcripts' ASR model (or any pair given on the command line), and scores the
outcome under both span policies:

  spans     a span is returned for every question (model.SPAN_ON_NO=1, what the
            portal credits: findings log entry 38)
  nulls     a question answered no returns no span (the policy the bench and the
            findings log entries 30 to 37 quoted)

Why it exists: the bench runs of 2026-09-17 on turbo transcripts were scored with
large-v3's offsets because ASR_MODEL was never exported (entry 38); the stored
`summary` blocks of those files are therefore wrong by 0.01 to 0.02 and this
script is the way to read them.

    python bench/llm/replay.py                        # every final result file, table on stdout
    python bench/llm/replay.py bench/results/llm/qwen3.8-27b.*.json --md research/replay.md
    python bench/llm/replay.py --offsets=-0.22,0.00   # what-if pair for every file (the = form: the value starts with a minus)
    python bench/llm/replay.py --per-question out.json  # also dump per-question replayed tIoU

The table lists, per run: accuracy, mean tIoU and score under both policies, the
score the file itself stored, and the offsets used. Runs whose config names a
variant this checkout does not know, or whose transcripts are missing, are listed
under "skipped".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
CASE = HERE.parent.parent
for p in (str(CASE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model                                      # noqa: E402
import prompts                                    # noqa: E402
from bench import load_words, TRANSCRIPTS, RESULTS  # noqa: E402
from model import make_units                      # noqa: E402
from utils import gold_evidence, group_questions_by_conversation, temporal_iou  # noqa: E402
from local_evaluator import Statistics, UNANSWERED  # noqa: E402


def set_offsets(start: float, end: float) -> None:
    """Point both modules that read the offsets at the given pair. model.span_from_ids
    reads model.START_OFFSET at call time; prompts._units_post/_words_post read their
    own imported copies."""
    model.START_OFFSET = start
    model.END_OFFSET = end
    prompts.START_OFFSET = start
    prompts.END_OFFSET = end


def fitted(asr: str) -> Tuple[float, float]:
    fwe, _us, end = model._FITTED.get(asr, model._FITTED['large-v3'])
    return fwe, end


_CONV_CACHE: Dict[Tuple[str, str, str], Tuple[list, float, list]] = {}


def conversation(stem: str, asr: str, unit_split: str = 'sentence'):
    """Words, duration and units of one conversation, the units cut the way the
    stored run cut them (config.unit_split; runs before 2026-09-18 are sentence)."""
    key = (stem, asr, unit_split)
    if key not in _CONV_CACHE:
        words, duration, _ = load_words(TRANSCRIPTS / f'{stem}.{asr}.json')
        _CONV_CACHE[key] = (words, duration, make_units(words, unit_split))
    return _CONV_CACHE[key]


def replay_file(path: Path, offsets: Optional[Tuple[float, float]], by_tid: Dict[str, Tuple[str, List[dict]]]) -> dict:
    d = json.loads(path.read_text(encoding='utf-8'))
    cfg = d.get('config', {})
    asr, vname = cfg.get('asr'), cfg.get('variant')
    if vname not in prompts.VARIANTS:
        raise ValueError(f'unknown variant {vname!r}')
    variant = prompts.VARIANTS[vname]
    pair = offsets or fitted(asr)
    set_offsets(*pair)
    spans, nulls = Statistics(), Statistics()
    per_q = []
    # group the stored records by conversation so few-shot / joint variants build once
    groups: Dict[str, List[dict]] = {}
    for r in d['questions']:
        groups.setdefault(r['transcript_id'], []).append(r)
    for tid, recs in groups.items():
        stem, rows = by_tid[tid]
        words, duration, units = conversation(stem, asr, cfg.get('unit_split') or 'sentence')
        if hasattr(variant, 'set_conversation'):
            variant.set_conversation(stem, asr)
        joint = getattr(variant, 'joint', False)
        p_joint = variant.build_all([r['question'] for r in recs], units) if joint else None
        for r in recs:
            gold = tuple(r['gold']) if r.get('gold') else None
            raw = r.get('raw')
            if raw is None:
                pred, span = UNANSWERED, None
            else:
                p = p_joint if joint else variant(r['question'], units)
                try:
                    yes, span = p.postprocess(raw, units, words, duration)
                except Exception:
                    yes, span = False, None
                pred = int(bool(yes))
                span = tuple(span) if span else None
            spans.record(r['question_type'], int(r['label']), pred, gold, span)
            nulls.record(r['question_type'], int(r['label']), pred, gold, span if pred == 1 else None)
            per_q.append({'question_id': r['question_id'], 'transcript_id': tid, 'label': int(r['label']),
                          'prediction': pred, 'span': list(span) if span else None, 'gold': list(gold) if gold else None,
                          'tiou': temporal_iou(gold, span) if (gold and span) else (0.0 if gold else None)})
    return {'file': path.name, 'model': cfg.get('model'), 'variant': vname, 'asr': asr,
            'unit_split': cfg.get('unit_split') or 'sentence',
            'partial': bool(d.get('summary', {}).get('partial')), 'questions': spans.total,
            'offsets': list(pair), 'stored_offsets': [cfg.get('start_offset'), cfg.get('end_offset')],
            'accuracy': spans.accuracy,
            'spans': {'mean_tiou': spans.mean_tiou, 'score': spans.final_score},
            'nulls': {'mean_tiou': nulls.mean_tiou, 'score': nulls.final_score},
            'stored': {'score': d.get('summary', {}).get('score'),
                       'nulls_score': (d.get('summary', {}).get('nulls_on_no') or {}).get('score')},
            'per_question': per_q}


def short(model_name: Optional[str]) -> str:
    if not model_name:
        return '?'
    return model_name.split('/')[-1][:24]


def table(rows: List[dict], md: bool) -> str:
    rows = sorted(rows, key=lambda r: -r['spans']['score'])
    head = ['model', 'variant', 'asr', 'n', 'acc', 'tIoU sp', 'score sp', 'tIoU nl', 'score nl', 'stored', 'offsets']
    lines = []
    for r in rows:
        st = r['stored']['score']
        lines.append([short(r['model']) + (' (partial)' if r['partial'] else ''), r['variant'], r['asr'], str(r['questions']),
                      f"{r['accuracy']:.3f}", f"{r['spans']['mean_tiou']:.3f}", f"{r['spans']['score']:.4f}",
                      f"{r['nulls']['mean_tiou']:.3f}", f"{r['nulls']['score']:.4f}",
                      f'{st:.4f}' if isinstance(st, (int, float)) else '-',
                      f"{r['offsets'][0]:+.2f}/{r['offsets'][1]:+.2f}"])
    if md:
        out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
        out += ['| ' + ' | '.join(l) + ' |' for l in lines]
        return '\n'.join(out)
    w = [max(len(h), *(len(l[i]) for l in lines)) if lines else len(h) for i, h in enumerate(head)]
    out = ['  '.join(h.ljust(w[i]) for i, h in enumerate(head))]
    out += ['  '.join(c.ljust(w[i]) for i, c in enumerate(l)) for l in lines]
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='result JSONs (default: every non-partial file under bench/results/llm)')
    ap.add_argument('--offsets', default='', help='start,end pair to use for every file instead of the fitted one, e.g. -0.22,0.00')
    ap.add_argument('--md', default='', help='write the table as markdown to this path')
    ap.add_argument('--json', default='', help='write the per-run summaries (no per-question rows) to this path')
    ap.add_argument('--per-question', default='', help='write per-question replayed rows for every run to this path')
    ap.add_argument('--include-partial', action='store_true')
    ap.add_argument('--rewrite', default='', help='directory: write a corrected copy of every replayed result file there '
                    '(same shape, spans/tiou/prediction/summary replaced, config.replayed=true) for tools that read result files')
    a = ap.parse_args()

    offsets = None
    if a.offsets:
        s, e = a.offsets.split(',')
        offsets = (float(s), float(e))
    files = [Path(f) for f in a.files] if a.files else sorted(p for p in RESULTS.glob('*.json')
                                                              if not p.name.endswith('.partial.json'))
    by_tid: Dict[str, Tuple[str, List[dict]]] = {}
    for fn, rows in group_questions_by_conversation():
        by_tid[rows[0]['transcript_id']] = (Path(fn).stem, rows)

    results, skipped = [], []
    for f in files:
        try:
            r = replay_file(f, offsets, by_tid)
        except Exception as exc:
            skipped.append(f'{f.name}: {type(exc).__name__}: {exc}')
            continue
        if r['partial'] and not a.include_partial:
            skipped.append(f'{f.name}: partial run')
            continue
        results.append(r)
        if a.rewrite:
            d = json.loads(f.read_text(encoding='utf-8'))
            by_q = {q['question_id']: q for q in r['per_question']}
            for rec in d['questions']:
                q = by_q[rec['question_id']]
                rec['span'], rec['tiou'], rec['prediction'] = q['span'], q['tiou'], q['prediction']
                rec['answer'] = None if q['prediction'] == UNANSWERED else bool(q['prediction'])
                rec['correct'] = q['prediction'] == q['label']
            d['summary'].update({'accuracy': r['accuracy'], 'mean_tiou': r['spans']['mean_tiou'], 'score': r['spans']['score'],
                                 'nulls_on_no': {'mean_tiou': r['nulls']['mean_tiou'], 'score': r['nulls']['score']}})
            d['config'].update({'start_offset': r['offsets'][0], 'end_offset': r['offsets'][1], 'replayed': True,
                                'replayed_from': f.name})
            out_dir = Path(a.rewrite); out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f.name).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f"{f.name}: acc {r['accuracy']:.3f}  spans {r['spans']['score']:.4f}  nulls {r['nulls']['score']:.4f}"
              f"  stored {r['stored']['score'] if r['stored']['score'] is not None else '-'}"
              f"  offsets {r['offsets'][0]:+.2f}/{r['offsets'][1]:+.2f}", file=sys.stderr, flush=True)

    print()
    print(table(results, md=False))
    for s in skipped:
        print(f'skipped {s}')
    if a.md:
        Path(a.md).write_text(table(results, md=True) + '\n', encoding='utf-8')
    if a.json:
        Path(a.json).write_text(json.dumps([{k: v for k, v in r.items() if k != 'per_question'} for r in results],
                                           indent=1), encoding='utf-8')
    if a.per_question:
        Path(a.per_question).write_text(json.dumps({r['file']: r['per_question'] for r in results}), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
