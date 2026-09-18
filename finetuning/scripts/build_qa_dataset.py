"""Build an extractive-QA training set from the medical-appointment case data.

The case ships 390 yes/no questions over 39 consultations, and for the 195
`positive` rows an evidence span in *seconds* of audio. A span extractor needs
*characters* of text, so this script joins the Whisper large-v3 word timings
into one context string per conversation and converts each gold time interval
into the character range of the words it covers.

    python scripts/build_qa_dataset.py

Outputs into data/:
  qa_train.jsonl        390 records, one per question (SQuAD-v2 shaped)
  qa_train_squad.json   the same, in the nested SQuAD-v2 layout run_qa.py wants
  contexts.json         per-conversation char<->time alignment (words + units)

The alignment file is the part that makes the dataset scoreable: a predicted
character span is only worth anything here once it is mapped back to seconds
and compared with the gold interval by tIoU.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

# Unit splitting, kept identical to medical-appointment/model.py so that spans
# snapped to units here mean the same thing as they do in the served pipeline.
TERMINAL = re.compile(r'[.!?]["\')\]]*$')
PAUSE_SPLIT = 0.6

# Edge offsets fitted leave-one-conversation-out on these same 39 files
# (medical-appointment/model.py, large-v3 row). The annotators' interval starts
# near the *end* of its first word, not its onset, so a span mapped back to
# seconds has to anchor there or it reads ~0.3 s early every time.
START_OFFSET = -0.14
END_OFFSET = 0.12

# Below this much of the gold interval covered by transcript words, the span is
# not recoverable from this transcript and the row is marked unaligned.
MIN_COVERAGE = 0.5

DEFAULT_CASE = Path(__file__).resolve().parents[2] / 'medical-appointment'
ASR_TAG = 'large-v3'


def tiou(a, b):
    """Temporal IoU, the case's own metric."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def load_conversation(case: Path, tid: str):
    """One transcript -> context string, words and units with char offsets."""
    path = case / 'transcripts' / f'conversation_{tid}.{ASR_TAG}.json'
    data = json.loads(path.read_text(encoding='utf-8'))

    parts, words, pos = [], [], 0
    for seg in data['segments']:
        for i, w in enumerate(seg['words']):
            tok = w['w']
            if not parts:            # no leading space on the very first token
                tok = tok.lstrip()
            if not tok:
                continue
            lead = len(tok) - len(tok.lstrip())
            parts.append(tok)
            words.append({
                'text': tok.strip(),
                'start': float(w['start']),
                'end': float(w['end']),
                'char_start': pos + lead,   # first non-space character
                'char_end': pos + len(tok),
                'seg_end': i == len(seg['words']) - 1,
            })
            pos += len(tok)
    context = ''.join(parts)

    # sentence-ish units: terminal punctuation, segment ends, or a long pause
    units, cur = [], []

    def flush():
        if not cur:
            return
        cs, ce = cur[0]['char_start'], cur[-1]['char_end']
        text = context[cs:ce].strip()
        if text:
            units.append({
                'idx': len(units),
                'start': cur[0]['start'],
                'end': cur[-1]['end'],
                'first_word_end': cur[0]['end'],
                'char_start': cs,
                'char_end': cs + len(text),
                'text': text,
            })
        cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if TERMINAL.search(w['text']) or w['seg_end']:
            flush()
        elif nxt is not None and nxt['start'] - w['end'] >= PAUSE_SPLIT:
            flush()
    flush()

    return {
        'transcript_id': tid,
        'audio_file': f'conversation_{tid}.mp3',
        'duration': float(data['duration']),
        'context': context,
        'words': words,
        'units': units,
    }


def anchored(first, last):
    """Word times -> the seconds convention the annotations actually use."""
    return first['end'] + START_OFFSET, last['end'] + END_OFFSET


def span_from_time(conv, gs: float, ge: float):
    """Gold seconds -> character range, via the words the interval covers."""
    hit = [w for w in conv['words'] if w['end'] > gs and w['start'] < ge]
    covered = sum(min(w['end'], ge) - max(w['start'], gs) for w in hit)
    coverage = covered / (ge - gs) if ge > gs else (1.0 if hit else 0.0)
    if not hit:  # interval lands in a gap; take the nearest word instead
        mid = (gs + ge) / 2
        hit = [min(conv['words'], key=lambda w: abs((w['start'] + w['end']) / 2 - mid))]
    return hit[0], hit[-1], coverage


def span_from_units(conv, gs: float, ge: float):
    """Same interval, snapped out to whole units."""
    hit = [u for u in conv['units'] if u['end'] > gs and u['start'] < ge]
    if not hit:
        mid = (gs + ge) / 2
        hit = [min(conv['units'], key=lambda u: abs((u['start'] + u['end']) / 2 - mid))]
    start = hit[0]['first_word_end'] + START_OFFSET if hit[0]['first_word_end'] > 0 else hit[0]['start']
    return (hit[0]['char_start'], hit[-1]['char_end'],
            start, hit[-1]['end'] + END_OFFSET, [u['idx'] for u in hit])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', type=Path, default=DEFAULT_CASE,
                    help='medical-appointment folder holding data/ and transcripts/')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'data')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.case / 'data' / 'question_train.csv', encoding='utf-8')))
    tids = sorted({r['transcript_id'] for r in rows})
    convs = {t: load_conversation(a.case, t) for t in tids}

    records, word_tiou, unit_tiou, fallbacks = [], [], [], []
    for r in rows:
        conv = convs[r['transcript_id']]
        positive = r['question_type'] == 'positive'
        rec = {
            'id': r['question_id'],
            'transcript_id': r['transcript_id'],
            'audio_file': conv['audio_file'],
            'duration': conv['duration'],
            'question': r['question'],
            'context': conv['context'],
            'answer_yesno': r['answer'],
            'label': int(r['label']),
            'question_type': r['question_type'],
            'is_impossible': not positive,
            'answers': {'text': [], 'answer_start': []},
        }
        if positive:
            gs, ge = float(r['evidence_start']), float(r['evidence_end'])
            first, last, coverage = span_from_time(conv, gs, ge)
            cs, ce = first['char_start'], last['char_end']
            ws, we = anchored(first, last)
            ucs, uce, us, ue, uids = span_from_units(conv, gs, ge)
            ok = coverage >= MIN_COVERAGE
            rec.update({
                'answers': {'text': [conv['context'][cs:ce]], 'answer_start': [cs]},
                'answer_end': ce,
                'evidence_start': gs,
                'evidence_end': ge,
                'answer_time_start': round(ws, 2),
                'answer_time_end': round(we, 2),
                'answers_unit': {'text': [conv['context'][ucs:uce]], 'answer_start': [ucs]},
                'answer_unit_end': uce,
                'unit_ids': uids,
                'unit_time_start': round(us, 2),
                'unit_time_end': round(ue, 2),
                'gold_coverage': round(coverage, 3),
                'alignment_ok': ok,
            })
            if ok:
                word_tiou.append(tiou((gs, ge), (ws, we)))
                unit_tiou.append(tiou((gs, ge), (us, ue)))
            else:
                fallbacks.append(r['question_id'])
        records.append(rec)

    a.out.mkdir(parents=True, exist_ok=True)

    with open(a.out / 'qa_train.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # nested SQuAD-v2 layout, one paragraph per conversation
    by_conv = {}
    for rec in records:
        by_conv.setdefault(rec['transcript_id'], []).append(rec)
    squad = {'version': 'medical-appointment-v1', 'data': []}
    for tid in tids:
        qas = []
        for rec in by_conv[tid]:
            answers = []
            if rec['answers']['text']:
                answers = [{'text': rec['answers']['text'][0],
                            'answer_start': rec['answers']['answer_start'][0]}]
            qas.append({
                'id': rec['id'],
                'question': rec['question'],
                'answers': answers,
                'is_impossible': rec['is_impossible'],
            })
        squad['data'].append({
            'title': tid,
            'paragraphs': [{'context': convs[tid]['context'], 'qas': qas}],
        })
    (a.out / 'qa_train_squad.json').write_text(
        json.dumps(squad, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')

    (a.out / 'contexts.json').write_text(
        json.dumps(convs, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')

    # ---- diagnostics -------------------------------------------------------
    n_pos = sum(r['question_type'] == 'positive' for r in rows)
    print(f'{len(records)} questions over {len(tids)} conversations -> {a.out}')
    for qt in ('positive', 'hard_negative', 'off_topic'):
        print(f'  {qt:<15} {sum(r["question_type"] == qt for r in rows)}')
    n_ok = len(word_tiou)
    print()
    print(f'Span conversion, seconds -> characters -> seconds, over the {n_ok} aligned '
          f'positives (tIoU vs the gold interval).')
    print('This is the ceiling: what a span extractor scores if it is always exactly right.')
    print(f'  word-exact span   mean {st.mean(word_tiou):.3f}   min {min(word_tiou):.3f}   '
          f'<0.5: {sum(x < 0.5 for x in word_tiou)}/{n_ok}')
    print(f'  unit-snapped span mean {st.mean(unit_tiou):.3f}   min {min(unit_tiou):.3f}   '
          f'<0.5: {sum(x < 0.5 for x in unit_tiou)}/{n_ok}')
    if fallbacks:
        print()
        print(f'  UNALIGNED: {len(fallbacks)}/{n_pos} positives whose gold interval is <'
              f'{MIN_COVERAGE:.0%} covered by transcript words.')
        print('  The ASR dropped that speech, so the span is not in the text at any offset.')
        print(f'  Marked alignment_ok=false, excluded above, kept in the file: {fallbacks}')
    lens = [len(c['context'].split()) for c in convs.values()]
    chars = [len(c['context']) for c in convs.values()]
    print()
    print(f'Context length: {min(lens)}-{max(lens)} words (median {int(st.median(lens))}), '
          f'{min(chars)}-{max(chars)} chars')
    ans = [len(r['answers']['text'][0]) for r in records if r['answers']['text']]
    print(f'Answer length:  {min(ans)}-{max(ans)} chars (median {int(st.median(ans))})')

    # every offset must actually point at its own text
    bad = [r['id'] for r in records if r['answers']['text']
           and convs[r['transcript_id']]['context']
               [r['answers']['answer_start'][0]:r['answer_end']] != r['answers']['text'][0]]
    print(f'Offset self-check: {"OK" if not bad else "FAILED " + str(bad[:5])}')


if __name__ == '__main__':
    raise SystemExit(main())
