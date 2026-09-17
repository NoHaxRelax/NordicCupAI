#!/usr/bin/env python3
"""ASR report: for every transcript tag under transcripts/, how good are the word
timestamps for evidence spans, and how good is the text.

Per tag (one JSON in bench/results/asr/<tag>.json, one row per markdown table):

  1. files covered and real-time factor RTF = transcription seconds / audio
     seconds, pooled over files like transcribe_cache.py's DONE line (the
     per-file mean is in the JSON too).
  2. oracle-selection tIoU ceiling for segment, sentence and merge<=K units with
     span_ceiling.py's definitions (its sentences / merges / tiou are imported,
     not copied); then the merge ceiling after shifting every candidate by the
     fitted median start/end offsets, and by the constants model.py serves with
     (START_OFFSET / END_OFFSET); the same for model.make_units (pause split),
     so the bench measures the unit code that serves.
  3. signed edge offsets gold minus model for the best merge: median, p25, p75,
     mean, for starts and for ends.
  4. where inside the nearest model word each annotated edge falls:
         f = (gold - word_start) / (word_end - word_start)
     f < 0 is before the word starts, 0..1 inside it, > 1 after it ends. Reported
     as median / p10 / p90 / mean and a histogram over the bins
     [-inf,-0.5,0,0.25,0.5,0.75,1,1.5,inf], separately for starts and for ends,
     plus the absolute distance in ms from each gold start to the nearest word
     start and from each gold end to the nearest word end, with the share
     within 20 / 50 / 100 ms (an edge on the annotators' own tool sits at 0 on
     the 20 ms grid). Nearest = smallest distance to the word's interval, 0 when
     inside; ties (marker exactly on a shared boundary) go to the later word for
     starts (f = 0) and to the earlier word for ends (f = 1).
  5. WER with jiwer 4.x against bench/ref/<stem>.txt files whose first line is
     "# checked" (hand-corrected, see bench/ref/PROTOCOL.md; the unchecked
     pre-fills are Whisper text and are skipped unless --include-unchecked) and
     against a silver tag given with --ref TAG; substitutions / deletions /
     insertions, and the WER restricted to the words inside the annotated
     evidence spans. A leading "[m:ss.s]" marker and a leading "Doctor:" /
     "Patient:" label on a reference line are stripped. A checked file whose
     normalised words still equal one tag's transcript was never corrected and
     is dropped with a warning (it would score that tag 0 and the others
     against Whisper). A transcript with no words counts as 100% deletions and
     stays in the pool (the "n=8, 1 empty" cell), rather than silently
     improving the pooled number.

Install (Linux venv; add to the bench venv spec in bench/hpc/env.sh):
    pip install "jiwer>=4,<5" num2words
    pip install requests pydantic      # optional: serve-units columns (model.py) and utils.py helpers

Run (from medical-appointment/):
    python bench/asr/compare.py                                        # every tag in transcripts/
    python bench/asr/compare.py --tags large-v3,parakeet-tdt-0.6b-v2 --ref canary-qwen-2.5b
    python bench/asr/compare.py --suggest-ref 8 --draft large-v3      # pick files + write drafts for PROTOCOL.md
    python bench/asr/compare.py --verbose                              # also the worst spans per tag

Outputs: markdown tables on stdout (also bench/results/asr/summary.md) and
bench/results/asr/<tag>.json per tag: the bench/README.md results schema
{tag, n_files, rtf, wer, ceiling, offset, first_word_fraction} plus
last_word_fraction, boundary_ms, wer_silver and a per-span list.

Caveats:
  - The text normaliser is ours: lowercase, digits -> num2words (years as
    "twenty seventeen", ordinals as "third"), % -> percent, letter/digit splits
    ("100mg" -> "one hundred mg"), punctuation stripped, hyphens split. It is
    not the Whisper normaliser, so absolute WER is not comparable with the Open
    ASR Leaderboard; it is applied to both sides, so tags compare with each other.
  - Quantiles use span_ceiling.py's convention (sorted[int(p*n)]) so the offset
    numbers match what that script prints.
  - Tags with "word_timestamps": false get RTF and WER only (in-span WER only
    when the reference side has word times, i.e. a silver tag with words).
  - Words with missing or non-finite times get zero duration at the previous
    word's end so sentence splitting still works; they are excluded from the
    nearest-word search and counted in n_untimed_words. Zero-duration words
    (the Qwen3-ForcedAligner bug) are counted in n_zero_duration_words.
  - A transcript that fails to load (truncated JSON from a runner killed
    mid-write, "segments": null) is skipped with a note on stderr and listed
    under unreadable_files; one that loads but has no words is listed under
    empty_files and scored as all deletions (a runner bug must not look like
    a better WER); a tag whose analysis fails is skipped with a traceback, and
    the other tags' JSONs and summary.md are still written.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics as st
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent            # bench/asr
CASE = HERE.parent.parent                          # medical-appointment
for _p in (str(HERE), str(CASE)):                  # CASE ends up first
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import common                                       # noqa: E402  bench/asr/common.py
from span_ceiling import merges, sentences, tiou    # noqa: E402  medical-appointment/span_ceiling.py

try:
    import utils as case_utils                      # gold_evidence, audio_filename_for_transcript, ...
    _UTILS_ERR = None
except Exception as _e:                             # dtos -> pydantic missing in a minimal venv
    case_utils, _UTILS_ERR = None, _e
try:
    import model as serving                         # make_units, START_OFFSET, END_OFFSET
    _SERVE_ERR = None
except Exception as _e:                             # requests missing in a minimal venv
    serving, _SERVE_ERR = None, _e
try:
    import jiwer                                    # https://jitsi.github.io/jiwer/usage/
except Exception:
    jiwer = None
try:
    from num2words import num2words                 # https://github.com/savoirfairelinux/num2words
except Exception:
    num2words = None

DEFAULT_CSV = CASE / 'data' / 'question_train.csv'
DEFAULT_REF = HERE.parent / 'ref'
DEFAULT_RESULTS = HERE.parent / 'results' / 'asr'
DEFAULT_DRAFTS = HERE.parent / 'results' / 'ref_drafts'

HIST_EDGES = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
HIST_LABELS = ['<-0.5', '[-0.5,0)', '[0,0.25)', '[0.25,0.5)', '[0.5,0.75)', '[0.75,1)', '[1,1.5)', '>=1.5']

Span = Tuple[float, float]
Tok = Tuple[str, Optional[float], Optional[float]]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def q(xs: Sequence[float], p: float) -> Optional[float]:
    """Quantile with span_ceiling.py's convention: sorted[int(p*n)]."""
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s) + 1e-9))]


def mean(xs: Sequence[float]) -> Optional[float]:
    return st.mean(xs) if xs else None


def fmt(x, nd: int = 3, plus: bool = False) -> str:
    if x is None:
        return '-'
    return f'{x:+.{nd}f}' if plus else f'{x:.{nd}f}'


def pct(x) -> str:
    return '-' if x is None else f'{100 * x:.0f}%'


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def discover_tags(tr_dir: Path, audio_dir: Path) -> Dict[str, Dict[str, Path]]:
    """{tag: {stem: path}} for every transcripts/<stem>.<tag>.json. Tags may
    contain dots (parakeet-tdt-0.6b-v2), so the stem is matched against the
    audio file stems first (longest first: sample_1 must not swallow sample_10)."""
    stems = sorted((p.stem for p in audio_dir.glob('*.mp3')), key=len, reverse=True)
    tags: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for p in sorted(tr_dir.glob('*.json')):
        name = p.name[:-5]
        stem = next((s for s in stems if name.startswith(s + '.')), None)
        if stem is None:
            m = re.match(r'^(conversation_[^.]+)\.(.+)$', name)
            if not m:
                continue
            stem, tag = m.groups()
        else:
            tag = name[len(stem) + 1:]
        if tag:
            tags[tag][stem] = p
    return dict(tags)


def read_rows(csv_path: Path) -> List[dict]:
    with open(csv_path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _gold_fallback(r: dict) -> Optional[Span]:
    """Only used when utils.py cannot be imported (pydantic missing)."""
    try:
        s, e = float(r['evidence_start']), float(r['evidence_end'])
    except (TypeError, ValueError, KeyError):
        return None
    return (s, e) if math.isfinite(s) and math.isfinite(e) and e >= s else None


def stem_of(transcript_id: str) -> str:
    if case_utils is not None:
        return Path(case_utils.audio_filename_for_transcript(transcript_id)).stem
    return f'conversation_{transcript_id}'


def gold_by_stem(rows: List[dict]) -> Dict[str, List[Tuple[dict, Span]]]:
    ev = case_utils.gold_evidence if case_utils is not None else _gold_fallback
    out: Dict[str, List[Tuple[dict, Span]]] = defaultdict(list)
    for r in rows:
        g = ev(r)
        if g is not None:
            out[stem_of(r['transcript_id'])].append((r, g))
    return dict(out)


def load_transcript(path: Path) -> dict:
    """Read one transcript and make it safe for span_ceiling.sentences: every
    segment has words/text/start/end, every word has finite float times."""
    d = common.read_transcript(path)
    d['segments'] = d.get('segments') or []           # "segments": null must not reach the loop (setdefault keeps None)
    n_untimed = n_zero = 0
    prev_end = 0.0
    for s in d['segments']:
        ws = s.get('words') or []
        s['words'] = ws
        for w in ws:
            w['w'] = str(w.get('w', ''))
            if finite(w.get('start')) and finite(w.get('end')):
                w['start'], w['end'] = float(w['start']), float(w['end'])
                if w['end'] < w['start']:
                    w['end'] = w['start']
                prev_end = w['end']
            else:
                n_untimed += 1
                w['start'] = w['end'] = prev_end
            if w['end'] <= w['start']:
                n_zero += 1
        if not isinstance(s.get('text'), str):
            s['text'] = ''.join(w['w'] for w in ws)
        if not finite(s.get('start')):
            s['start'] = ws[0]['start'] if ws else 0.0
        if not finite(s.get('end')):
            s['end'] = ws[-1]['end'] if ws else float(d.get('duration') or 0.0)
        s['start'], s['end'] = float(s['start']), float(s['end'])
    d['_n_untimed'] = n_untimed
    d['_n_zero'] = n_zero
    d['_wt'] = bool(d.get('word_timestamps', True)) and any(s['words'] for s in d['segments'])
    return d


# --------------------------------------------------------------------------- #
# units and ceilings
# --------------------------------------------------------------------------- #

def best_merge(units: Sequence, g: Span, k: int, ds: float = 0.0, de: float = 0.0):
    """Best tIoU over merges of 1..k consecutive units, candidates shifted by
    (ds, de). units are (start, end, ...) tuples as span_ceiling.merges expects."""
    best, span = 0.0, None
    for (s0, s1, n) in merges(units, k):
        v = tiou(g, (s0 + ds, s1 + de))
        if v > best:
            best, span = v, (s0, s1, n)
    return best, span


def serve_units(d: dict) -> List[Tuple[float, float]]:
    """The units model.py serves with (punctuation, segment ends, pauses)."""
    words = []
    for s in d['segments']:
        for w in s['words']:
            words.append(serving.Word(w['w'], w['start'], w['end']))
        if s['words']:
            words[-1].w += '\x00'          # segment boundary marker, as model._cached_transcript sets it
    return [(u.start, u.end) for u in serving.make_units(words)]


def nearest_word(t: float, words: Sequence[Tuple[str, float, float]], later: bool) -> int:
    """Index of the word whose interval is nearest t (0 when t is inside).
    Ties go to the later word when later=True, else to the earlier one."""
    best, bestd = -1, math.inf
    for i, (_, s, e) in enumerate(words):
        dist = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
        if dist < bestd - 1e-9 or (later and abs(dist - bestd) <= 1e-9):
            best, bestd = i, dist
    return best


def fraction_stats(fs: List[float]) -> dict:
    if not fs:
        return {'n': 0}
    hist = [0] * len(HIST_LABELS)
    for f in fs:
        hist[bisect.bisect_right(HIST_EDGES, f)] += 1
    n = len(fs)
    return {
        'n': n, 'median': q(fs, .5), 'p10': q(fs, .1), 'p90': q(fs, .9), 'mean': st.mean(fs),
        'before_word': sum(f < 0 for f in fs) / n,
        'inside_word': sum(0 <= f <= 1 for f in fs) / n,
        'after_word': sum(f > 1 for f in fs) / n,
        'hist': dict(zip(HIST_LABELS, hist)),
    }


def boundary_stats(signed: List[float]) -> dict:
    """signed = gold edge minus nearest same-kind word boundary, seconds."""
    if not signed:
        return {'n': 0}
    a = [abs(x) * 1000 for x in signed]
    n = len(a)
    return {
        'n': n, 'abs_median_ms': q(a, .5), 'abs_p10_ms': q(a, .1), 'abs_p90_ms': q(a, .9),
        'abs_mean_ms': st.mean(a), 'signed_median_ms': 1000 * q(signed, .5),
        'within_20ms': sum(x <= 20.5 for x in a) / n,
        'within_50ms': sum(x <= 50.5 for x in a) / n,
        'within_100ms': sum(x <= 100.5 for x in a) / n,
    }


def offset_block(offs: List[Span], best_n: List[int]) -> dict:
    ds = [o[0] for o in offs]
    de = [o[1] for o in offs]
    if not offs:
        return {'n': 0}
    return {
        'n': len(offs),
        'start_median': q(ds, .5), 'start_p25': q(ds, .25), 'start_p75': q(ds, .75), 'start_mean': st.mean(ds),
        'end_median': q(de, .5), 'end_p25': q(de, .25), 'end_p75': q(de, .75), 'end_mean': st.mean(de),
        'best_merge_sizes': {str(k): v for k, v in sorted(Counter(best_n).items())},
    }


# --------------------------------------------------------------------------- #
# text normalisation and WER
# --------------------------------------------------------------------------- #

_APOS = str.maketrans({'’': "'", '‘': "'", 'ʼ': "'", '´': "'"})
_SYMS = [(re.compile(r'%'), ' percent '), (re.compile(r'&'), ' and '), (re.compile(r'\+'), ' plus '),
         (re.compile(r'[/–—-]'), ' ')]
_ORD = re.compile(r'\b(\d+)(st|nd|rd|th)\b')
_ALNUM = re.compile(r'(?<=\d)(?=[^\d\s.,])|(?<=[^\d\s.,])(?=\d)')
_NUM = re.compile(r'\d+(?:[.,]\d+)*')
_THOUSANDS = re.compile(r'\d{1,3}(?:,\d{3})+')
_PUNCT = re.compile(r"[^\w' ]+")
_SPEAKER = re.compile(r'^\s*(?:doctor|patient|dr|pt|gp|d|p)\s*:\s*', re.I)   # colon only: "D-dimer", "P-value", "GP-led" are words, not labels
_LINE_TS = re.compile(r'^\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?[\])]?\s*')   # [0:05.2]  (00:05)  1:02:03.450


def _num_words(s: str) -> str:
    if _THOUSANDS.fullmatch(s):
        s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')                     # 2,5 -> 2.5
    try:
        if '.' in s:                                # 5.0 -> "five point zero", 4.25 -> "four point two five" (as spoken)
            whole, frac = s.split('.', 1)
            words = num2words(int(whole or 0), lang='en') + ' point ' + ' '.join(num2words(int(c), lang='en') for c in frac if c.isdigit())
        else:
            v = int(s)                              # https://github.com/savoirfairelinux/num2words: to='year' | 'cardinal'
            words = num2words(v, to='year' if 1100 <= v <= 2099 else 'cardinal', lang='en')
    except Exception:
        return s
    return f' {words} '


def _ord_words(m: re.Match) -> str:
    try:
        return f" {num2words(int(m.group(1)), to='ordinal', lang='en')} "   # 3rd -> "third"
    except Exception:
        return m.group(0)


def norm_chunk(text: str) -> List[str]:
    """Normalise one chunk of text (a word, a line, a whole file) to tokens:
    lowercase, numbers spelled out, symbols to words, punctuation gone."""
    t = text.lower().translate(_APOS)
    for rx, rep in _SYMS:
        t = rx.sub(rep, t)
    if num2words is not None:
        t = _ORD.sub(_ord_words, t)
        t = _ALNUM.sub(' ', t)                      # 100mg -> 100 mg, hba1c -> hba 1 c
        t = _NUM.sub(lambda m: _num_words(m.group(0)), t)
    t = _PUNCT.sub(' ', t).replace('_', ' ')
    return [x.strip("'") for x in t.split() if x.strip("'")]


def ref_is_checked(path: Path) -> bool:
    """bench/ref/INDEX.md convention: a hand-corrected file starts with '# checked'.
    The pre-filled files are Whisper text and would score WER 0 against it.
    main() additionally drops a checked file whose words still equal a tag's."""
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            return line.strip().lower().startswith('# checked')
    return False


def read_ref_txt(path: Path) -> str:
    """bench/ref/<stem>.txt: plain text, '#' lines ignored, a leading line
    timestamp ('[0:05.2]') and a leading speaker label ('Doctor:', 'P:') stripped."""
    lines = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip().startswith('#'):
            continue
        line = _LINE_TS.sub('', line)
        lines.append(_SPEAKER.sub('', line))
    return ' '.join(lines)


_NUM_CONT = re.compile(r'\d$'), re.compile(r'^[.,/:]\d'), re.compile(r'\d[.,/:]$'), re.compile(r'^\d')


def merged_words(d: dict) -> List[Tuple[str, float, float]]:
    """Words joined back into whitespace-delimited words for text scoring.
    Whisper-family runners start every new word with a space and split "7.0"
    into " 7" + ".0"; runners without leading spaces get only the number
    continuation rule ("7" + ".0", "135" + "/88")."""
    ws = [w for s in d['segments'] for w in s['words'] if w['w'].strip()]
    spaced = bool(ws) and sum(w['w'][:1].isspace() for w in ws) > 0.5 * len(ws)
    out: List[List] = []
    for w in ws:
        txt = w['w']
        if out:
            prev, cur = out[-1][0].rstrip(), txt.lstrip()
            cont = (spaced and not txt[:1].isspace()) or \
                (_NUM_CONT[0].search(prev) and _NUM_CONT[1].match(cur)) or \
                (_NUM_CONT[2].search(prev) and _NUM_CONT[3].match(cur))
            if cont:
                out[-1][0] += txt
                out[-1][2] = max(out[-1][2], w['end'])
                continue
        out.append([txt, w['start'], w['end']])
    return [(t, s, e) for t, s, e in out]


def tokens_of(d: dict) -> Tuple[List[Tok], bool]:
    """Normalised tokens of a transcript, each with the source word's times
    when the tag has word timestamps."""
    if d['_wt']:
        return [(t, s, e) for (w, s, e) in merged_words(d) for t in norm_chunk(w)], True
    return [(t, None, None) for s in d['segments'] for t in norm_chunk(s['text'])], False


def span_flags(toks: List[Tok], spans: List[Span]) -> List[bool]:
    """True for tokens whose word interval overlaps any gold span."""
    return [any(t[1] is not None and min(t[2], g1) - max(t[1], g0) > 0 for g0, g1 in spans) for t in toks]


def wer_pair(ref: List[Tok], hyp: List[Tok], ref_flags: Optional[List[bool]],
             hyp_flags: Optional[List[bool]]) -> Optional[dict]:
    if not ref:
        return None
    if not hyp:
        # The runner wrote no words for this file (failed chunk, empty decoder
        # output, a truncated file recovered as "segments": []). Dropping it
        # would *improve* the pooled WER and shrink n_files without a trace, so
        # it stays in the pool as 100% deletions; wer_block counts it as empty.
        res = {'n_ref': len(ref), 'n_hyp': 0, 'hits': 0, 'sub': 0, 'del': len(ref), 'ins': 0, 'wer': 1.0}
        if ref_flags is not None and len(ref_flags) == len(ref):
            n_in = sum(ref_flags)
            res.update(n_in_span=n_in, err_in_span=n_in, wer_in_span=1.0 if n_in else None)
        return res
    # https://jitsi.github.io/jiwer/usage/ : process_words(reference, hypothesis,
    # reference_transform=, hypothesis_transform=) -> WordOutput with wer, hits,
    # substitutions, insertions, deletions, alignments (list per sentence of
    # AlignmentChunk(type, ref_start_idx, ref_end_idx, hyp_start_idx, hyp_end_idx),
    # type in equal / substitute / insert / delete)
    tr = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    out = jiwer.process_words(' '.join(t[0] for t in ref), ' '.join(t[0] for t in hyp),
                              reference_transform=tr, hypothesis_transform=tr)
    res = {'n_ref': len(ref), 'n_hyp': len(hyp), 'hits': out.hits, 'sub': out.substitutions,
           'del': out.deletions, 'ins': out.insertions, 'wer': out.wer}
    side = 'hyp' if hyp_flags is not None else ('ref' if ref_flags is not None else None)
    if side is None:
        return res
    flags = hyp_flags if side == 'hyp' else ref_flags
    if len(flags) != (len(hyp) if side == 'hyp' else len(ref)):
        return res

    def neighbour(lo: int) -> bool:                 # empty range on the flagged side: look at the words around it
        return bool((lo - 1 >= 0 and flags[lo - 1]) or (lo < len(flags) and flags[lo]))

    # jiwer merges runs of words into one chunk (equal/substitute chunks advance
    # both sides in lockstep), so the in-span test is made per word, not per chunk.
    n_in = err_in = 0
    for c in out.alignments[0]:
        if c.type in ('equal', 'substitute'):
            for r_i, h_i in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                if flags[h_i] if side == 'hyp' else flags[r_i]:
                    n_in += 1
                    err_in += c.type == 'substitute'
        elif c.type == 'insert':                    # hyp words with no reference counterpart
            for h_i in range(c.hyp_start_idx, c.hyp_end_idx):
                if flags[h_i] if side == 'hyp' else neighbour(c.ref_start_idx):
                    err_in += 1
        elif c.type == 'delete':                    # reference words the hypothesis missed
            for r_i in range(c.ref_start_idx, c.ref_end_idx):
                if flags[r_i] if side == 'ref' else neighbour(c.hyp_start_idx):
                    n_in += 1
                    err_in += 1
    res.update(n_in_span=n_in, err_in_span=err_in, wer_in_span=(err_in / n_in) if n_in else None)
    return res


def wer_block(data: Dict[str, dict], gold: Dict[str, List[Tuple[dict, Span]]],
              refs: Dict[str, Tuple[List[Tok], bool]], source: str) -> Optional[dict]:
    """Pooled WER of the tag's transcripts against refs {stem: (tokens, has_times)}."""
    if jiwer is None:
        return {'source': source, 'error': 'jiwer not installed'}
    per_file, tot = {}, Counter()
    for stem in sorted(set(data) & set(refs)):
        hyp, hyp_t = tokens_of(data[stem])
        ref, ref_t = refs[stem]
        spans = [g for _, g in gold.get(stem, [])]
        r = wer_pair(ref, hyp, span_flags(ref, spans) if (ref_t and spans) else None,
                     span_flags(hyp, spans) if (hyp_t and spans) else None)
        if r is None:
            continue
        per_file[stem] = r
        for key in ('n_ref', 'hits', 'sub', 'del', 'ins', 'n_in_span', 'err_in_span'):
            tot[key] += r.get(key, 0) or 0
    if not per_file:
        return {'source': source, 'n_files': 0, 'wer': None}
    n = tot['n_ref']
    return {
        'source': source, 'n_files': len(per_file), 'n_ref_words': n,
        'n_empty_files': sum(r['n_hyp'] == 0 for r in per_file.values()),   # scored as 100% deletions, see wer_pair
        'wer': (tot['sub'] + tot['del'] + tot['ins']) / n if n else None,
        'sub_rate': tot['sub'] / n if n else None, 'del_rate': tot['del'] / n if n else None,
        'ins_rate': tot['ins'] / n if n else None,
        'sub': tot['sub'], 'del': tot['del'], 'ins': tot['ins'],
        'n_in_span_words': tot['n_in_span'],
        'wer_in_span': tot['err_in_span'] / tot['n_in_span'] if tot['n_in_span'] else None,
        'files': per_file,
    }


# --------------------------------------------------------------------------- #
# one tag
# --------------------------------------------------------------------------- #

def analyse(tag: str, files: Dict[str, Path], gold: Dict[str, List[Tuple[dict, Span]]], k: int,
            ref_gold: Dict[str, Tuple[List[Tok], bool]], ref_silver: Optional[Dict[str, Tuple[List[Tok], bool]]],
            ref_silver_tag: Optional[str]) -> dict:
    data: Dict[str, dict] = {}
    unreadable: List[str] = []
    for stem, p in files.items():                   # a truncated JSON (runner killed mid-write) must not abort the tag
        try:
            data[stem] = load_transcript(p)
        except Exception as e:
            unreadable.append(stem)
            print(f'{tag}: skipping {p.name}: {e!r}', file=sys.stderr)
    # a transcript that loaded but has no words at all (the unreadable list only
    # catches JSON that fails to load); wer_pair scores it as 100% deletions
    empty = sorted(stem for stem, d in data.items() if not tokens_of(d)[0])
    if empty:
        print(f'{tag}: {len(empty)} transcript(s) with no words, scored as 100% deletions in WER: {empty}', file=sys.stderr)
    R: dict = {'tag': tag, 'k': k, 'n_files': len(data), 'unreadable_files': unreadable, 'empty_files': empty,
               'word_timestamps': bool(data) and all(d['_wt'] for d in data.values()),
               'n_untimed_words': sum(d['_n_untimed'] for d in data.values()),
               'n_zero_duration_words': sum(d['_n_zero'] for d in data.values())}

    # 1. RTF
    pairs = [(float(d['seconds']), float(d['duration'])) for d in data.values()
             if finite(d.get('seconds')) and finite(d.get('duration')) and d['duration'] > 0]
    R['audio_seconds'] = sum(p[1] for p in pairs)
    R['transcribe_seconds'] = sum(p[0] for p in pairs)
    R['rtf'] = R['transcribe_seconds'] / R['audio_seconds'] if R['audio_seconds'] else None
    R['rtf_per_file_mean'] = mean([s / dd for s, dd in pairs])

    # 2-4. ceilings, offsets, word fractions (first pass)
    seg_b, sen_b, mer_b, su_b = [], [], [], []
    offs, best_n, su_offs = [], [], []
    f_start, f_end, b_start, b_end = [], [], [], []
    spans_out, missing = [], []
    units_cache: Dict[str, Tuple[list, list]] = {}
    for stem, items in sorted(gold.items()):
        d = data.get(stem)
        if d is None:
            missing.append(stem)
            continue
        sents = sentences(d) if d['_wt'] else []
        su = serve_units(d) if (serving is not None and d['_wt']) else []
        units_cache[stem] = (sents, su)
        words = [(w['w'], w['start'], w['end']) for s in d['segments'] for w in s['words'] if w['end'] > w['start']]
        for row, g in items:
            rec: dict = {'question_id': row.get('question_id'), 'stem': stem, 'gold': [g[0], g[1]]}
            rec['segment'] = max((tiou(g, (s['start'], s['end'])) for s in d['segments']), default=0.0)
            seg_b.append(rec['segment'])
            if sents:
                rec['sentence'] = max((tiou(g, (u[0], u[1])) for u in sents), default=0.0)
                mb, mspan = best_merge(sents, g, k)
                rec['merge'] = mb
                sen_b.append(rec['sentence']); mer_b.append(mb)
                if mspan:
                    rec['best_merge'] = [mspan[0], mspan[1], mspan[2]]
                    rec['offset'] = [g[0] - mspan[0], g[1] - mspan[1]]
                    offs.append((rec['offset'][0], rec['offset'][1])); best_n.append(mspan[2])
            if su:
                sb, sspan = best_merge(su, g, k)
                rec['serve_units_merge'] = sb
                su_b.append(sb)
                if sspan:
                    su_offs.append((g[0] - sspan[0], g[1] - sspan[1]))
            if words:
                i = nearest_word(g[0], words, later=True)
                j = nearest_word(g[1], words, later=False)
                ws, we = words[i][1], words[i][2]
                rec['f_start'] = (g[0] - ws) / (we - ws)
                rec['first_word'] = words[i][0].strip()
                ws, we = words[j][1], words[j][2]
                rec['f_end'] = (g[1] - ws) / (we - ws)
                rec['last_word'] = words[j][0].strip()
                rec['d_start'] = min((w[1] - g[0] for w in words), key=abs) * -1   # gold - nearest word start
                rec['d_end'] = min((w[2] - g[1] for w in words), key=abs) * -1     # gold - nearest word end
                f_start.append(rec['f_start']); f_end.append(rec['f_end'])
                b_start.append(rec['d_start']); b_end.append(rec['d_end'])
            spans_out.append(rec)

    # second pass: shifted ceilings with the fitted and the served offsets
    fit = (q([o[0] for o in offs], .5), q([o[1] for o in offs], .5)) if offs else None
    su_fit = (q([o[0] for o in su_offs], .5), q([o[1] for o in su_offs], .5)) if su_offs else None
    served = (serving.START_OFFSET, serving.END_OFFSET) if serving is not None else None
    shift_fit, shift_serve, su_shift_fit, su_shift_serve = [], [], [], []
    for stem, items in sorted(gold.items()):
        if stem not in units_cache:
            continue
        sents, su = units_cache[stem]
        for _, g in items:
            if sents and fit:
                shift_fit.append(best_merge(sents, g, k, *fit)[0])
            if sents and served:
                shift_serve.append(best_merge(sents, g, k, *served)[0])
            if su and su_fit:
                su_shift_fit.append(best_merge(su, g, k, *su_fit)[0])
            if su and served:
                su_shift_serve.append(best_merge(su, g, k, *served)[0])

    R['n_gold_spans'] = len(seg_b)
    R['missing_gold_files'] = missing
    R['ceiling'] = {
        'n': len(seg_b), 'segment': mean(seg_b), 'sentence': mean(sen_b), f'merge{k}': mean(mer_b),
        f'merge{k}_below_half': sum(x < 0.5 for x in mer_b) if mer_b else None,
        f'merge{k}_shift_fit': mean(shift_fit), f'merge{k}_shift_serve': mean(shift_serve),
        f'serve_units_merge{k}': mean(su_b), f'serve_units_merge{k}_shift_fit': mean(su_shift_fit),
        f'serve_units_merge{k}_shift_serve': mean(su_shift_serve),
        'fit_offsets': list(fit) if fit else None, 'serve_units_fit_offsets': list(su_fit) if su_fit else None,
        'served_offsets': list(served) if served else None,
    }
    R['offset'] = offset_block(offs, best_n)
    R['offset']['serve_units'] = offset_block(su_offs, [])
    R['first_word_fraction'] = fraction_stats(f_start)
    R['last_word_fraction'] = fraction_stats(f_end)
    R['boundary_ms'] = {'start': boundary_stats(b_start), 'end': boundary_stats(b_end)}

    # 5. WER: `wer` / `wer_silver` are the numbers (or null) per the README
    # schema; the S/D/I counts, in-span WER and per-file rows sit under *_detail.
    R['wer_detail'] = wer_block(data, gold, ref_gold, 'bench/ref') if ref_gold else None
    if ref_silver is None:
        R['wer_silver_detail'] = None
    elif tag == ref_silver_tag:
        R['wer_silver_detail'] = {'source': f'silver:{ref_silver_tag}', 'self': True, 'wer': None}
    else:
        R['wer_silver_detail'] = wer_block(data, gold, ref_silver, f'silver:{ref_silver_tag}')
    R['wer'] = (R['wer_detail'] or {}).get('wer')
    R['wer_silver'] = (R['wer_silver_detail'] or {}).get('wer')
    R['spans'] = spans_out
    return R


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def _wer_cells(block: Optional[dict]) -> Tuple[str, str, str]:
    if not block or block.get('wer') is None:
        if block and block.get('self'):
            return 'ref', '-', '-'
        return '-', '-', '-'
    sdi = f"{pct(block['sub_rate'])}/{pct(block['del_rate'])}/{pct(block['ins_rate'])}"
    n_empty = block.get('n_empty_files') or 0
    n_cell = f"n={block['n_files']}" + (f', {n_empty} empty' if n_empty else '')
    return f"{100 * block['wer']:.2f}% ({n_cell})", sdi, \
        '-' if block.get('wer_in_span') is None else f"{100 * block['wer_in_span']:.2f}%"


def tables(results: List[dict], k: int) -> str:
    out = []
    out.append(f'### A. timestamps: oracle-selection tIoU ceiling and edge offsets (gold minus model, seconds)\n')
    out.append(f'| tag | files | RTF | seg | sent | merge<={k} | +fit shift | +served shift | serve units m<={k} | serve +served | start med (p25,p75) | end med (p25,p75) | fit (start,end) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|')
    for R in results:
        c, o = R['ceiling'], R['offset']
        so = '-' if o.get('n', 0) == 0 else f"{fmt(o['start_median'], 2, True)} ({fmt(o['start_p25'], 2, True)},{fmt(o['start_p75'], 2, True)})"
        eo = '-' if o.get('n', 0) == 0 else f"{fmt(o['end_median'], 2, True)} ({fmt(o['end_p25'], 2, True)},{fmt(o['end_p75'], 2, True)})"
        fitc = '-' if not c['fit_offsets'] else f"{fmt(c['fit_offsets'][0], 2, True)},{fmt(c['fit_offsets'][1], 2, True)}"
        out.append(f"| {R['tag']} | {R['n_files']} | {fmt(R['rtf'])} | {fmt(c['segment'])} | {fmt(c['sentence'])} | "
                   f"{fmt(c[f'merge{k}'])} | {fmt(c[f'merge{k}_shift_fit'])} | {fmt(c[f'merge{k}_shift_serve'])} | "
                   f"{fmt(c[f'serve_units_merge{k}'])} | {fmt(c[f'serve_units_merge{k}_shift_serve'])} | {so} | {eo} | {fitc} |")
    out.append('')
    out.append('### B. where the annotated edges fall inside the nearest model word, distance to the nearest word boundary, WER\n')
    out.append('f = (gold - word_start) / (word_end - word_start): <0 before the word, 0..1 inside, >1 after. '
               'bnd = |gold start - nearest word start| and |gold end - nearest word end|.\n')
    out.append('| tag | f_start med [p10,p90] | start inside | f_end med [p10,p90] | end inside | bnd start med ms (<=20ms) | bnd end med ms (<=20ms) | WER ref | S/D/I | WER in-span | WER silver | S/D/I | in-span |')
    out.append('|---|---|---:|---|---:|---|---|---|---|---|---|---|---|')
    for R in results:
        fs, fe, bs, be = R['first_word_fraction'], R['last_word_fraction'], R['boundary_ms']['start'], R['boundary_ms']['end']
        fsc = '-' if not fs['n'] else f"{fmt(fs['median'], 2, True)} [{fmt(fs['p10'], 2, True)},{fmt(fs['p90'], 2, True)}]"
        fec = '-' if not fe['n'] else f"{fmt(fe['median'], 2, True)} [{fmt(fe['p10'], 2, True)},{fmt(fe['p90'], 2, True)}]"
        bsc = '-' if not bs['n'] else f"{bs['abs_median_ms']:.0f} ({pct(bs['within_20ms'])})"
        bec = '-' if not be['n'] else f"{be['abs_median_ms']:.0f} ({pct(be['within_20ms'])})"
        w1, s1, i1 = _wer_cells(R.get('wer_detail'))
        w2, s2, i2 = _wer_cells(R.get('wer_silver_detail'))
        out.append(f"| {R['tag']} | {fsc} | {pct(fs.get('inside_word'))} | {fec} | {pct(fe.get('inside_word'))} | "
                   f"{bsc} | {bec} | {w1} | {s1} | {i1} | {w2} | {s2} | {i2} |")
    out.append('')
    out.append('### C. histogram of f per tag (counts)\n')
    out.append('| tag | edge | ' + ' | '.join(HIST_LABELS) + ' | n |')
    out.append('|---|---|' + '---:|' * len(HIST_LABELS) + '---:|')
    for R in results:
        for name, blk in (('start', R['first_word_fraction']), ('end', R['last_word_fraction'])):
            if not blk['n']:
                continue
            out.append(f"| {R['tag']} | {name} | " + ' | '.join(str(blk['hist'][l]) for l in HIST_LABELS) + f" | {blk['n']} |")
    return '\n'.join(out)


def print_verbose(R: dict, n: int = 8) -> None:
    print(f"\n{R['tag']}: untimed words {R['n_untimed_words']}, zero-duration words {R['n_zero_duration_words']}, "
          f"best-merge sizes {R['offset'].get('best_merge_sizes')}")
    if R['missing_gold_files']:
        print(f"  gold conversations without a transcript: {len(R['missing_gold_files'])}")
    if R.get('unreadable_files'):
        print(f"  transcripts that failed to load (skipped): {R['unreadable_files']}")
    if R.get('empty_files'):
        print(f"  transcripts with no words (WER counts them as 100% deletions): {R['empty_files']}")
    worst = sorted((s for s in R['spans'] if 'merge' in s), key=lambda s: s['merge'])[:n]
    if worst:
        print('  worst gold spans even with oracle selection:')
        for s in worst:
            print(f"    {s['merge']:.2f} {s['stem']:<22} [{s['gold'][0]:.2f}-{s['gold'][1]:.2f}] "
                  f"first={s.get('first_word')!r} f={fmt(s.get('f_start'), 2, True)}  last={s.get('last_word')!r} f={fmt(s.get('f_end'), 2, True)}")


# --------------------------------------------------------------------------- #
# --suggest-ref: which files a teammate should hand-correct (PROTOCOL.md)
# --------------------------------------------------------------------------- #

def suggest_ref(rows: List[dict], tags: Dict[str, Dict[str, Path]], args) -> None:
    by_stem: Dict[str, dict] = defaultdict(lambda: {'digit': 0, 'n': 0, 'ex': []})
    for r in rows:
        s = by_stem[stem_of(r['transcript_id'])]
        s['n'] += 1
        if re.search(r'\d', r['question']):
            s['digit'] += 1
            s['ex'].append(r['question'])
    dur: Dict[str, float] = {}
    src = args.draft if args.draft in tags else (next(iter(tags)) if tags else None)
    if src:
        for stem, p in tags[src].items():
            dur[stem] = float(common.read_transcript(p).get('duration') or 0.0)
    for stem in by_stem:
        if stem not in dur and case_utils is not None:
            try:
                dur[stem] = case_utils.audio_duration_seconds(case_utils.load_sample_audio(f'{stem}.mp3')) or 0.0
            except Exception:
                dur[stem] = 0.0
    ranked = sorted(by_stem, key=lambda s: (-by_stem[s]['digit'], dur.get(s, 0.0)))
    chosen, total = [], 0.0
    for s in ranked:
        if len(chosen) >= args.suggest_ref:
            break
        if total + dur.get(s, 0.0) > args.ref_budget:
            continue
        chosen.append(s)
        total += dur.get(s, 0.0)
    print(f'{len(chosen)} conversations to hand-correct ({total / 60:.1f} min of audio, budget {args.ref_budget / 60:.0f} min), '
          f'ranked by questions containing digits, then shortest first:\n')
    print('| stem | audio s | questions with digits | example |')
    print('|---|---:|---:|---|')
    for s in chosen:
        ex = by_stem[s]['ex'][0] if by_stem[s]['ex'] else ''
        print(f"| {s} | {dur.get(s, 0.0):.0f} | {by_stem[s]['digit']}/{by_stem[s]['n']} | {ex} |")
    if args.draft:
        if args.draft not in tags:
            print(f'\n--draft {args.draft}: no transcripts with that tag; nothing written')
            return
        args.drafts_dir.mkdir(parents=True, exist_ok=True)
        for s in chosen:
            p = tags[args.draft].get(s)
            if p is None:
                print(f'  no {args.draft} transcript for {s}')
                continue
            out = args.drafts_dir / f'{s}.txt'
            if out.exists() and not args.force:
                print(f'  keep {out} (use --force to overwrite)')
                continue
            d = load_transcript(p)
            lines = [f'# draft from {args.draft}: correct the words, make the first line "# checked", save as bench/ref/{s}.txt']
            for seg in d['segments']:                       # same [m:ss.s] line format as the pre-filled bench/ref files
                if seg['text'].strip():
                    total_ds = round(seg['start'] * 10)         # deciseconds, rounded before splitting into m/s
                    mm, ss = divmod(total_ds, 600)
                    lines.append(f"[{mm}:{ss / 10:04.1f}] {seg['text'].strip()}")
            out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f'  wrote {out}')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--transcripts', type=Path, default=common.DEFAULT_OUT, help='transcripts/ directory')
    ap.add_argument('--audio-dir', type=Path, default=common.DEFAULT_AUDIO, help='data/audio, for the file stems')
    ap.add_argument('--csv', type=Path, default=DEFAULT_CSV, help='question_train.csv with gold spans')
    ap.add_argument('--tags', default='', help='comma-separated tags (default: every tag found)')
    ap.add_argument('--ref', default=None, help='silver reference tag for WER, e.g. canary-qwen-2.5b')
    ap.add_argument('--ref-dir', type=Path, default=DEFAULT_REF, help='hand-corrected <stem>.txt files')
    ap.add_argument('--include-unchecked', action='store_true',
                    help='score ref files without a "# checked" first line too (they are Whisper text: WER 0 vs large-v3)')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_RESULTS, help='where <tag>.json and summary.md go')
    ap.add_argument('--k', type=int, default=4, help='max sentences merged (span_ceiling.py --k)')
    ap.add_argument('--no-json', action='store_true', help='print only, write nothing')
    ap.add_argument('--verbose', action='store_true', help='worst spans and diagnostics per tag')
    ap.add_argument('--suggest-ref', type=int, default=0, metavar='N',
                    help='print the N conversations to hand-correct for bench/ref and exit')
    ap.add_argument('--ref-budget', type=float, default=900.0, help='audio seconds budget for --suggest-ref')
    ap.add_argument('--draft', default=None, help='with --suggest-ref: write drafts from this tag')
    ap.add_argument('--drafts-dir', type=Path, default=DEFAULT_DRAFTS)
    ap.add_argument('--force', action='store_true', help='overwrite existing drafts')
    args = ap.parse_args()

    if case_utils is None:
        print(f'note: utils.py not importable ({_UTILS_ERR!r}); using a local gold reader', file=sys.stderr)
    if serving is None:
        print(f'note: model.py not importable ({_SERVE_ERR!r}); serve-units columns will be empty', file=sys.stderr)
    if jiwer is None:
        print('note: jiwer not installed; WER will be empty (pip install "jiwer>=4,<5")', file=sys.stderr)
    if num2words is None:
        print('note: num2words not installed; digits are compared as digits (pip install num2words)', file=sys.stderr)

    tags = discover_tags(args.transcripts, args.audio_dir)
    rows = read_rows(args.csv) if args.csv.exists() else []
    if args.suggest_ref:
        suggest_ref(rows, tags, args)
        return 0
    if args.tags:
        wanted = [t.strip() for t in args.tags.split(',') if t.strip()]
        unknown = [t for t in wanted if t not in tags]
        if unknown:
            print(f'no transcripts for tag(s) {unknown}; available: {sorted(tags)}', file=sys.stderr)
        tags = {t: tags[t] for t in wanted if t in tags}
    if not tags:
        print(f'no transcripts found under {args.transcripts}', file=sys.stderr)
        return 1
    gold = gold_by_stem(rows)
    n_gold = sum(len(v) for v in gold.values())

    ref_gold: Dict[str, Tuple[List[Tok], bool]] = {}
    ref_checked: List[str] = []
    n_ref_files = n_checked = 0
    if args.ref_dir.is_dir():
        for p in sorted(args.ref_dir.glob('*.txt')):
            n_ref_files += 1
            checked = ref_is_checked(p)
            n_checked += checked
            if not checked and not args.include_unchecked:
                continue
            toks = [(t, None, None) for t in norm_chunk(read_ref_txt(p))]
            if toks:
                ref_gold[p.stem] = (toks, False)
                if checked:
                    ref_checked.append(p.stem)
    # The "# checked" marker is taken on trust, but a checked file whose words
    # still equal one model's output was never corrected (bench/ref/
    # conversation_sample_4.txt on 2026-09-17: large-v3 text with two
    # punctuation edits). Counting it scores that tag WER 0 and every other tag
    # against Whisper, so it is dropped with a warning. Only the checked stems
    # are compared, at most n_checked x n_tags transcript reads.
    ref_dropped: Dict[str, str] = {}
    for stem in ref_checked:
        words = [t[0] for t in ref_gold[stem][0]]
        for tag in sorted(tags):
            p = tags[tag].get(stem)
            if p is None:
                continue
            try:
                hyp = [t[0] for t in tokens_of(load_transcript(p))[0]]
            except Exception:                       # unreadable transcript: analyse() reports it
                continue
            if hyp == words:
                print(f'warning: {stem}.txt is marked "# checked" but is identical to {tag} text after '
                      f'normalisation; not counting it (correct the words or remove the marker)', file=sys.stderr)
                ref_dropped[stem] = tag
                del ref_gold[stem]
                break
    ref_silver = None
    if args.ref:
        if args.ref not in tags:
            print(f'--ref {args.ref}: no transcripts with that tag; silver WER skipped', file=sys.stderr)
        else:
            ref_silver = {}
            for stem, p in tags[args.ref].items():
                try:
                    ref_silver[stem] = tokens_of(load_transcript(p))
                except Exception as e:              # same failure mode as in analyse(): skip the file, keep the run
                    print(f'--ref {args.ref}: skipping {p.name}: {e!r}', file=sys.stderr)

    print(f'{len(tags)} tag(s), {n_gold} gold spans over {len(gold)} conversations, K={args.k}, '
          f'ref: {n_checked} checked of {n_ref_files} file(s) in {args.ref_dir}'
          + (f', {len(ref_dropped)} dropped as uncorrected model text ({", ".join(sorted(ref_dropped))})' if ref_dropped else '')
          + (' (unchecked included)' if args.include_unchecked else '')
          + (f', silver ref {args.ref}' if ref_silver else '') + '\n')
    results, failed = [], []
    for tag in sorted(tags):
        try:
            R = analyse(tag, tags[tag], gold, args.k, ref_gold, ref_silver, args.ref)
        except Exception:                           # one bad tag must not cost the others their JSON and summary.md
            traceback.print_exc()
            print(f'tag {tag} failed, continuing', file=sys.stderr)
            failed.append(tag)
            continue
        results.append(R)
        if not args.no_json:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f'{tag}.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
    if not results:
        print(f'every tag failed: {failed}', file=sys.stderr)
        return 1
    md = tables(results, args.k)
    print(md)
    if args.verbose:
        for R in results:
            print_verbose(R)
    if not args.no_json:
        (args.out_dir / 'summary.md').write_text(md + '\n', encoding='utf-8')
        print(f'\nwrote {args.out_dir}/<tag>.json and summary.md')
    if failed:                                      # outputs are on disk; the exit code still flags the job step
        print(f'\n{len(failed)} tag(s) failed (traceback above): {failed}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
