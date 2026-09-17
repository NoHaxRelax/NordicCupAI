"""Prompt / output variants for the offline answering bench (bench/llm/bench.py).

A variant is a named function

    build(question: str, units: List[model.Unit]) -> Prompt(system, user, schema, postprocess)

where ``postprocess(out, units, words, duration) -> (answer: bool, span | None)``
maps the JSON the model returned back to seconds. ``VARIANTS`` maps the CLI
name to the function. The system prompt, JSON schema, unit builder, transcript
renderer and id-to-seconds rule are imported from model.py (the serving code),
so the bench measures exactly what serves; a variant only changes what is
asked and how the answer is read back.

Variants
  units           the current model.py design: {"quote", "answer", "segments"};
                  span = model.span_from_ids (contiguous run of cited unit ids,
                  plus START_OFFSET / END_OFFSET). Tag questions get the same
                  parenthetical hint model.ask_llm adds.
  units-claim     same output; tag questions ("..., didn't it?") are rewritten
                  to a declarative "Claim: ..." and the instruction says
                  "decide whether the claim is established" (research/04 section 3).
  words           the model returns the first and last few words of the
                  evidence passage verbatim (plus the unit id that holds it);
                  the phrase is located in the word stream with difflib on a
                  window, preferring the cited unit; span = first-word start /
                  last-word end + START_OFFSET / END_OFFSET.
  units-nooffset  'units' with the edge offsets removed (ablation).

Install (Linux venv): nothing beyond model.py's own imports
    pip install requests
Example
    python -c "import sys; sys.path.insert(0,'bench/llm'); import prompts; print(prompts.VARIANTS.keys())"
Caveats
  - model.py is imported, so its environment variables apply (START_OFFSET,
    END_OFFSET, PAUSE_SPLIT). Set them before running the bench if you want
    other values; the bench records the values in effect.
  - 'units-nooffset' removes the offsets after model.span_from_ids has applied
    and clamped them, so a cited unit that touches the very end of the audio
    can differ from the raw unit end by up to END_OFFSET. Negligible for an
    ablation; noted so nobody hunts for it.
"""
from __future__ import annotations

import bisect
import difflib
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

HERE = Path(__file__).resolve().parent            # bench/llm
CASE = HERE.parent.parent                         # medical-appointment
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

from model import (  # noqa: E402  (serving code; see module docstring)
    END_OFFSET, SCHEMA, START_OFFSET, SYSTEM, Unit, Word, make_units,
    render_transcript, span_from_ids, _TAG as TAG,
)

__all__ = ['Prompt', 'VARIANTS', 'make_units', 'render_transcript', 'claim_text',
           'units_question']

Span = Tuple[float, float]
Postprocess = Callable[[dict, List[Unit], List[Word], float], Tuple[bool, Optional[Span]]]


class Prompt(NamedTuple):
    system: str
    user: str
    schema: dict
    postprocess: Postprocess
    demos: Optional[List[Tuple[str, str]]] = None   # (user, assistant) turns sent before the real user turn


# --------------------------------------------------------------------------- #
# Question rewriting
# --------------------------------------------------------------------------- #

# First words that are safe to lower-case when a tag question becomes a claim
# ("The lipid profile ..." -> "the lipid profile ..."); anything else (HbA1c,
# a drug name, a proper noun) keeps its capital.
_LOWER_FIRST = {
    'the', 'a', 'an', 'both', 'this', 'that', 'these', 'those', 'there', 'it',
    'he', 'she', 'they', 'his', 'her', 'their', 'its', 'no', 'some', 'all',
    'one', 'only', 'after', 'before', 'during', 'at', 'in', 'on', 'alongside',
    'according', 'nothing', 'every', 'each', 'another', 'any',
}


def strip_tag(question: str) -> Optional[str]:
    """'The X came back normal, didn't it?' -> 'The X came back normal', or None
    if the question carries no tag suffix (same regex as model.ask_llm)."""
    q = question.strip()
    m = TAG.search(q)
    if not m:
        return None
    return q[:m.start()].strip().rstrip(',').strip()


def claim_text(question: str) -> str:
    """Declarative form used by 'units-claim': a tag question becomes
    'CLAIM: the x came back normal.'; any other question stays a question."""
    stem = strip_tag(question)
    if stem is None:
        return f'QUESTION: {question.strip()}'
    first, _, rest = stem.partition(' ')
    if first.lower() in _LOWER_FIRST:
        stem = (first.lower() + ' ' + rest).strip()
    return f'CLAIM: {stem.rstrip(".")}.'


def units_question(question: str) -> str:
    """The question exactly as model.ask_llm sends it (tag hint included)."""
    claim = question.strip()
    tag = TAG.search(claim)
    if tag:
        claim = (f'{claim}  (Read this as the plain question: is it established that '
                 f'{claim[:tag.start()].strip().rstrip(",")}?)')
    return f'QUESTION: {claim}'


def _user(transcript: str, ask: str) -> str:
    # Transcript first, question last: the shared prefix is what vLLM's prefix
    # cache reuses across the ten questions of one conversation.
    return f'TRANSCRIPT:\n{transcript}\n\n{ask}'


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

_TAG_SENTENCE = ('Tag questions ("..., right?", "..., didn\'t it?") are ordinary questions; the phrasing\n'
                 'does not hint at the answer.')
_CLAIM_SENTENCE = ('The input is a CLAIM (or a plain QUESTION). Decide whether the transcript establishes\n'
                   'the claim; for a question, whether it establishes a "yes". The wording of the claim\n'
                   'carries no hint about the answer.')

if _TAG_SENTENCE in SYSTEM:
    CLAIM_SYSTEM = SYSTEM.replace(_TAG_SENTENCE, _CLAIM_SENTENCE)
else:                                   # model.SYSTEM was edited; keep working
    CLAIM_SYSTEM = SYSTEM.replace('Return JSON only:', _CLAIM_SENTENCE + '\nReturn JSON only:', 1)

_RETURN_MARK = 'Return JSON only:'
_SYSTEM_HEAD = SYSTEM.split(_RETURN_MARK, 1)[0]

WORDS_SYSTEM = _SYSTEM_HEAD + """Return JSON only:
{"segment": <id of the single utterance that contains the evidence passage; -1 when the answer is no>,
 "first_words": "<the first three words of the evidence passage, copied verbatim from the transcript; \\"\\" when no>",
 "last_words": "<the last three words of the evidence passage, copied verbatim from the transcript; \\"\\" when no>",
 "answer": "yes" | "no"}
The evidence passage is the shortest stretch of speech (usually one sentence, at most three)
that establishes the statement. It may start or end in the middle of an utterance. Copy the
words exactly as written, without the [id] prefix."""

WORDS_SCHEMA = {
    'type': 'object',
    'properties': {
        'segment': {'type': 'integer'},
        'first_words': {'type': 'string'},
        'last_words': {'type': 'string'},
        'answer': {'type': 'string', 'enum': ['yes', 'no']},
    },
    'required': ['segment', 'first_words', 'last_words', 'answer'],
}


# --------------------------------------------------------------------------- #
# Post-processing helpers
# --------------------------------------------------------------------------- #

def _is_yes(out: dict) -> bool:
    return str(out.get('answer', '')).strip().lower() == 'yes'


def _int_list(v) -> List[int]:
    ids: List[int] = []
    for i in (v or []):
        if isinstance(i, bool):
            continue
        if isinstance(i, (int, float)):
            ids.append(int(i))
        elif isinstance(i, str) and i.strip().lstrip('-').isdigit():
            ids.append(int(i.strip()))
    return ids


def _int_or(v, default: int) -> int:
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().lstrip('-').isdigit():
        return int(v.strip())
    return default


def _clamp(start: float, end: float, duration: float) -> Span:
    # Same clamping rule as model.span_from_ids.
    start = max(0.0, min(start, duration))
    end = max(start + 0.05, min(end, duration))
    return round(start, 2), round(end, 2)


def _units_post(out: dict, units: List[Unit], words: List[Word], duration: float,
                offsets: bool = True) -> Tuple[bool, Optional[Span]]:
    """Mirror of the per-question logic in model.answer_all."""
    yes = _is_yes(out)
    # Same anchoring as model.answer_all: the quote's unit first, cited ids
    # only when adjacent to it; falls back to the cited ids when no quote matches.
    import model as _m
    ids = _m.anchor_ids(out, units)
    span = span_from_ids(ids, units, duration)
    # The span is kept for no answers too; bench.py scores both policies
    # (spans on every question vs nulls on no) from the same run.
    if span is not None and not offsets:
        span = _clamp(span[0] - START_OFFSET, span[1] - END_OFFSET, duration)
    return yes, span


# --- 'words': locate a verbatim phrase in the word stream ------------------- #

_NONWORD = re.compile(r"[^a-z0-9']+")
_ID_PREFIX = re.compile(r'^\s*\[\s*\d+\s*\]\s*')
MATCH_MIN = 0.72          # difflib ratio a window must reach to count as found
PREFER_BONUS = 0.15       # added to windows inside the cited unit (+-1 unit)
END_WINDOW = 80           # words searched after the start for the last phrase


def _norm(w: str) -> str:
    return _NONWORD.sub('', w.replace('\x00', '').lower())


def _phrase_tokens(s: str) -> List[str]:
    s = _ID_PREFIX.sub('', str(s or ''))
    return [t for t in (_norm(x) for x in s.split()) if t]


def _unit_ranges(units: List[Unit], words: List[Word]) -> Dict[int, Tuple[int, int]]:
    """unit idx -> [first word index, last word index + 1). Units are contiguous
    runs of words, so a word belongs to the last unit starting at or before it."""
    starts = [u.start for u in units]
    ranges: Dict[int, Tuple[int, int]] = {}
    for i, w in enumerate(words):
        k = max(0, bisect.bisect_right(starts, w.start) - 1) if starts else 0
        lo, hi = ranges.get(k, (i, i))
        ranges[k] = (min(lo, i), i + 1)
    return ranges


def _find(phrase: List[str], toks: List[str], lo: int, hi: int,
          prefer: Optional[Tuple[int, int]]) -> Tuple[int, float]:
    """Best window start for `phrase` in toks[lo:hi]; returns (index, raw ratio)
    or (-1, 0). Windows inside `prefer` win ties and near-ties."""
    n = len(phrase)
    if n == 0:
        return -1, 0.0
    target = ' '.join(phrase)
    best_i, best_raw, best_score = -1, 0.0, 0.0
    sm = difflib.SequenceMatcher(None, '', target)
    for i in range(max(0, lo), min(hi, len(toks) - n + 1)):
        cand = ' '.join(toks[i:i + n])
        if not cand.strip():
            continue
        sm.set_seq1(cand)
        if sm.quick_ratio() < MATCH_MIN:
            continue
        raw = sm.ratio()
        score = raw + (PREFER_BONUS if prefer and prefer[0] <= i < prefer[1] else 0.0)
        if score > best_score:
            best_i, best_raw, best_score = i, raw, score
    return best_i, best_raw


def _words_post(out: dict, units: List[Unit], words: List[Word], duration: float
                ) -> Tuple[bool, Optional[Span]]:
    if not _is_yes(out):
        return False, None
    if not words:
        return True, None
    seg = _int_or(out.get('segment'), -1)
    if not (0 <= seg < len(units)):
        seg = -1
    toks = [_norm(w.w) for w in words]
    ranges = _unit_ranges(units, words)

    def rng(a: int, b: int) -> Optional[Tuple[int, int]]:
        # word-index range covering units a..b (clamped), None if unknown
        keys = [k for k in range(max(0, a), min(len(units), b + 1)) if k in ranges]
        if not keys:
            return None
        return min(ranges[k][0] for k in keys), max(ranges[k][1] for k in keys)

    first = _phrase_tokens(out.get('first_words'))
    last = _phrase_tokens(out.get('last_words'))
    prefer = rng(seg - 1, seg + 1) if seg >= 0 else None

    i, raw = _find(first, toks, 0, len(toks), prefer)
    if i < 0 or raw < MATCH_MIN:
        # Phrase not found: fall back to the cited unit, else nothing.
        return True, (span_from_ids([seg], units, duration) if seg >= 0 else None)

    # unit that holds the start word, for the end fallback and the end preference
    unit_of_start = max((k for k, (lo, hi) in ranges.items() if lo <= i < hi), default=seg)
    prefer_end = rng(unit_of_start, unit_of_start + 2) if unit_of_start >= 0 else None
    j, raw_j = _find(last, toks, i, i + END_WINDOW, prefer_end)
    if j >= 0 and raw_j >= MATCH_MIN:
        end_idx = j + len(last) - 1
    elif unit_of_start >= 0 and unit_of_start in ranges:
        end_idx = ranges[unit_of_start][1] - 1          # end of the start's unit
    else:
        end_idx = i + len(first) - 1
    end_idx = max(end_idx, i + len(first) - 1)
    end_idx = min(end_idx, len(words) - 1)
    return True, _clamp(words[i].start + START_OFFSET, words[end_idx].end + END_OFFSET, duration)


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

def units(question: str, units_: List[Unit]) -> Prompt:
    return Prompt(SYSTEM, _user(render_transcript(units_), units_question(question)),
                  SCHEMA, lambda out, u, w, d: _units_post(out, u, w, d, offsets=True))


def units_claim(question: str, units_: List[Unit]) -> Prompt:
    return Prompt(CLAIM_SYSTEM, _user(render_transcript(units_), claim_text(question)),
                  SCHEMA, lambda out, u, w, d: _units_post(out, u, w, d, offsets=True))


def units_nooffset(question: str, units_: List[Unit]) -> Prompt:
    return Prompt(SYSTEM, _user(render_transcript(units_), units_question(question)),
                  SCHEMA, lambda out, u, w, d: _units_post(out, u, w, d, offsets=False))


def words(question: str, units_: List[Unit]) -> Prompt:
    return Prompt(WORDS_SYSTEM, _user(render_transcript(units_), units_question(question)),
                  WORDS_SCHEMA, _words_post)


# --------------------------------------------------------------------------- #
# Few-shot variants: show the annotators' own granularity
# --------------------------------------------------------------------------- #

_STOP = set('the a an is was were are did does do has have had be been being of to for in on at by with '
            'and or any this that it its there they them their he she his her you your we our i me my not no '
            'yes patient doctor conversation mention mentioned discuss discussed correct right isn t didn wasn '
            'about as from into than then which who whom what when where while also still ever any some'.split())
_FEWSHOT_NOTE = ('\nEXAMPLES below show, for other consultations, the exact stretch of speech the annotators marked\n'
                 'as the evidence for a question. Match their granularity: when the fact is completed by the\n'
                 'question that prompted it or by the confirming reply, the marked stretch includes those\n'
                 'utterances too; otherwise it is the single utterance that states the fact.')


def _qtokens(q: str) -> set:
    return {t for t in _NONWORD.split(q.lower()) if t and t not in _STOP}


class FewShot:
    """Callable variant: examples from OTHER training conversations (leave-one-out),
    chosen by question-word overlap, each with the transcript words inside the gold
    span. bench.py calls set_conversation(stem, asr) before each conversation."""

    def __init__(self, base: str, k: int = 12, k_neg: int = 2):
        assert base in ('units', 'words')
        self.base, self.k, self.k_neg = base, k, k_neg
        self.exclude: Optional[str] = None
        self.asr = 'large-v3-turbo'
        self._pool: Optional[List[dict]] = None
        self._pool_asr: Optional[str] = None

    def set_conversation(self, stem: str, asr: str) -> None:
        self.exclude, self.asr = stem, asr

    def pool(self) -> List[dict]:
        if self._pool is not None and self._pool_asr == self.asr:
            return self._pool
        import csv
        import json
        rows = list(csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')))
        words_by: Dict[str, List[dict]] = {}
        pool: List[dict] = []
        for r in rows:
            stem = f"conversation_{r['transcript_id']}"
            if stem not in words_by:
                tf = CASE / 'transcripts' / f'{stem}.{self.asr}.json'
                ws: List[dict] = []
                if tf.exists():
                    for s in json.loads(tf.read_text(encoding='utf-8'))['segments']:
                        ws.extend(s.get('words') or [])
                words_by[stem] = ws
            ex = {'stem': stem, 'q': r['question'].strip(), 'yes': r['answer'] == 'yes', 'text': '',
                  'toks': _qtokens(r['question'])}
            if ex['yes'] and r['evidence_start']:
                a, b = float(r['evidence_start']), float(r['evidence_end'])
                ex['text'] = ' '.join(w['w'].strip() for w in words_by[stem] if a <= (w['start'] + w['end']) / 2 <= b)
            if ex['yes'] and not ex['text']:
                continue
            pool.append(ex)
        self._pool, self._pool_asr = pool, self.asr
        return pool

    def examples(self, question: str) -> str:
        qt = _qtokens(question)
        cands = [e for e in self.pool() if e['stem'] != self.exclude]

        def sim(e):
            u = len(qt | e['toks'])
            return len(qt & e['toks']) / u if u else 0.0
        pos = sorted((e for e in cands if e['yes']), key=lambda e: (-sim(e), e['q']))[:self.k]
        neg = sorted((e for e in cands if not e['yes']), key=lambda e: (-sim(e), e['q']))[:self.k_neg]
        lines = [f'- Q: {e["q"]}  ->  "{e["text"]}"' for e in pos]
        lines += [f'- Q: {e["q"]}  ->  no evidence (answer no)' for e in neg]
        return 'EXAMPLES (other consultations, question -> the stretch marked as evidence):\n' + '\n'.join(lines)

    def __call__(self, question: str, units_: List[Unit]) -> Prompt:
        block = self.examples(question)
        user = f'TRANSCRIPT:\n{render_transcript(units_)}\n\n{block}\n\n{units_question(question)}'
        if self.base == 'words':
            return Prompt(WORDS_SYSTEM + _FEWSHOT_NOTE, user, WORDS_SCHEMA, _words_post)
        return Prompt(SYSTEM + _FEWSHOT_NOTE, user, SCHEMA,
                      lambda out, u, w, d: _units_post(out, u, w, d, offsets=True))


# --------------------------------------------------------------------------- #
# Joint variant: all ten questions of a conversation in one request
# --------------------------------------------------------------------------- #

_JOINT_NOTE = ('\nYou receive ALL questions about this consultation at once and answer them in one JSON object.\n'
               'Different questions usually rest on different utterances: pick, for each question, the utterance\n'
               'that states that question\'s own detail, and reuse an utterance for two questions only when it\n'
               'really is the most specific evidence for both. Answer every question; keep their numbering.')
JOINT_SYSTEM = SYSTEM.replace(_RETURN_MARK, _JOINT_NOTE + '\n' + _RETURN_MARK, 1) \
    .replace('Return JSON only:\n{', 'Return JSON only:\n{"answers": [ for each question, in order, {"q": <question number>,\n  ', 1) \
    .replace(']}', ']} ]}', 1)

JOINT_SCHEMA = {
    'type': 'object',
    'properties': {
        'answers': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'q': {'type': 'integer'},
                    'quote': {'type': 'string'},
                    'answer': {'type': 'string', 'enum': ['yes', 'no']},
                    'segments': {'type': 'array', 'items': {'type': 'integer'}},
                },
                'required': ['q', 'quote', 'answer', 'segments'],
            },
        },
    },
    'required': ['answers'],
}


class Joint:
    """Conversation-level variant: bench.py calls build_all(questions, units) once per
    conversation and split(out, n) to get the per-question dicts, which go through the
    ordinary 'units' post-processing."""

    joint = True

    def build_all(self, questions: List[str], units_: List[Unit]) -> Prompt:
        asks = '\n'.join(f'{i + 1}. {units_question(q)[len("QUESTION: "):]}' for i, q in enumerate(questions))
        user = f'TRANSCRIPT:\n{render_transcript(units_)}\n\nQUESTIONS:\n{asks}'
        return Prompt(JOINT_SYSTEM, user, JOINT_SCHEMA,
                      lambda out, u, w, d: _units_post(out, u, w, d, offsets=True))

    @staticmethod
    def split(out: dict, n: int) -> List[dict]:
        per = [{'quote': '', 'answer': 'no', 'segments': []} for _ in range(n)]
        for item in (out.get('answers') or []):
            try:
                k = int(item.get('q', 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= k < n:
                per[k] = item
        return per

    def __call__(self, question: str, units_: List[Unit]) -> Prompt:   # --print-prompt uses this
        return self.build_all([question], units_)


class JointDemo(Joint):
    """Joint variant with K whole worked conversations as prior chat turns: each demo is
    the transcript plus its ten questions (user turn) and the annotators' answers in the
    same JSON shape, unit ids derived from the gold spans (assistant turn). Demos come
    from the training conversations whose questions overlap the current ones most, never
    from the conversation under test (bench.py calls set_conversation first)."""

    def __init__(self, k: int = 2):
        self.k = k
        self.exclude: Optional[str] = None
        self.asr = 'large-v3-turbo'
        self._pool: Optional[List[dict]] = None
        self._pool_asr: Optional[str] = None

    def set_conversation(self, stem: str, asr: str) -> None:
        self.exclude, self.asr = stem, asr

    def pool(self) -> List[dict]:
        if self._pool is not None and self._pool_asr == self.asr:
            return self._pool
        import csv
        import json
        rows_by: Dict[str, List[dict]] = {}
        for r in csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')):
            rows_by.setdefault(f"conversation_{r['transcript_id']}", []).append(r)
        pool: List[dict] = []
        for stem, rows in rows_by.items():
            tf = CASE / 'transcripts' / f'{stem}.{self.asr}.json'
            if not tf.exists():
                continue
            words: List[Word] = []
            for s in json.loads(tf.read_text(encoding='utf-8'))['segments']:
                for w in s.get('words', []):
                    words.append(Word(w['w'], float(w['start']), float(w['end'])))
                if s.get('words'):
                    words[-1].w += '\x00'
            units_ = make_units(words)
            answers = []
            for i, r in enumerate(rows):
                if r['answer'] == 'yes' and r['evidence_start']:
                    g0, g1 = float(r['evidence_start']), float(r['evidence_end'])
                    ids = [k for k, u in enumerate(units_)
                           if min(u.end, g1) - max(u.start, g0) > 0.3 * max(0.05, u.end - u.start)]
                    if not ids:
                        mid = (g0 + g1) / 2
                        ids = [min(range(len(units_)), key=lambda k: abs((units_[k].start + units_[k].end) / 2 - mid))]
                    answers.append({'q': i + 1, 'quote': units_[ids[0]].text.strip(), 'answer': 'yes', 'segments': ids})
                else:
                    answers.append({'q': i + 1, 'quote': '', 'answer': 'no', 'segments': []})
            qs = [r['question'].strip() for r in rows]
            asks = '\n'.join(f'{i + 1}. {units_question(q)[len("QUESTION: "):]}' for i, q in enumerate(qs))
            pool.append({'stem': stem, 'toks': [_qtokens(q) for q in qs],
                         'user': f'TRANSCRIPT:\n{render_transcript(units_)}\n\nQUESTIONS:\n{asks}',
                         'assistant': json.dumps({'answers': answers}, ensure_ascii=False)})
        self._pool, self._pool_asr = pool, self.asr
        return pool

    def demos_for(self, questions: List[str]) -> List[Tuple[str, str]]:
        qt = [_qtokens(q) for q in questions]

        def sim(e):
            s = 0.0
            for a in qt:
                best = 0.0
                for b in e['toks']:
                    u = len(a | b)
                    best = max(best, len(a & b) / u if u else 0.0)
                s += best
            return s
        cands = [e for e in self.pool() if e['stem'] != self.exclude]
        cands.sort(key=lambda e: (-sim(e), e['stem']))
        return [(e['user'], e['assistant']) for e in cands[:self.k]]

    def build_all(self, questions: List[str], units_: List[Unit]) -> Prompt:
        p = super().build_all(questions, units_)
        system = p.system.replace(_JOINT_NOTE, _JOINT_NOTE +
                                  '\nThe earlier exchanges in this chat are worked examples from other consultations,\n'
                                  'answered exactly the way the annotators did: copy their choice of utterances and\n'
                                  'their granularity (a question plus its answer, a statement plus its number).', 1)
        return Prompt(system, p.user, p.schema, p.postprocess, self.demos_for(questions))


VARIANTS: Dict[str, Callable[[str, List[Unit]], Prompt]] = {
    'units': units,
    'units-claim': units_claim,
    'units-nooffset': units_nooffset,
    'words': words,
    'units-fewshot': FewShot('units'),
    'words-fewshot': FewShot('words'),
    'units-joint': Joint(),
    'units-joint-demo': JointDemo(2),
}


if __name__ == '__main__':          # tiny CPU self-check of the rewrites and the matcher
    for q in ('The lipid profile came back normal, didn\'t it?',
              'Did the HbA1c come out at 43 mmol/mol?',
              'HbA1c was 47 mmol/mol, right?'):
        print(claim_text(q), '|', units_question(q))
    ws = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(
        ' Morning, Dr Fabricius. Your HbA1c is 47 mmol/mol. That is fine.'.split(' ')[1:])]
    us = make_units(ws)
    print(render_transcript(us))
    print(_words_post({'segment': 1, 'first_words': 'Your HbA1c', 'last_words': 'mmol/mol',
                       'answer': 'yes'}, us, ws, 10.0))
