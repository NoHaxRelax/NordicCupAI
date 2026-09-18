"""Experiment v1: timestamped ASR -> sentence units -> local LLM picks unit ids.

Pipeline for one request:

  1. decode MP3 bytes, transcribe with faster-whisper (word timestamps), or load
     the cached transcript when TRANSCRIPT_CACHE=1 and one exists (dev only).
  2. cut the word stream into sentence units at terminal punctuation, segment
     boundaries and pauses longer than PAUSE_SPLIT seconds.
  3. for each question, ask the LLM (JSON schema output) for a verbatim quote,
     yes/no, and the ids of the units that support a yes. The prompt is one of
     the variants in bench/llm/prompts.py (LLM_VARIANT), so what serves is
     exactly what the bench measured.
  4. map ids back to seconds, apply the fitted edge offsets, return.

Everything is configured by environment variables so the same file runs on the
laptop (small model) and on a big GPU (large model) without edits.

  ASR_MODEL          faster-whisper model name           (default large-v3)
  ASR_DEVICE         cuda | cpu                          (default cuda)
  TRANSCRIPT_CACHE   1 = use transcripts/<stem>.<ASR_MODEL>.json when present
  LLM_URL            Ollama base URL, or an OpenAI-compatible base URL ending in
                     /v1 (vLLM)                          (default http://localhost:11434)
  LLM_BACKEND        ollama | vllm                       (default: vllm when LLM_URL
                     ends in /v1, else ollama)
  LLM_MODEL          model name as the server knows it   (default qwen3.5:4b for
                     Ollama; for vLLM '' = the first id of GET /v1/models)
  LLM_VARIANT        prompt variant from bench/llm/prompts.py (default units for
                     Ollama, units-fewshot for vLLM: the served 27B design)
  LLM_TIMEOUT        seconds per LLM request             (default 25)
  LLM_DEADLINE       seconds after the request arrived by which every LLM answer
                     must be in, ASR included            (default 40; the endpoint
                     has 60 s per conversation)
  LLM_MAX_TOKENS     completion budget, vLLM only        (default 600)
  LLM_NO_THINK       vllm | ollama | none: how thinking is switched off on the
                     OpenAI-compatible path              (default ollama when
                     LLM_URL has port 11434, else vllm)
  LLM_FALLBACK_MODEL Ollama tag answered locally when the vLLM request fails,
                     times out or would overrun the deadline (default qwen3:4b
                     on the vllm backend, '' = no fallback)
  LLM_FALLBACK_URL   Ollama base URL for the fallback    (default http://localhost:11434)
  START_RULE         first-word-end | unit-start         (default first-word-end)
  START_OFFSET       seconds added to the start anchor   (default -0.14 for
                     first-word-end, +0.36 for unit-start; fitted on the 39
                     training conversations, see research/07-findings-log.md)
  END_OFFSET         seconds added to unit ends          (default 0.12)
  PAUSE_SPLIT        split a sentence at a pause >= this (default 0.6)
  UNIT_SPLIT         sentence | clause | clause-all: how far the word stream is
                     cut up. sentence = terminal punctuation, segment ends and
                     pauses only (default). clause = additionally cut a sentence
                     at a comma or semicolon that starts a new clause (a
                     conjunction follows, or a subject-verb clause of at least
                     four words does), pieces of at least three words.
                     clause-all = cut at every comma or semicolon with at least
                     three words on both sides. clause-and = clause-all plus a
                     cut before a coordinating conjunction (and, but, or, so)
                     that has no comma. The annotators mark a clause
                     inside a sentence for 15 % of the training golds
                     (research/07-findings-log.md entries 40, 44, 45).
  ASR_CLEAN          1 = temperature 0, no conditioning on previous text, no
                     fallback ladder (removes the slow-file tail)  (default 1)
  SPAN_ON_NO         1 = return the best span for every question, including
                     those answered no (the scorer credits spans on annotated
                     yes questions whatever we answered)        (default 1)
  REQUEST_DUMP_DIR   if set, example.py writes each incoming audio + questions
                     there (off by default)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import site
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# Windows: CUDA runtime DLLs live in pip's nvidia/* packages, not on PATH.
_nv = Path(site.getsitepackages()[0]) / 'Lib' / 'site-packages' / 'nvidia'
for _d in ('cublas', 'cudnn', 'cuda_nvrtc'):
    _p = _nv / _d / 'bin'
    if _p.is_dir():
        os.add_dll_directory(str(_p))
        os.environ['PATH'] = str(_p) + os.pathsep + os.environ['PATH']
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ASR_MODEL = os.environ.get('ASR_MODEL', 'large-v3')
ASR_DEVICE = os.environ.get('ASR_DEVICE', 'cuda')
TRANSCRIPT_CACHE = os.environ.get('TRANSCRIPT_CACHE', '0') == '1'
LLM_URL = os.environ.get('LLM_URL', 'http://localhost:11434').rstrip('/')
LLM_BACKEND = os.environ.get('LLM_BACKEND', 'vllm' if LLM_URL.endswith('/v1') else 'ollama')
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3.5:4b' if LLM_BACKEND == 'ollama' else '')
LLM_VARIANT = os.environ.get('LLM_VARIANT', 'units' if LLM_BACKEND == 'ollama' else 'units-fewshot')
LLM_TIMEOUT = float(os.environ.get('LLM_TIMEOUT', '25'))
LLM_DEADLINE = float(os.environ.get('LLM_DEADLINE', '40'))
LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '600'))
LLM_NO_THINK = os.environ.get('LLM_NO_THINK', 'ollama' if ':11434' in LLM_URL else 'vllm')
LLM_FALLBACK_URL = os.environ.get('LLM_FALLBACK_URL', 'http://localhost:11434').rstrip('/')
LLM_FALLBACK_MODEL = os.environ.get('LLM_FALLBACK_MODEL', 'qwen3:4b' if LLM_BACKEND == 'vllm' else '')
LLM_NUM_CTX = int(os.environ.get('LLM_NUM_CTX', '3072'))
START_RULE = os.environ.get('START_RULE', 'first-word-end')
# Edge offsets fitted leave-one-conversation-out on the 39 training files
# (bench/asr/fit_edges.py, research/07-findings-log.md entries 5 and 17), keyed
# by ASR model: (start offset for the first-word-end rule, start offset for the
# unit-start rule, end offset). Models not listed fall back to large-v3's.
_FITTED = {
    'large-v3': (-0.14, 0.36, 0.12),
    'large-v3-turbo': (-0.20, 0.28, -0.02),
}
_fwe, _us, _end = _FITTED.get(ASR_MODEL, _FITTED['large-v3'])
START_OFFSET = float(os.environ.get('START_OFFSET', str(_fwe if START_RULE == 'first-word-end' else _us)))
END_OFFSET = float(os.environ.get('END_OFFSET', str(_end)))
PAUSE_SPLIT = float(os.environ.get('PAUSE_SPLIT', '0.6'))
UNIT_SPLIT = os.environ.get('UNIT_SPLIT', 'sentence')
# Off by default: on the 39 training files the no-ladder decode gave cleaner
# text but worse timestamps (raw ceiling 0.730 vs 0.753 for large-v3), and the
# fitted offsets above were measured with the default decode.
ASR_CLEAN = os.environ.get('ASR_CLEAN', '0') == '1'
SPAN_ON_NO = os.environ.get('SPAN_ON_NO', '1') == '1'

Span = Tuple[float, float]


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #

@dataclass
class Word:
    w: str
    start: float
    end: float


_asr = None


def _load_asr():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel
        t0 = time.time()
        compute = 'float16' if ASR_DEVICE == 'cuda' else 'int8'
        _asr = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=compute)
        logger.info('ASR %s loaded in %.1fs', ASR_MODEL, time.time() - t0)
    return _asr


def asr_kwargs() -> dict:
    kw = dict(language='en', word_timestamps=True, beam_size=5, vad_filter=False)
    if ASR_CLEAN:
        # No temperature fallback ladder and no conditioning on the previous
        # window: the slow files in the training set were the ones where the
        # ladder looped on low-probability words (research/06, idea 6).
        kw.update(temperature=0.0, condition_on_previous_text=False)
    return kw


def _cached_transcript(audio_filename: str) -> Optional[Tuple[List[Word], float]]:
    if not TRANSCRIPT_CACHE:
        return None
    f = HERE / 'transcripts' / f'{Path(audio_filename).stem}.{ASR_MODEL}.json'
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding='utf-8'))
    words: List[Word] = []
    for s in d['segments']:
        for w in s['words']:
            words.append(Word(w['w'], w['start'], w['end']))
        if s['words']:
            words[-1].w += '\x00'      # segment boundary, same marker as the live path
    return words, float(d['duration'])


def transcribe(audio_bytes: bytes, audio_filename: str) -> Tuple[List[Word], float]:
    """Word stream with timestamps, plus audio duration in seconds."""
    cached = _cached_transcript(audio_filename)
    if cached is not None:
        logger.info('%s: transcript from cache', audio_filename)
        return cached
    model = _load_asr()
    segments, info = model.transcribe(io.BytesIO(audio_bytes), **asr_kwargs())
    words: List[Word] = []
    for s in segments:
        for w in (s.words or []):
            words.append(Word(w.word, w.start, w.end))
        # segment boundary: mark so the unit splitter can cut here too
        if words:
            words[-1].w = words[-1].w + '\x00'
    return words, float(info.duration)


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #

@dataclass
class Unit:
    idx: int
    start: float
    end: float
    text: str
    first_word_end: float = 0.0


_TERMINAL = re.compile(r'[.!?]["\')\]]*\x00?$')

# --- clause cut (UNIT_SPLIT=clause) ----------------------------------------- #
# The annotators mark a clause inside one of our sentence units for about 15 %
# of the training golds; 37 of the 27B's 195 spans lose tIoU that way and no
# whole-sentence citation can reach them (findings log 40, 44, 45). These
# lists drive the cut; they are deliberately small and closed, because every
# extra split costs the model an id it has to choose between.
_CLAUSE_LEAD = frozenset("""
    and but so because which who where when while if then or although though
    unless since as after before
""".split())
# a comma followed by one of these plus a verb nearby starts a new clause even
# without a conjunction ("I take the tablet, my blood sugar has been fine")
_CLAUSE_SUBJ = frozenset("""
    i you he she it we they that this there the a an my your his her its our their
    everything nothing something anything someone everyone nobody one
""".split())
_CLAUSE_VERB = frozenset("""
    is are was were am be been being has have had having do does did done
    will would can could should shall may might must get gets got take takes took
    feel feels felt think thinks thought said says say want wants need needs
    seem seems look looks keep keeps kept go goes went come comes came
    make makes made give gives gave start starts started stop stops stopped
    stay stays stayed remain remains use uses used
""".split())
# never cut between a number and its unit ("53, mmol/mol" is one measurement)
_CLAUSE_UNITS = frozenset("""
    mg mcg ug g kg ml l dl cl mmol mol mmhg kpa cm mm m km iu ius unit units
    percent bpm milligrams milligram micrograms grams kilos kilograms millilitres
    millilitre litres litre mmol/mol mg/dl mg/ml mmol/l ml/min
""".split())
# clause-and: a coordinating conjunction starts a new piece even without a comma
_CLAUSE_COORD = frozenset({'and', 'but', 'or', 'so'})
_NUMERIC = re.compile(r'^[0-9][0-9.,/-]*$')
_CLAUSE_END = re.compile(r'[,;]["\')\]]*$')       # the comma is the word's own tail


def _clause_tok(w: Word) -> str:
    return w.w.replace('\x00', '').strip()


def _clause_bare(w: Word) -> str:
    """Lower-cased word without surrounding punctuation ('mmol/mol' keeps its slash)."""
    return _clause_tok(w).strip('.,;:!?"\'()[]-').lower()


def _clause_nwords(piece: List[Word]) -> int:
    return sum(1 for w in piece if _clause_bare(w))


def _clause_is_break(words: List[Word], i: int) -> bool:
    """True when words[i] ends with a comma or semicolon that may be cut after."""
    if i + 1 >= len(words):
        return False
    if not _CLAUSE_END.search(_clause_tok(words[i])):
        return False
    # never inside a number or between a number and its unit
    head, nxt = _clause_bare(words[i]), _clause_bare(words[i + 1])
    if not nxt:
        return False
    if _NUMERIC.match(head) and (_NUMERIC.match(nxt) or nxt in _CLAUSE_UNITS):
        return False
    return nxt not in _CLAUSE_UNITS


def _clause_chunk(words: List[Word], i: int) -> List[Word]:
    """The words after the break at i, up to the next break or the unit's end."""
    out: List[Word] = []
    for j in range(i + 1, len(words)):
        out.append(words[j])
        if _CLAUSE_END.search(_clause_tok(words[j])):
            break
    return out


def _clause_subject_verb(chunk: List[Word]) -> bool:
    """A new subject-verb clause of at least four words."""
    bare = [b for b in (_clause_bare(w) for w in chunk) if b]
    if len(bare) < 4:
        return False
    if not any(b in _CLAUSE_SUBJ for b in bare[:2]):
        return False
    return any(b in _CLAUSE_VERB for b in bare[1:5])


def _clause_split(group: List[Word], mode: str) -> List[List[Word]]:
    """One sentence unit's words cut into clause pieces of at least 3 words."""
    breaks = []
    for i in range(len(group) - 1):
        if not _clause_is_break(group, i):
            if mode == 'clause-and' and _clause_bare(group[i + 1]) in _CLAUSE_COORD:
                breaks.append(i)
            continue
        if mode in ('clause-all', 'clause-and'):
            breaks.append(i)
        elif (_clause_bare(group[i + 1]) in _CLAUSE_LEAD
                or _clause_subject_verb(_clause_chunk(group, i))):
            breaks.append(i)
    if not breaks:
        return [group]
    pieces: List[List[Word]] = []
    last = 0
    for i in breaks:
        if _clause_nwords(group[last:i + 1]) < 3:
            continue                                  # piece too short: merge right
        if _clause_nwords(group[i + 1:]) < 3:
            break                                     # tail too short: merge left
        pieces.append(group[last:i + 1])
        last = i + 1
    pieces.append(group[last:])
    return pieces


def make_units(words: List[Word], mode: Optional[str] = None) -> List[Unit]:
    """Sentence-ish units: cut at terminal punctuation, segment ends, long pauses.

    With mode (or UNIT_SPLIT) 'clause' or 'clause-all' each sentence is cut
    again at clause commas; 'sentence' (the default) is the served behaviour.
    """
    mode = (mode or UNIT_SPLIT).lower()
    groups: List[List[Word]] = []
    cur: List[Word] = []

    def flush():
        if cur:
            groups.append(list(cur))
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _TERMINAL.search(w.w.strip()) or w.w.endswith('\x00'):
            flush()
        elif nxt is not None and nxt.start - w.end >= PAUSE_SPLIT:
            flush()
    flush()

    if mode in ('clause', 'clause-all', 'clause-and'):
        groups = [p for g in groups for p in _clause_split(g, mode)]
    elif mode != 'sentence':
        raise ValueError(f'UNIT_SPLIT={mode}: expected sentence, clause, clause-all or clause-and')

    units: List[Unit] = []
    for g in groups:
        text = ''.join(w.w for w in g).replace('\x00', '').strip()
        if text:
            units.append(Unit(len(units), g[0].start, g[-1].end, text, g[0].end))
    return units


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

SYSTEM = """You verify claims about a recorded doctor-patient consultation using ONLY its transcript.
The transcript is a numbered list of short utterances. Some questions are near-misses: they
name the right topic but change one detail (a dose, a duration, a drug, a body part, a
direction such as increase vs. keep unchanged). A near-miss is answered "no".
Answer "yes" only if the transcript actually establishes the statement with every detail
matching. If the topic never comes up, answer "no".
Tag questions ("..., right?", "..., didn't it?") are ordinary questions; the phrasing
does not hint at the answer.
Return JSON only:
{"quote": "<the utterance, copied verbatim, that most specifically states the detail the question asks about>",
 "answer": "yes" | "no",
 "segments": [<ids of every consecutive utterance the statement rests on: the one with the detail plus a
              neighbour when the fact is spread over two (a question and its answer, a statement and its
              number); usually one or two, at most four; empty for no>]}"""

SCHEMA = {
    'type': 'object',
    'properties': {
        'quote': {'type': 'string'},
        'answer': {'type': 'string', 'enum': ['yes', 'no']},
        'segments': {'type': 'array', 'items': {'type': 'integer'}},
    },
    'required': ['quote', 'answer', 'segments'],
}

_TAG = re.compile(r",?\s*(right|correct|isn't it|wasn't it|didn't it|doesn't it|don't they|"
                  r"isn't that right|is that right|is that correct|weren't they|won't it|"
                  r"hasn't it|haven't they|aren't they|did they|was it|is it|does it)\s*\?$", re.I)


def render_transcript(units: List[Unit]) -> str:
    return '\n'.join(f'[{u.idx}] {u.text}' for u in units)


_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.S)
_THINK = re.compile(r'<think>.*?</think>', re.S)


def parse_json(content: str) -> dict:
    """The JSON object in a completion, tolerating a stray <think> block or a
    code fence (same reader as bench/llm/bench.py)."""
    s = _THINK.sub('', content or '').strip()
    s = _FENCE.sub('', s).strip()
    try:
        out = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find('{'), s.rfind('}')
        if a < 0 or b <= a:
            raise
        out = json.loads(s[a:b + 1])
    if not isinstance(out, dict):
        raise ValueError(f'model returned {type(out).__name__}, not an object')
    return out


_prompts_mod = None


def prompts_module():
    """bench/llm/prompts.py, imported lazily: that module imports this one
    (SYSTEM, SCHEMA, the unit builder, the span rule), so it can only load
    once this module is complete."""
    global _prompts_mod
    if _prompts_mod is None:
        import sys
        p = str(HERE / 'bench' / 'llm')
        if p not in sys.path:
            sys.path.insert(0, p)
        import prompts
        _prompts_mod = prompts
    return _prompts_mod


def variant():
    """The prompt builder named by LLM_VARIANT."""
    return prompts_module().VARIANTS[LLM_VARIANT]


def _chat_ollama(url: str, model_tag: str, system: str, user: str, schema: dict,
                 timeout: float, num_ctx: int = LLM_NUM_CTX) -> dict:
    """Ollama's native chat API: schema-constrained JSON, thinking off."""
    body = {
        'model': model_tag,
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'stream': False,
        'think': False,
        'format': schema,
        # num_ctx sizes Ollama's per-slot KV cache. A 3.5-minute consultation is
        # ~1k tokens; 3072 leaves room for the system prompt and the reply
        # without spilling the cache to CPU on an 8 GB card.
        'options': {'temperature': 0.0, 'num_ctx': num_ctx, 'num_predict': 200},
        'keep_alive': -1,
    }
    r = requests.post(f'{url}/api/chat', json=body, timeout=timeout)
    r.raise_for_status()
    return parse_json(r.json()['message']['content'])


_session = requests.Session()
_session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=32, pool_connections=4))
_session.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=32, pool_connections=4))
_resolved_model: Optional[str] = None


def llm_model() -> str:
    """LLM_MODEL, or for vLLM with LLM_MODEL unset the first id GET /v1/models
    returns (cached once it answered)."""
    global _resolved_model
    if LLM_MODEL:
        return LLM_MODEL
    if _resolved_model is None:
        r = _session.get(f'{LLM_URL}/models', timeout=(3.05, 10))
        r.raise_for_status()
        _resolved_model = r.json()['data'][0]['id']
        logger.info('LLM model resolved from %s/models: %s', LLM_URL, _resolved_model)
    return _resolved_model


def _chat_openai(system: str, user: str, schema: dict, demos, timeout: float) -> dict:
    """OpenAI-compatible chat completions (vLLM): schema-constrained JSON, the
    worked examples of a demo variant as prior turns, thinking off. Same body
    as bench/llm/bench.py's Client minus the log-probabilities."""
    messages = [{'role': 'system', 'content': system}]
    for du, da in (demos or []):
        messages.append({'role': 'user', 'content': du})
        messages.append({'role': 'assistant', 'content': da})
    messages.append({'role': 'user', 'content': user})
    body: dict = {
        'model': llm_model(),
        'messages': messages,
        'temperature': 0.0,
        'max_tokens': LLM_MAX_TOKENS,
        'stream': False,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema}},
    }
    if LLM_NO_THINK == 'vllm':
        body['chat_template_kwargs'] = {'enable_thinking': False}
    elif LLM_NO_THINK == 'ollama':
        body['reasoning_effort'] = 'none'
    # connect timeout short: a pod that is gone must fail over within seconds
    r = _session.post(f'{LLM_URL}/chat/completions', json=body, timeout=(3.05, max(1.0, timeout)))
    r.raise_for_status()
    return parse_json(r.json()['choices'][0]['message'].get('content') or '')


class _Breaker:
    """After a transport failure of the primary LLM every question goes straight
    to the fallback for a while instead of each waiting out its own timeout."""

    def __init__(self, hold: float = 20.0):
        self.hold = hold
        self.until = 0.0

    def open(self) -> bool:
        return time.time() < self.until

    def trip(self, exc: Exception) -> None:
        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            self.until = time.time() + self.hold
            logger.warning('primary LLM unreachable (%s); fallback only for %.0f s', type(exc).__name__, self.hold)


_breaker = _Breaker()
_FALLBACK_RESERVE = 12.0      # seconds kept for the local fallback when the primary is tried
_FALLBACK_GRACE = 12.0        # the fallback may run this long past LLM_DEADLINE (still < 60 s)


def ask_primary(p, deadline: float) -> dict:
    """Send one built prompt to the configured backend and return the parsed
    JSON. On the vLLM backend the request gets what is left before `deadline`
    minus the fallback reserve, and is skipped (TimeoutError) while the breaker
    is open or when that budget is under two seconds."""
    remaining = deadline - time.time()
    if LLM_BACKEND != 'vllm':
        return _chat_ollama(LLM_URL, LLM_MODEL, p.system, p.user, p.schema, min(LLM_TIMEOUT, max(3.0, remaining)))
    budget = min(LLM_TIMEOUT, remaining - (_FALLBACK_RESERVE if LLM_FALLBACK_MODEL else 0.0))
    if _breaker.open():
        raise TimeoutError('primary LLM skipped: breaker open after a transport failure')
    if budget < 2.0:
        raise TimeoutError(f'primary LLM skipped: {remaining:.1f} s left before the deadline')
    try:
        return _chat_openai(p.system, p.user, p.schema, getattr(p, 'demos', None), budget)
    except Exception as exc:
        _breaker.trip(exc)
        raise


def ask_fallback(p, deadline: float) -> dict:
    """The local Ollama fallback model on the same prompt, allowed to run a
    little past the LLM deadline (still inside the 60 s of the endpoint)."""
    return _chat_ollama(LLM_FALLBACK_URL, LLM_FALLBACK_MODEL, p.system, p.user, p.schema,
                        max(3.0, deadline + _FALLBACK_GRACE - time.time()), num_ctx=max(LLM_NUM_CTX, 4096))


def ask(p, deadline: float) -> dict:
    """ask_primary, then ask_fallback when the primary fails and a fallback
    model is configured. Raises when both fail."""
    try:
        return ask_primary(p, deadline)
    except Exception as exc:
        if LLM_BACKEND != 'vllm' or not LLM_FALLBACK_MODEL:
            raise
        logger.warning('primary LLM failed (%s: %s); falling back to %s',
                       type(exc).__name__, str(exc)[:120], LLM_FALLBACK_MODEL)
    return ask_fallback(p, deadline)


def warm_llm() -> None:
    """Resolve the model, build the few-shot pool (and refuse to serve without
    it), and send one real request so the first conversation is not the slow one."""
    v = variant()
    if hasattr(v, 'pool'):
        pool = v.pool()
        n_pos = sum(1 for e in pool if e.get('yes') and e.get('text'))
        if n_pos < 150:
            raise RuntimeError(f'few-shot pool for {LLM_VARIANT} has {n_pos} positives with evidence text '
                               f'(expected >= 150): bench/llm/pool/{ASR_MODEL}.json or the transcripts are missing')
        logger.info('few-shot pool: %d examples, %d positives', len(pool), n_pos)
    if hasattr(v, 'set_conversation'):
        v.set_conversation('', ASR_MODEL)
    units = [Unit(0, 0.0, 1.0, 'Hello.', 0.4)]
    try:
        if LLM_BACKEND == 'vllm':
            logger.info('LLM backend vllm at %s, model %s, variant %s, fallback %s',
                        LLM_URL, llm_model(), LLM_VARIANT, LLM_FALLBACK_MODEL or 'none')
        else:
            logger.info('LLM backend ollama at %s, model %s, variant %s', LLM_URL, LLM_MODEL, LLM_VARIANT)
        t0 = time.time()
        p = v.build_all(['Did anyone say hello?'], units) if getattr(v, 'joint', False) else v('Did anyone say hello?', units)
        out = ask(p, time.time() + 30.0)
        logger.info('LLM warm in %.1fs: %s', time.time() - t0, json.dumps(out)[:120])
    except Exception:
        logger.exception('LLM warm-up failed (is the server at %s running with %s?)', LLM_URL, LLM_MODEL or 'the model')
    if LLM_BACKEND == 'vllm' and LLM_FALLBACK_MODEL:
        try:
            t0 = time.time()
            _chat_ollama(LLM_FALLBACK_URL, LLM_FALLBACK_MODEL, SYSTEM, 'TRANSCRIPT:\n[0] Hello.\n\nQUESTION: Did anyone say hello?',
                         SCHEMA, 30.0)
            logger.info('fallback %s warm in %.1fs', LLM_FALLBACK_MODEL, time.time() - t0)
        except Exception:
            logger.exception('fallback warm-up failed (is Ollama at %s running with %s?)', LLM_FALLBACK_URL, LLM_FALLBACK_MODEL)


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #

_NONALNUM = re.compile(r'[^a-z0-9 ]')


def locate_quote(quote: str, units: List[Unit], min_ratio: float = 0.5) -> Optional[int]:
    """The unit whose text best matches the model's verbatim quote, or None.
    On the training set the quote lands on a better unit than the cited id
    (mean tIoU 0.528 vs 0.505 with qwen3:4b), so it is the primary anchor."""
    import difflib
    q = ' '.join(_NONALNUM.sub(' ', quote.lower()).split())
    if len(q) < 8:
        return None
    best, best_idx = 0.0, None
    for u in units:
        r = difflib.SequenceMatcher(None, q, ' '.join(_NONALNUM.sub(' ', u.text.lower()).split())).ratio()
        if r > best:
            best, best_idx = r, u.idx
    return best_idx if best >= min_ratio else None


def anchor_ids(out: dict, units: List[Unit]) -> List[int]:
    """Unit ids to build the span from: the quote's unit first, plus any cited
    ids adjacent to it (the model under-merges: 184 of 186 citations were a
    single unit while a quarter of the gold spans cover two sentences)."""
    cited = [int(i) for i in out.get('segments', []) if isinstance(i, (int, float)) and 0 <= int(i) < len(units)]
    qi = locate_quote(str(out.get('quote', '')), units)
    if qi is None:
        return cited
    keep = [qi] + [i for i in cited if abs(i - qi) <= 2 and i != qi]
    return sorted(set(keep))


def span_from_ids(ids: List[int], units: List[Unit], duration: float) -> Optional[Span]:
    ids = sorted({i for i in ids if 0 <= i < len(units)})
    if not ids:
        return None
    # Keep the cited run contiguous: start from the first id and extend while
    # the next cited id is adjacent (gap of at most one unit). Anything beyond
    # is a stray citation and only dilutes IoU.
    run = [ids[0]]
    for i in ids[1:]:
        if i - run[-1] <= 2:
            run.append(i)
        else:
            break
    first = units[run[0]]
    anchor = first.first_word_end if (START_RULE == 'first-word-end' and first.first_word_end > 0) else first.start
    start = anchor + START_OFFSET
    end = units[run[-1]].end + END_OFFSET
    start = max(0.0, min(start, duration))
    end = max(start + 0.05, min(end, duration))
    return round(start, 2), round(end, 2)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def answer_all(audio_bytes: bytes, audio_filename: str, questions: List[str]
               ) -> Tuple[List[bool], List[Optional[Span]]]:
    t0 = time.time()
    words, duration = transcribe(audio_bytes, audio_filename)
    units = make_units(words)
    t_asr = time.time() - t0
    logger.info('%s: %d words, %d units, ASR %.1fs', audio_filename, len(words), len(units), t_asr)
    deadline = t0 + LLM_DEADLINE
    v = variant()
    if hasattr(v, 'set_conversation'):          # few-shot variants: examples never come from this file
        v.set_conversation(Path(audio_filename).stem, ASR_MODEL)
    prompts = prompts_module()

    def finish(out: dict, p) -> Tuple[bool, Optional[Span]]:
        yes, span = p.postprocess(out, units, words, duration)
        if not yes and not SPAN_ON_NO:
            span = None
        return bool(yes), span

    def one(q: str) -> Tuple[bool, Optional[Span]]:
        try:
            p = v(q, units)
            return finish(ask(p, deadline), p)
        except Exception:
            logger.exception('question failed, guessing: %s', q)
            return True, None

    def one_fallback(q: str) -> Tuple[bool, Optional[Span]]:
        # joint request failed: each question alone, plain units prompt, local fallback model
        try:
            p = prompts.units(q, units)
            return finish(ask_fallback(p, deadline), p)
        except Exception:
            logger.exception('fallback question failed, guessing: %s', q)
            return True, None

    if getattr(v, 'joint', False):
        try:
            p = v.build_all(list(questions), units)
            results = [finish(item, p) for item in v.split(ask_primary(p, deadline), len(questions))]
        except Exception:
            logger.exception('joint request failed; %s', 'per-question fallback' if LLM_FALLBACK_MODEL else 'guessing')
            if LLM_FALLBACK_MODEL:
                with ThreadPoolExecutor(max_workers=len(questions) or 1) as ex:
                    results = list(ex.map(one_fallback, questions))
            else:
                results = [(True, None)] * len(questions)
    else:
        with ThreadPoolExecutor(max_workers=len(questions) or 1) as ex:
            results = list(ex.map(one, questions))

    answers = [a for a, _ in results]
    spans = [s for _, s in results]
    logger.info('%s: total %.1fs (ASR %.1fs, LLM %.1fs), yes=%d/%d',
                audio_filename, time.time() - t0, t_asr, time.time() - t0 - t_asr,
                sum(answers), len(answers))
    return answers, spans


def warm_up() -> None:
    """Load the ASR, run one real transcription, and exercise the LLM once, at
    import, before the attempt starts. The first CUDA inference is several times
    slower than steady state, and the attempt has no grace period for it."""
    if not TRANSCRIPT_CACHE:
        model = _load_asr()
        import numpy as np
        t0 = time.time()
        sample = HERE / 'data' / 'audio' / 'conversation_sample_4.mp3'
        audio = str(sample) if sample.exists() else np.zeros(16000 * 20, dtype=np.float32)
        segs, _ = model.transcribe(audio, **asr_kwargs())
        n = sum(1 for _ in segs)
        logger.info('ASR warm: %d segments in %.1fs', n, time.time() - t0)
    warm_llm()
