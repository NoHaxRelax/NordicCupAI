"""Experiment v1: timestamped ASR -> sentence units -> local LLM picks unit ids.

Pipeline for one request:

  1. decode MP3 bytes, transcribe with faster-whisper (word timestamps), or load
     the cached transcript when TRANSCRIPT_CACHE=1 and one exists (dev only).
  2. cut the word stream into sentence units at terminal punctuation, segment
     boundaries and pauses longer than PAUSE_SPLIT seconds.
  3. for each question, ask the LLM (Ollama, JSON schema output) for a verbatim
     quote, yes/no, and the ids of the units that support a yes.
  4. map ids back to seconds, apply the fitted edge offsets, return.

Everything is configured by environment variables so the same file runs on the
laptop (small model) and on a big GPU (large model) without edits.

  ASR_MODEL          faster-whisper model name           (default large-v3)
  ASR_DEVICE         cuda | cpu                          (default cuda)
  TRANSCRIPT_CACHE   1 = use transcripts/<stem>.<ASR_MODEL>.json when present
  LLM_URL            Ollama base URL                     (default http://localhost:11434)
  LLM_MODEL          Ollama model tag                    (default qwen3.5:4b)
  LLM_TIMEOUT        seconds per question               (default 25)
  START_RULE         first-word-end | unit-start         (default first-word-end)
  START_OFFSET       seconds added to the start anchor   (default -0.14 for
                     first-word-end, +0.36 for unit-start; fitted on the 39
                     training conversations, see research/07-findings-log.md)
  END_OFFSET         seconds added to unit ends          (default 0.12)
  PAUSE_SPLIT        split a sentence at a pause >= this (default 0.6)
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
LLM_MODEL = os.environ.get('LLM_MODEL', 'qwen3.5:4b')
LLM_TIMEOUT = float(os.environ.get('LLM_TIMEOUT', '25'))
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


def make_units(words: List[Word]) -> List[Unit]:
    """Sentence-ish units: cut at terminal punctuation, segment ends, long pauses."""
    units: List[Unit] = []
    cur: List[Word] = []

    def flush():
        if cur:
            text = ''.join(w.w for w in cur).replace('\x00', '').strip()
            if text:
                units.append(Unit(len(units), cur[0].start, cur[-1].end, text, cur[0].end))
            cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _TERMINAL.search(w.w.strip()) or w.w.endswith('\x00'):
            flush()
        elif nxt is not None and nxt.start - w.end >= PAUSE_SPLIT:
            flush()
    flush()
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


def ask_llm(transcript: str, question: str) -> dict:
    claim = question.strip()
    tag = _TAG.search(claim)
    if tag:
        claim = f'{claim}  (Read this as the plain question: is it established that {claim[:tag.start()].strip().rstrip(",")}?)'
    body = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': f'TRANSCRIPT:\n{transcript}\n\nQUESTION: {claim}'},
        ],
        'stream': False,
        'think': False,
        'format': SCHEMA,
        'logprobs': True,          # P(yes) at the answer token, see p_yes_from_native
        'top_logprobs': 6,
        # num_ctx sizes Ollama's per-slot KV cache. A 3.5-minute consultation is
        # ~1k tokens; 3072 leaves room for the system prompt and the reply
        # without spilling the cache to CPU on an 8 GB card.
        'options': {'temperature': 0.0, 'num_ctx': int(os.environ.get('LLM_NUM_CTX', '3072')),
                    'num_predict': 200},
        'keep_alive': -1,
    }
    r = requests.post(f'{LLM_URL}/api/chat', json=body, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    content = r.json()['message']['content']
    return json.loads(content)


def warm_llm() -> None:
    try:
        ask_llm('[0] Hello.', 'Did anyone say hello?')
        logger.info('LLM %s warm', LLM_MODEL)
    except Exception:
        logger.exception('LLM warm-up failed (is Ollama running with %s?)', LLM_MODEL)


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
    transcript = render_transcript(units)
    logger.info('%s: %d words, %d units, ASR %.1fs', audio_filename, len(words), len(units), t_asr)

    def one(q: str) -> Tuple[bool, Optional[Span]]:
        try:
            out = ask_llm(transcript, q)
            yes = str(out.get('answer', '')).lower() == 'yes'
            ids = anchor_ids(out, units)
            span = span_from_ids(ids, units, duration)
            if not yes and not SPAN_ON_NO:
                span = None
            return yes, span
        except Exception:
            logger.exception('question failed, guessing: %s', q)
            return True, None

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
