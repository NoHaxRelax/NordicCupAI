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
  START_OFFSET       seconds added to unit starts        (default 0.36)
  END_OFFSET         seconds added to unit ends          (default 0.12)
  PAUSE_SPLIT        split a sentence at a pause >= this (default 0.6)
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
START_OFFSET = float(os.environ.get('START_OFFSET', '0.36'))
END_OFFSET = float(os.environ.get('END_OFFSET', '0.12'))
PAUSE_SPLIT = float(os.environ.get('PAUSE_SPLIT', '0.6'))

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
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes), language='en', word_timestamps=True,
        beam_size=5, vad_filter=False,
    )
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


_TERMINAL = re.compile(r'[.!?]["\')\]]*\x00?$')


def make_units(words: List[Word]) -> List[Unit]:
    """Sentence-ish units: cut at terminal punctuation, segment ends, long pauses."""
    units: List[Unit] = []
    cur: List[Word] = []

    def flush():
        if cur:
            text = ''.join(w.w for w in cur).replace('\x00', '').strip()
            if text:
                units.append(Unit(len(units), cur[0].start, cur[-1].end, text))
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
{"quote": "<the single utterance, copied verbatim, that best supports or refutes the claim>",
 "answer": "yes" | "no",
 "segments": [<ids of the utterances that make a yes true; usually one, at most three; empty for no>]}"""

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
    start = units[run[0]].start + START_OFFSET
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
            ids = [int(i) for i in out.get('segments', []) if isinstance(i, (int, float))]
            if yes and not ids:
                # The model said yes but cited nothing: find the quote instead.
                quote = str(out.get('quote', '')).strip().lower()
                ids = [u.idx for u in units if quote and quote[:40] in u.text.lower()][:1]
            span = span_from_ids(ids, units, duration) if yes else None
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
        segs, _ = model.transcribe(audio, language='en', word_timestamps=True, beam_size=5)
        n = sum(1 for _ in segs)
        logger.info('ASR warm: %d segments in %.1fs', n, time.time() - t0)
    warm_llm()
