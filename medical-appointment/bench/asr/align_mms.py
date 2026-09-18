"""Re-time an existing transcript with the torchaudio MMS_FA forced aligner.

Reads  transcripts/<stem>.<tag>.json      (any runner's output; bench/README.md schema)
writes transcripts/<stem>.<tag-out>.json  (default <tag>+mms): the same segments and the
same words, with every word's start/end replaced by MMS forced-alignment boundaries.

How: the words of one conversation are joined into a single text and normalised for a
character-level CTC aligner (lowercase, digits and numbers to words with num2words, units
expanded, accents folded, punctuation removed). That text is force-aligned to the 16 kHz
mono audio with torchaudio.pipelines.MMS_FA (MMS-300M wav2vec2, one frame per 20 ms).
An index map (one original word -> zero or more aligner words) gives every ORIGINAL word,
punctuation and all, start = start of its first aligned sub-word and end = end of its last.
Segment start/end are recomputed from their words; text and spellings are untouched, so
span_ceiling.py, compare.py and model.py read the output like any other transcript.

Install: bench/hpc/env.sh builds venv-asr (/dtu/blackhole/1e/205502/venvs/venv-asr) with the
/appl9 Python 3.11 called by absolute path, torch from the cu129 wheel index and torchaudio >= 2.11
(the final torchaudio release; it needs torch >= 2.11), and bench/hpc/asr_bench.lsf runs this file
from that venv. Two things the obvious one-liner gets wrong on gbar: the bare python3 on the login
node is the OS interpreter, older than the 3.10 that torch 2.14 and torchaudio 2.11 require
(torchaudio 2.11 ships cp310-cp314 manylinux wheels only, no sdist), and `module load python3`
exports PYTHONHOME and PYTHONPATH, which leak into any venv (see the env.sh caveats), so env.sh
does not load it. By hand, the equivalent of what env.sh does:

    /appl9/python/3.11.13/bin/python3.11 -m venv /dtu/blackhole/1e/205502/venvs/mms
    . /dtu/blackhole/1e/205502/venvs/mms/bin/activate
    pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
        "torch>=2.11" "torchaudio>=2.11" num2words numpy soundfile librosa
    export TORCH_HOME=/dtu/blackhole/1e/205502/torch-home   # 1.2 GB checkpoint -> $TORCH_HOME/hub/checkpoints
    python -c "import torch, torchaudio; print(torch.cuda.is_available(), torchaudio.__version__)"  # on a gpuh100 node: True 2.11.x

The H100 nodes run driver 610.57.04 (env.sh, `lshosts -gpu` on 2026-09-17), so any cu12x/cu13x
wheel loads there. main() still refuses --device cuda when torch.cuda.is_available() is False
(exit 2, before the checkpoint is loaded) instead of dying in model.to('cuda').

Run (from medical-appointment/):

    python bench/asr/align_mms.py --tag large-v3                          # -> <stem>.large-v3+mms.json
    python bench/asr/align_mms.py --tag parakeet-tdt-0.6b-v3 --tag-out parakeet-tdt-0.6b-v3+mms
    python bench/asr/align_mms.py --tag canary-qwen-2.5b                  # text-only source: words = text.split()
    python bench/asr/align_mms.py --tag large-v3 --dry-run --show 40      # normalisation only; no torch import

Licence: the MMS_FA weights are CC-BY-NC 4.0 (stated on
https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html, pointing at
https://github.com/facebookresearch/fairseq/tree/100cd91db19bb27277a06a25eb4154c805b10189/examples/mms#license).
Non-commercial use only: fine for this bench, not for a product. run_whisperx.py's wav2vec2
English aligner is the BSD alternative.

Caveats
- torchaudio >= 2.10 is required. The C++ forced_align op was slated for removal in 2.9 and
  then preserved (pytorch/audio#3902, 2.10 release notes); pipelines.MMS_FA, functional
  .forced_align, merge_tokens and TokenSpan are all in the 2.11 stable docs. Audio I/O moved
  to torchcodec, but this file never calls torchaudio.load (common.load_audio decodes).
- Times are 0.02 s * frame index (exact 320-sample stride; frame f covers samples
  [320 f, 320 f + 400)). The torchaudio tutorial uses ratio = n_samples / n_frames, which
  converges to 320 for long audio (logged as sample_ratio); the difference is well under one
  frame for a 3-minute file. Any constant bias is what compare.py measures and model.py
  corrects with its offsets.
- The audio goes through the model in --window second chunks with --context seconds of
  neighbouring audio on each side (default 30 + 2, as ctc-forced-aligner does); the context
  frames are dropped, the chunk emissions concatenated, and forced_align runs once over the
  whole file. --window 0 runs the whole conversation in one pass (self-attention over ~15k
  frames for 5 minutes: fits an 80 GB H100, not the laptop).
- Words whose normalised form is empty (pure punctuation, hallucinated CJK, ...) cannot be
  aligned; they get a zero-width span at the previous word's end and are counted in
  n_words_empty. Characters outside the MMS dictionary (a-z and the apostrophe) are dropped
  (--oov drop, the tutorial's regex behaviour) or replaced by the <star> wildcard token
  (--oov star, so the audio they cover is consumed instead of being pushed onto neighbours).
- Wrong ASR words are forced onto the audio; no wildcard is inserted automatically.
- num2words writes British cardinals ("one hundred and twenty"); an unspoken "and" costs a
  few frames. Years 1900-2099 are read as years ("twenty seventeen"). Units get American
  spelling ("milliliters"), matching the Whisper output in this corpus ("centimeters").
- p on every word is the aligner's length-weighted mean token probability; the source
  model's word probability, if any, is kept as p_asr.
- No digit reaches the OOV filter: a/b fractions are read "a over b" (120/80, 1/2, 24/7),
  decimals digit by digit after the point (7.0 -> "seven point zero", 7.50 -> "seven point five
  zero"), decades as plurals (20s, 1990s -> "twenties", "nineteen nineties") and a digit run left
  in a token that is neither a number nor a unit (5'10", 12,5) is spelled out where it stands.
  Words that still normalise to nothing, and OOV characters, are listed per file as WARNING.
- One file failing (more CTC steps than frames, CUDA OOM with --window 0, a malformed source
  JSON, a torchaudio build without the forced_align op) is logged with its traceback, counted
  as FAILED and the run continues with the next file. Exit code: 0, 1 if any file failed, 2 if
  --device cuda has no CUDA. The LSF step wrapper records the rc and keeps what was written.
"""
from __future__ import annotations

import logging
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from num2words import num2words  # https://github.com/savoirfairelinux/num2words : num2words(number, lang='en', to='cardinal'|'ordinal'|'year')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, audio_files, load_audio, read_transcript, standard_args, write_transcript  # noqa: E402

log = logging.getLogger('align_mms')

# wav2vec2 feature extractor: conv strides 5,2,2,2,2,2,2 (kernels 10,3,3,3,3,2,2) -> one emission
# frame per 320 samples (20 ms at 16 kHz) with a 400-sample receptive field.
# https://docs.pytorch.org/audio/main/generated/torchaudio.models.wav2vec2_model.html
FRAME_STRIDE = 320
FRAME_RF = 400
FRAME_SEC = FRAME_STRIDE / SR                      # 0.02

# MMS_FA labels: blank '-' at index 0, these 27, then '*' (star) at 28 with with_star=True.
# https://github.com/pytorch/audio/blob/main/src/torchaudio/pipelines/_wav2vec2/impl.py
MMS_LABELS = ('a', 'i', 'e', 'n', 'o', 'u', 't', 's', 'r', 'm', 'k', 'l', 'd', 'g', 'h', 'y',
              'b', 'p', 'w', 'c', 'v', 'j', 'z', 'f', "'", 'q', 'x')
BLANK = '-'
STAR = '*'


# --------------------------------------------------------------------------- #
# Text normalisation (original word -> aligner words)
# --------------------------------------------------------------------------- #

# typographic apostrophes -> ', micro sign / Greek mu -> u (so "µg" becomes "ug")
_CHARMAP = str.maketrans({'’': "'", '‘': "'", 'ʼ': "'", '`': "'", '´': "'",
                          'µ': 'u', 'μ': 'u'})
# ordered: the longer symbol first
_SYMBOLS = (('%', ' percent '), ('‰', ' per mille '), ('°c', ' degrees celsius '),
            ('°f', ' degrees fahrenheit '), ('°', ' degrees '), ('+', ' plus '),
            ('&', ' and '), ('=', ' equals '))
_THOUSANDS = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')                 # 1,000 -> 1000
_TIME = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)')               # 10:30 -> 10 30
_ORDINAL = re.compile(r'(?<![\w.])(\d+)(st|nd|rd|th)(?!\w)')        # 21st -> twenty-first
_DECADE = re.compile(r'(?<![\d.])(\d*0)s(?!\w)')                    # 20s, 1990s -> twenties, nineteen nineties (before _ATTACHED, which would make "20 s" = 20 seconds)
_ATTACHED = re.compile(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)')        # 100mg -> 100 mg, a1c -> a 1 c
_NUMBER = re.compile(r'^\.?\d+(?:\.\d+)?$')                        # 28, 2.5, .9
_FRACTION = re.compile(r'^(\.?\d+(?:\.\d+)?)/(\.?\d+(?:\.\d+)?)$')  # 120/80, 1/2, 24/7 -> a over b ('/' survives _SPLIT1)
_DIGITS = re.compile(r'\d+')                                       # last resort: digit runs in a token that is neither number nor unit (5'10", 12,5)
_EDGE_PUNCT = re.compile(r'^[^\w.]+|[^\w]+$')                       # keeps a leading '.' (".9" from "138" ".9.")
_SPLIT1 = re.compile(r'[\s\-–—_]+')                       # first pass: '/' survives for mmol/l
_SPLIT2 = re.compile(r'[\s\-–—_/]+')                      # second pass
_YEARS = (1900, 2099)

# unit -> (singular, plural); plural unless the preceding number is exactly 1
UNITS: Dict[str, Tuple[str, str]] = {
    'mg': ('milligram', 'milligrams'), 'mcg': ('microgram', 'micrograms'), 'ug': ('microgram', 'micrograms'),
    'ng': ('nanogram', 'nanograms'), 'g': ('gram', 'grams'), 'gm': ('gram', 'grams'), 'gms': ('gram', 'grams'),
    'kg': ('kilogram', 'kilograms'), 'kgs': ('kilogram', 'kilograms'), 'lb': ('pound', 'pounds'),
    'lbs': ('pound', 'pounds'), 'oz': ('ounce', 'ounces'),
    'ml': ('milliliter', 'milliliters'), 'dl': ('deciliter', 'deciliters'), 'cl': ('centiliter', 'centiliters'),
    'l': ('liter', 'liters'),
    'mm': ('millimeter', 'millimeters'), 'cm': ('centimeter', 'centimeters'), 'm': ('meter', 'meters'),
    'km': ('kilometer', 'kilometers'), 'ft': ('foot', 'feet'),
    'mmol': ('millimole', 'millimoles'), 'mol': ('mole', 'moles'),
    'iu': ('international unit', 'international units'),
    'mmhg': ('millimeters of mercury', 'millimeters of mercury'),
    'bpm': ('beats per minute', 'beats per minute'),
    'kcal': ('kilocalorie', 'kilocalories'), 'cal': ('calorie', 'calories'),
    'h': ('hour', 'hours'), 'hr': ('hour', 'hours'), 'hrs': ('hour', 'hours'),
    'min': ('minute', 'minutes'), 'mins': ('minute', 'minutes'),
    'sec': ('second', 'seconds'), 'secs': ('second', 'seconds'), 's': ('second', 'seconds'),
    'wk': ('week', 'weeks'), 'wks': ('week', 'weeks'), 'yr': ('year', 'years'), 'yrs': ('year', 'years'),
    'x': ('times', 'times'), 'pct': ('percent', 'percent'),
}
# ambiguous as bare tokens: only expanded directly after a number
AFTER_NUMBER_ONLY = {'g', 'l', 'm', 'h', 's', 'x', 'cal', 'min', 'sec'}
ABBREV = {'dr': 'doctor', 'mr': 'mister', 'mrs': 'missus', 'vs': 'versus', 'etc': 'etcetera'}


def _fold(text: str) -> str:
    """Strip diacritics from Latin letters (é -> e); leave everything else for the OOV filter."""
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        base = ''.join(c for c in unicodedata.normalize('NFKD', ch) if unicodedata.category(c) != 'Mn')
        out.append(base if base.isascii() and base else ch)
    return ''.join(out)


def _number_words(tok: str) -> Tuple[str, float]:
    """'.9' -> 'point nine'; '2.5' -> 'two point five'; '2017' -> 'twenty seventeen'; '1000' -> 'one thousand'."""
    if tok.startswith('.'):
        return 'point ' + ' '.join(num2words(int(d)) for d in tok[1:]), float('0' + tok)
    if '.' in tok:                                            # digit by digit after the point: num2words('7.0')
        ip, fp = tok.split('.', 1)                            # is 'seven' and '7.50' 'seven point five', but
        return num2words(int(ip)) + ' point ' + ' '.join(num2words(int(d)) for d in fp), float(tok)  # the zero is spoken
    value = float(tok)
    if len(tok) > 6:                                          # phone-number-like: digit by digit
        return ' '.join(num2words(int(d)) for d in tok), value
    if len(tok) == 4 and _YEARS[0] <= value <= _YEARS[1]:
        return num2words(int(tok), to='year'), value
    return num2words(int(tok)), value


def _decade_words(tok: str) -> str:
    """'20' -> 'twenties', '1990' -> 'nineteen nineties', '2000' -> 'two thousands'."""
    words, _ = _number_words(tok)
    return words[:-1] + 'ies' if words.endswith('y') else words + 's'


def _unit_words(tok: str, value: Optional[float]) -> Optional[str]:
    """'mg' -> 'milligrams', 'mmol/l' -> 'millimoles per liter'; None if not a unit."""
    parts = tok.split('/')
    if any(p not in UNITS for p in parts):
        return None
    if len(parts) == 1 and parts[0] in AFTER_NUMBER_ONLY and value is None:
        return None
    singular = value is not None and value == 1.0
    return ' per '.join([UNITS[parts[0]][0 if singular else 1]] + [UNITS[p][0] for p in parts[1:]])


def normalise_word(word: str, carry: Optional[float], allowed: frozenset, oov: str,
                   stats: Counter) -> Tuple[List[str], Optional[float]]:
    """One original word -> its aligner words (0..n), plus the numeric value to carry to the
    next word (so "100" followed by "mg." pluralises and expands)."""
    w = _fold(word.translate(_CHARMAP)).lower()
    w = _THOUSANDS.sub('', w)
    for sym, rep in _SYMBOLS:
        w = w.replace(sym, rep)
    w = _TIME.sub(lambda m: f'{int(m.group(1))} {int(m.group(2))}' if int(m.group(2)) else str(int(m.group(1))), w)
    w = _ORDINAL.sub(lambda m: num2words(int(m.group(1)), to='ordinal'), w)
    w = _DECADE.sub(lambda m: _decade_words(m.group(1)), w)
    w = _ATTACHED.sub(' ', w)
    pieces: List[str] = []
    value = carry
    for tok in _SPLIT1.split(w):
        core = _EDGE_PUNCT.sub('', tok)
        stats['chars_punct'] += len(tok) - len(core)
        if not core:
            continue
        if _NUMBER.match(core):
            text, value = _number_words(core)
            pieces.append(text)
            continue
        m = _FRACTION.match(core)
        if m:                                                 # 120/80 -> one hundred and twenty over eighty
            a, va = _number_words(m.group(1))
            b, vb = _number_words(m.group(2))
            pieces.append(f'{a} over {b}')
            value = va / vb if vb else None                   # 1/1 mg -> milligram, 1/2 l -> liters
            continue
        unit = _unit_words(core, value)
        value = None
        if unit is not None:
            pieces.append(unit)
        elif _DIGITS.search(core):                            # 5'10" -> five ' ten, 12,5 -> twelve , five
            pieces.append(_DIGITS.sub(lambda m: f' {_number_words(m.group())[0]} ', core))
        else:
            pieces.append(ABBREV.get(core, core))
    out: List[str] = []
    for tok in _SPLIT2.split(' '.join(pieces)):
        buf = []
        for ch in tok:
            if ch in allowed:
                buf.append(ch)
            elif unicodedata.category(ch)[0] in 'PZS':      # punctuation, separators, symbols
                stats['chars_punct'] += 1
                buf.append(' ')
            else:                                           # letters/digits the dictionary lacks
                stats['chars_oov'] += 1
                buf.append(STAR if oov == 'star' else ' ')
        for sub in ''.join(buf).split():
            sub = sub.strip("'")
            if sub:
                out.append(sub)
    if not out:
        stats['words_empty'] += 1
    return out, value


def normalise_words(words: List[str], allowed: frozenset, oov: str) -> Tuple[List[List[str]], Counter]:
    """Index map: groups[i] are the aligner words of original word i (may be empty)."""
    stats: Counter = Counter()
    groups: List[List[str]] = []
    carry: Optional[float] = None
    for w in words:
        g, carry = normalise_word(w, carry, allowed, oov, stats)
        groups.append(g)
    stats['n_words'] = len(words)
    stats['n_aligner_words'] = sum(len(g) for g in groups)
    return groups, stats


# --------------------------------------------------------------------------- #
# MMS_FA
# --------------------------------------------------------------------------- #

def load_mms(device: str, model_dir: Optional[str]):
    """Model (log-probs, star dimension appended), tokenizer, aligner and dictionary.
    https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.Wav2Vec2FABundle.html"""
    from torchaudio.pipelines import MMS_FA as bundle  # https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html
    assert int(bundle.sample_rate) == SR, bundle.sample_rate
    # dl_kwargs go to torch.hub.load_state_dict_from_url; model_dir overrides $TORCH_HOME/hub/checkpoints
    # https://docs.pytorch.org/docs/2.14/hub.html#torch.hub.load_state_dict_from_url
    dl_kwargs = {'model_dir': model_dir} if model_dir else None
    t0 = time.time()
    model = bundle.get_model(with_star=True, dl_kwargs=dl_kwargs)   # get_model(with_star: bool = True, *, dl_kwargs=None)
    model = model.to(device).eval()
    tokenizer = bundle.get_tokenizer()      # List[str] words -> List[List[int]]; KeyError on a char outside the dict
    aligner = bundle.get_aligner()          # (emission (T, C) log-probs, tokens) -> List[List[TokenSpan]]
    dictionary = bundle.get_dict()          # {'-': 0, 'a': 1, ..., "'": .., '*': 28}
    log.info('MMS_FA loaded in %.1fs on %s: %d labels, blank=%d star=%d', time.time() - t0, device,
             len(dictionary), dictionary[BLANK], dictionary[STAR])
    return model, tokenizer, aligner, dictionary


def compute_emission(model, wave: np.ndarray, device: str, window_s: float, context_s: float):
    """Frame-wise log-probs (T, C) for the whole file. Chunked: each window of `window_s`
    seconds is fed with `context_s` seconds of real audio on both sides, the context frames
    are cut off, and the pieces are concatenated so that frame f always starts at sample
    320 f, exactly as a single pass would place it."""
    import torch
    n = wave.shape[0]
    n_expect = max(0, (n - FRAME_RF) // FRAME_STRIDE + 1)       # frames a single pass produces
    x = torch.from_numpy(wave).unsqueeze(0)                     # (1, n) float32
    with torch.inference_mode():
        if window_s <= 0:
            # forward(waveforms (batch, frames), lengths=None) -> (output (batch, frames, labels), lengths)
            # https://docs.pytorch.org/audio/main/generated/torchaudio.models.Wav2Vec2Model.html
            em, _ = model(x.to(device))
            return em[0]
        W = max(FRAME_STRIDE, int(round(window_s * SR / FRAME_STRIDE)) * FRAME_STRIDE)
        C = max(0, int(round(context_s * SR / FRAME_STRIDE)) * FRAME_STRIDE)
        wf = W // FRAME_STRIDE
        starts = list(range(0, n, W))
        while len(starts) > 1 and n - starts[-1] < FRAME_RF:
            # A leftover tail shorter than one receptive field has no frame of its own; fold
            # it into the previous window instead of silently dropping that audio (dropping it
            # otherwise depends only on n mod W, not on the words actually needing it, and can
            # turn a good file into a spurious FAILED in align_words()'s frame-count check).
            log.debug('folding %d-sample tail into the previous %.0fs window', n - starts[-1], window_s)
            starts.pop()
        pieces = []
        for s in starts:
            lo, hi = max(0, s - C), min(n, s + W + C)
            if hi - lo < FRAME_RF:                              # whole file shorter than one receptive field
                break
            em, _ = model(x[:, lo:hi].to(device))               # (1, F, C); the bundle wrapper normalises the
            skip = (s - lo) // FRAME_STRIDE                     # waveform and applies log_softmax itself
            pieces.append(em[0, skip:skip + wf])
        em = torch.cat(pieces, 0) if pieces else x.new_zeros((0, 1))
    if em.shape[0] != n_expect:
        log.warning('emission has %d frames, a single pass would give %d', em.shape[0], n_expect)
    return em


def align_words(aligner, tokenizer, emission, transcript: List[str]):
    """List[List[TokenSpan]], one list per aligner word. TokenSpan(token, start, end, score):
    frame indices, end exclusive, score = mean per-frame probability of the token.
    https://github.com/pytorch/audio/blob/main/src/torchaudio/functional/_alignment.py"""
    import torch
    tokens = tokenizer(transcript)
    flat = [t for ts in tokens for t in ts]
    need = len(flat) + sum(1 for a, b in zip(flat, flat[1:]) if a == b)   # CTC needs a blank between repeats
    if need > emission.shape[0]:
        raise ValueError(f'{need} CTC steps needed for {len(flat)} characters but only '
                         f'{emission.shape[0]} frames of audio')
    with torch.inference_mode():
        try:
            return aligner(emission, tokens)      # Aligner.__call__(emission: (T, C), tokens: List[List[int]])
        except RuntimeError as e:                 # e.g. a build without the CUDA forced_align op
            log.warning('aligner failed on %s (%s); retrying on cpu', emission.device, str(e).splitlines()[0])
            return aligner(emission.cpu(), tokens)


# --------------------------------------------------------------------------- #
# Transcript in, transcript out
# --------------------------------------------------------------------------- #

def source_segments(data: dict) -> Tuple[List[dict], bool]:
    """Copy of the source segments with word dicts. A text-only source (words: []) gets
    words from text.split(), each with a leading space like the Whisper tokenizer."""
    from_text = False
    segs = []
    for s in data['segments']:
        words = [dict(w) for w in s.get('words') or []]
        if not words and s.get('text', '').strip():
            words = [{'w': ' ' + t} for t in s['text'].split()]
            from_text = True
        segs.append({'start': s.get('start'), 'end': s.get('end'), 'text': s.get('text', ''), 'words': words})
    return segs, from_text


def retime(segments: List[dict], groups: List[List[str]], word_spans) -> Counter:
    """Write start/end/p into every word from the aligned sub-word spans; segment times from
    words. Returns shift statistics (aligned minus source, when the source had times)."""
    flat = [w for s in segments for w in s['words']]
    assert len(flat) == len(groups)
    k = 0
    last_end: Optional[float] = None
    pending: List[dict] = []                     # empty words waiting for a boundary
    shifts: Counter = Counter()
    dstart, dend = [], []
    for w, g in zip(flat, groups):
        spans = [sp for j in range(len(g)) for sp in word_spans[k + j]]
        k += len(g)
        if 'p' in w:
            w['p_asr'] = w.pop('p')
        if not spans:
            pending.append(w)
            continue
        start = spans[0].start * FRAME_SEC
        end = spans[-1].end * FRAME_SEC
        total = sum(len(sp) for sp in spans)
        if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float)):
            dstart.append(start - w['start']); dend.append(end - w['end'])
        w['start'], w['end'] = round(start, 3), round(end, 3)
        w['p'] = round(sum(sp.score * len(sp) for sp in spans) / total, 4) if total else None
        for pw in pending:                       # empties before the first aligned word sit at its start
            pw['start'] = pw['end'] = round(last_end if last_end is not None else start, 3)
            pw['p'] = None
        pending.clear()
        last_end = end
    for pw in pending:                           # trailing empties
        pw['start'] = pw['end'] = round(last_end if last_end is not None else 0.0, 3)
        pw['p'] = None
    for s in segments:
        if s['words']:
            s['start'], s['end'] = s['words'][0]['start'], s['words'][-1]['end']
    if dstart:
        shifts['start_shift_median'] = float(np.median(dstart))
        shifts['end_shift_median'] = float(np.median(dend))
        shifts['abs_shift_p90'] = float(np.percentile(np.abs(dstart + dend), 90))
    return shifts


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _extra(ap):
    ap.add_argument('--tag-out', default=None, help='output tag (default <tag>+mms)')
    ap.add_argument('--in-dir', default=None, help='where <stem>.<tag>.json lives (default --out-dir)')
    ap.add_argument('--window', type=float, default=30.0, help='seconds of audio per model pass; 0 = whole file')
    ap.add_argument('--context', type=float, default=2.0, help='seconds of context on each side of a window')
    ap.add_argument('--oov', choices=('drop', 'star'), default='drop',
                    help='characters outside the MMS dictionary: drop them or align them as <star>')
    ap.add_argument('--model-dir', default=None, help='checkpoint dir (default $TORCH_HOME/hub/checkpoints)')
    ap.add_argument('--dry-run', action='store_true', help='normalise only: no torch import, nothing written')
    ap.add_argument('--show', type=int, default=0, help='print the first N non-trivial word normalisations')


def _trivial(word: str) -> List[str]:
    t = re.sub(r"[^a-z']", '', word.lower()).strip("'")
    return [t] if t else []


def _release_cuda() -> None:
    """After a failed file (e.g. OOM): give the cached blocks back so the next file gets them."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  best effort, never mask the original error
        pass


def main() -> int:
    args = standard_args('large-v3', extra=_extra)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    tag_out = args.tag_out or f'{args.tag}+mms'
    in_dir = Path(args.in_dir or args.out_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = frozenset(MMS_LABELS)
    if not args.dry_run:
        import torch
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            log.error('--device %s but torch %s reports no CUDA (torch.cuda.is_available() is False): '
                      'a CPU or mismatched wheel in this venv, or not on a GPU node; pass --device cpu to go on',
                      args.device, torch.__version__)
            return 2
        model, tokenizer, aligner, dictionary = load_mms(args.device, args.model_dir)
        allowed = frozenset(k for k in dictionary if k not in (BLANK, STAR))
        if allowed != frozenset(MMS_LABELS):
            log.warning('MMS dictionary differs from the expected labels: %s', sorted(allowed))

    totals: Counter = Counter()
    failed: List[str] = []
    shown = 0
    total_audio = total_dt = 0.0
    n_done = 0
    for path in audio_files(args):
        src = in_dir / f'{path.stem}.{args.tag}.json'
        out = out_dir / f'{path.stem}.{tag_out}.json'
        if not src.exists():
            log.warning('%s: no source transcript %s', path.name, src.name)
            totals['missing'] += 1
            continue
        if out.exists() and not args.force and not args.dry_run:
            log.info('skip %s', out.name)
            continue
        try:
            data = read_transcript(src)
            segments, from_text = source_segments(data)
            orig = [w['w'] for s in segments for w in s['words']]
            if not orig:
                log.warning('%s: source has no words', src.name)
                totals['empty_source'] += 1
                continue
            groups, stats = normalise_words(orig, allowed, args.oov)
            totals.update(stats)
            if args.show and shown < args.show:
                for w, g in zip(orig, groups):
                    if shown >= args.show:
                        break
                    if g != _trivial(w):
                        print(f'  {w.strip()!r:>18} -> {" ".join(g)!r}')
                        shown += 1
            line = (f'{path.name}: {stats["n_words"]} words -> {stats["n_aligner_words"]} aligner words, '
                    f'oov chars {stats["chars_oov"]}, empty words {stats["words_empty"]}')
            if stats['words_empty'] or stats['chars_oov']:
                empties = [w.strip() for w, g in zip(orig, groups) if not g]
                log.warning('%s: %d words normalise to nothing (zero-width span, audio pushed onto the '
                            'neighbours) %s; %d oov chars (%s)', path.name, stats['words_empty'],
                            empties[:12] + (['...'] if len(empties) > 12 else []), stats['chars_oov'],
                            'aligned as <star>' if args.oov == 'star' else 'dropped')
            if args.dry_run:
                log.info(line)
                continue

            audio, _ = load_audio(path)
            duration = len(audio) / SR
            transcript = [t for g in groups for t in g]
            t0 = time.time()
            emission = compute_emission(model, audio, args.device, args.window, args.context)
            t1 = time.time()
            word_spans = align_words(aligner, tokenizer, emission, transcript)
            dt = time.time() - t0
            assert len(word_spans) == len(transcript)
            shifts = retime(segments, groups, word_spans)
            n_frames = int(emission.shape[0])
            extra = {
                'source_tag': args.tag, 'source_file': src.name, 'aligner': 'torchaudio.pipelines.MMS_FA',
                'frame_seconds': FRAME_SEC, 'window_seconds': args.window, 'context_seconds': args.context,
                'oov': args.oov, 'n_frames': n_frames,
                'sample_ratio': (len(audio) / n_frames) if n_frames else None,   # the tutorial's frames->samples ratio, for the record
                'seconds_emission': t1 - t0, 'seconds_align': dt - (t1 - t0),
                'words_from_text': from_text,
                **{k: int(v) for k, v in stats.items()},
                **{k: round(v, 3) for k, v in shifts.items()},
            }
            write_transcript(out, path, tag_out, duration, dt, segments, word_timestamps=True, extra=extra)
        except Exception:  # noqa: BLE001  one bad file must not abort the tag run (bench/README.md)
            log.exception('%s: FAILED (tag %s), going on with the next file', path.name, tag_out)
            failed.append(path.name)
            _release_cuda()
            continue
        total_audio += duration; total_dt += dt; n_done += 1
        log.info('%s; %.1fs audio, %.2fs (emission %.2fs), start shift median %+.3fs, end %+.3fs',
                 line, duration, dt, t1 - t0, shifts.get('start_shift_median', float('nan')),
                 shifts.get('end_shift_median', float('nan')))

    log.info('TOTAL words %d -> aligner words %d, oov chars %d, punct chars %d, empty words %d, '
             'missing sources %d, failed files %d', totals['n_words'], totals['n_aligner_words'],
             totals['chars_oov'], totals['chars_punct'], totals['words_empty'], totals['missing'], len(failed))
    if n_done and total_audio:
        log.info('DONE %d files, RTF %.4f, tag %s', n_done, total_dt / total_audio, tag_out)
    if failed:
        log.error('FAILED %d: %s', len(failed), ' '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
