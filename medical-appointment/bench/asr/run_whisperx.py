"""WhisperX runner: Whisper (faster-whisper, batched over VAD windows) plus
wav2vec2 CTC forced alignment for word timestamps, with a digit-aware
pre-alignment normalisation. Writes the transcript schema in bench/README.md.

Pipeline per file
  1. common.load_audio -> float32 mono 16 kHz (identical samples for every runner)
  2. whisperx.load_model(...).transcribe(audio, batch_size, chunk_size):
     VAD (pyannote segmentation shipped inside the wheel, or silero) -> speech
     regions -> min-cut at chunk_size -> packed into <= chunk_size windows ->
     Whisper decodes each window WITHOUT timestamps. One text per window.
     There are no Whisper-native segments or DTW timestamps in this mode: every
     time in the output comes from the aligner, window edges come from the VAD.
  3. per window: build an ALIGNMENT COPY of the text in which each token is
     spelled with letters only (digits via num2words, units and titles
     expanded, punctuation dropped: "2.5mg," -> "two point five milligrams"),
     keep an index map token -> copy words, call whisperx.align on the copy.
     Each original token keeps its own spelling in the output and gets
     start = first copy word start, end = last copy word end, p = mean CTC
     score of its copy words.
  4. tokens with no alignable letters at all ("—", "...") or inside a window
     whose alignment failed get a time interpolated between their aligned
     neighbours (proportional to token length), p = 0 and "interp": true.
  5. the word stream is cut into sentence segments with common.words_to_segments
     (README rule for word-stream models); --chunk-segments keeps VAD windows.

Why the normalisation (whisperX issues #869 / #98, report 3): the English
aligner WAV2VEC2_ASR_BASE_960H only knows [a-z] and the apostrophe. Stock
whisperx >= 3.8.2 maps every other character (digits, "%", ".", ",", "-") to a
wildcard CTC column (max non-blank emission per frame), so "100" is forced onto
whatever three frames look most like speech, and the "." after every sentence
steals at least one 20 ms frame from the last word. "-" is worse: it maps to
label index 0, the CTC blank. Spelling numbers as words aligns real letters to
real speech; dropping punctuation removes the wildcard frames. --no-normalise
gives the stock behaviour for an A/B (tag suffix +raw).

Install (Linux venv on blackhole; whisperx 3.8.6 pins torch~=2.8 / torchaudio~=2.8,
whose PyPI wheels ship CUDA 12.8, fine for an H100; Python >=3.10,<3.14;
`bench/hpc/env.sh venvs` builds it as /dtu/blackhole/1e/205502/venvs/venv-whisperx)
  python3.11 -m venv $BLACKHOLE/venv-whisperx && . $BLACKHOLE/venv-whisperx/bin/activate
  pip install -U pip
  pip install "whisperx>=3.8.6" num2words soundfile librosa
  (pyannote-audio 4 depends on torchcodec, which needs FFmpeg 4-9 shared libraries.
   UNVERIFIED: whether `import whisperx` fails on gbar without them; if it does,
   `module load ffmpeg` before running. whisperx itself never decodes with them,
   audio comes from common.load_audio.)

Stage the downloads ON THE LOGIN NODE (CPU, no GPU; the compute nodes have no
internet and bench/hpc/asr_bench.lsf runs with HF_HUB_OFFLINE=1):
  export HF_HOME=$BLACKHOLE/hf TORCH_HOME=$BLACKHOLE/torch-home NLTK_DATA=$BLACKHOLE/nltk_data
  python bench/asr/run_whisperx.py --prefetch     # same --model/--align-model/--vad-method as the run
--prefetch fetches the faster-whisper weights (Systran/faster-whisper-large-v3,
~3 GB) into $HF_HOME/hub (or --download-root), the wav2vec2 aligner checkpoint
(BASE_960H ~360 MB, LV60K ~1.2 GB) into $TORCH_HOME/hub/checkpoints (or
--align-dir), nltk punkt_tab (whisperx.align's sentence splitter) into $NLTK_DATA
and, with --vad-method silero, snakers4/silero-vad into $TORCH_HOME/hub.
`bench/hpc/env.sh prefetch` stages the first two but not punkt_tab or silero.
The same three variables must be exported INSIDE the job script (the login
environment does not carry into an LSF job; asr_bench.lsf exports HF_HOME and
TORCH_HOME, NLTK_DATA still has to be added there): the runner refuses to start
when any of the three caches would land under the quota-limited $HOME, because
that ends as 'Disk quota exceeded' mid-download with no scheduler reason.
--allow-home-cache overrides the check (laptop).

Run (from medical-appointment/, with the three cache variables exported)
  python bench/asr/run_whisperx.py --prefetch                 # login node, once per model/aligner/VAD choice
  python bench/asr/run_whisperx.py                            # tag whisperx-large-v3
  python bench/asr/run_whisperx.py --model large-v3-turbo     # tag whisperx-large-v3-turbo
  python bench/asr/run_whisperx.py --no-normalise             # stock whisperx, tag +raw
  python bench/asr/run_whisperx.py --no-vad-merge --limit 3   # one Whisper call per speech region
  python bench/asr/run_whisperx.py --align-model WAV2VEC2_ASR_LARGE_LV60K_960H
  python bench/asr/run_whisperx.py --selftest                 # CPU only, no whisperx/torch import
  python bench/asr/run_whisperx.py --allow-home-cache ...     # laptop only: caches under $HOME are fine there

Caveats
  - GPU only in practice (faster-whisper large-v3 fp16 + wav2vec2 + pyannote VAD);
    the laptop is for --selftest and py_compile.
  - CTC word ends are typically early and starts late by a few 10 ms (report 3);
    nothing here corrects that, model.py owns the offsets.
  - p is the aligner's mean per-character CTC probability, not a Whisper
    token probability; not comparable with faster-whisper's p.
  - --no-normalise passes tokens verbatim; digit-only tokens then get wildcard
    timestamps from whisperx and are marked p=0, "interp": true so that the
    span analysis can tell them apart.
  - No diarization (pyannote speaker-diarization-community-1 needs an HF token
    and is irrelevant to spans). The VAD needs no token: its weights are
    whisperx/assets/pytorch_model.bin inside the wheel.
  - Every network fetch (whisper weights via huggingface_hub, the torchaudio
    aligner checkpoint from download.pytorch.org into $TORCH_HOME/hub/checkpoints,
    nltk punkt_tab, silero via torch.hub) happens on first use unless --prefetch
    staged it; inside the job that is an error, not a download (no internet).
  - One failing file (MP3 decode error, CUDA OOM on a long window at batch 16,
    a CTranslate2 error, a hallucination loop whose token count blows the
    aligner's frames x tokens trellis, a pandas error inside align) is logged as
    ``FAIL <name>`` on stderr with a traceback and skipped; the run goes on and
    exits 1 at the end with a ``FAILED n: names`` line, so the other
    conversations keep their transcripts and the log names what to redo. The
    transcript is written only after both stages succeeded and is removed again
    if the write itself dies, so the skip rule never accepts a half file.
    Windows whose alignment failed do not fail the run (their words carry
    interpolated times, p=0, "interp": true); they are counted per file in
    n_align_failed_chunks and summed in a WARN line at the end.

APIs verified 2026-09-17 against whisperX main (3.8.7rc1) and tag v3.8.6:
  https://github.com/m-bain/whisperX (README usage)
  https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/asr.py        load_model, transcribe
  https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py  load_align_model, align
  https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/vads/pyannote.py, vads/silero.py, vads/vad.py
  https://raw.githubusercontent.com/m-bain/whisperX/main/pyproject.toml         pins
  https://pypi.org/project/whisperx/ (3.8.6, 2026-05-25)  https://pypi.org/project/num2words/ (0.5.14)
  https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/pipelines/_wav2vec2/utils.py (EN labels)
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Timer, audio_files, load_audio, out_path, standard_args,  # noqa: E402
                    words_to_segments, write_transcript)

try:
    # https://pypi.org/project/num2words/  num2words(number, to='cardinal'|'ordinal'|'year', lang='en')
    from num2words import num2words
except ImportError:                      # only required unless --no-normalise
    num2words = None

DEFAULT_MODEL = 'large-v3'
DEFAULT_TAG = 'whisperx-large-v3'
DEFAULT_CHUNK = 30.0
# whisperx/alignment.py DEFAULT_ALIGN_MODELS_TORCH['en']
DEFAULT_ALIGN_EN = 'WAV2VEC2_ASR_BASE_960H'


# --------------------------------------------------------------------------- #
# Pre-alignment normalisation (letters only; used for the alignment copy)
# --------------------------------------------------------------------------- #

# Expansions for letters glued to a number ("100mg", "5pm", "2x"); the copy is
# only used for alignment, the transcript keeps the model's spelling.
_ATTACHED: Dict[str, str] = {
    'mg': 'milligrams', 'mcg': 'micrograms', 'ug': 'micrograms', 'g': 'grams', 'gm': 'grams',
    'kg': 'kilograms', 'ml': 'millilitres', 'l': 'litres', 'cc': 'c c', 'cm': 'centimetres',
    'mm': 'millimetres', 'm': 'metres', 'km': 'kilometres', 'iu': 'international units',
    'u': 'units', 'mmhg': 'millimetres of mercury', 'bpm': 'beats per minute',
    'h': 'hours', 'hr': 'hours', 'hrs': 'hours', 'min': 'minutes', 'mins': 'minutes',
    's': 'seconds', 'sec': 'seconds', 'secs': 'seconds', 'ms': 'milliseconds',
    'x': 'times', 'am': 'a m', 'pm': 'p m', 'k': 'thousand', 'c': 'celsius', 'f': 'fahrenheit',
    'yo': 'year old', 'lb': 'pounds', 'lbs': 'pounds', 'oz': 'ounces',
}
# Whole-token abbreviations ("mg", "Dr.", "hrs"). Single letters and the
# ambiguous ones (am = verb, ms = multiple sclerosis, cc) are deliberately absent.
_STANDALONE: Dict[str, str] = {k: v for k, v in _ATTACHED.items()
                               if len(k) > 1 and k not in ('am', 'pm', 'ms', 'cc', 'yo')}
_STANDALONE.update({'dr': 'doctor', 'mr': 'mister', 'mrs': 'missus', 'vs': 'versus',
                    'tsp': 'teaspoon', 'tbsp': 'tablespoon', 'etc': 'etcetera'})
_CURRENCY = {'$': 'dollars', '£': 'pounds', '€': 'euros'}

_CURRENCY_RE = re.compile(r'([$£€])\s*(\d+(?:\.\d+)?)')
_THOUSANDS = re.compile(r'(?<=\d),(?=\d{3}(?!\d))')
_FRACTION = re.compile(r'(?<!\d)(1/2|1/4|3/4|1/3|2/3)(?!\d)')
_FRACTIONS = {'1/2': 'one half', '1/4': 'one quarter', '3/4': 'three quarters',
              '1/3': 'one third', '2/3': 'two thirds'}
_ORDINAL = re.compile(r'(?<![\d.])(\d+)(st|nd|rd|th)(?![a-z])')
_TIME = re.compile(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)')
_NUM_ATTACHED = re.compile(r'(\d+(?:\.\d+)?)([a-z]+)')
_ALPHA_DIGIT = re.compile(r'([a-z])(\d)')
_NUMBER = re.compile(r'\d+(?:\.\d+)?')
_LETTERS = re.compile(r"[a-z']+")
_HAS_ALPHA = re.compile(r'[a-z]')


def spell_number(s: str) -> str:
    """'100' -> 'one hundred', '2.5' -> 'two point five', '2019' -> 'twenty nineteen'."""
    try:
        if '.' in s:
            return num2words(float(s))
        n = int(s)
        if len(s) == 4 and 1900 <= n <= 2099:
            return num2words(n, to='year')
        return num2words(n)
    except Exception:                                    # absurd digit strings
        return ' '.join(num2words(int(d)) for d in s if d.isdigit())


def _ordinal(m: re.Match) -> str:
    try:
        return num2words(int(m.group(1)), to='ordinal')
    except Exception:
        return spell_number(m.group(1))


def _clock(m: re.Match) -> str:
    h, mm = int(m.group(1)), m.group(2)
    if mm == '00':
        return num2words(h)
    if mm[0] == '0':
        return f'{num2words(h)} oh {num2words(int(mm))}'
    return f'{num2words(h)} {num2words(int(mm))}'


def spell_token(token: str) -> List[str]:
    """Letter-only words ([a-z']) standing in for one transcript token in the
    alignment copy. May be empty (pure punctuation) or several words."""
    t = token.lower().replace('µ', 'u').replace('μ', 'u')
    t = _CURRENCY_RE.sub(lambda m: f'{m.group(2)} {_CURRENCY[m.group(1)]}', t)
    t = (t.replace('%', ' percent ').replace('°', ' degrees ')
          .replace('&', ' and ').replace('+', ' plus '))
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode()
    t = _THOUSANDS.sub('', t)
    t = _FRACTION.sub(lambda m: f' {_FRACTIONS[m.group(1)]} ', t)
    t = _ORDINAL.sub(_ordinal, t)
    t = _TIME.sub(_clock, t)
    t = _NUM_ATTACHED.sub(lambda m: f'{m.group(1)} {_ATTACHED.get(m.group(2), m.group(2))}', t)
    t = _ALPHA_DIGIT.sub(r'\1 \2', t)
    t = _NUMBER.sub(lambda m: spell_number(m.group(0)), t)
    words: List[str] = []
    for w in _LETTERS.findall(t):
        words.extend(_STANDALONE.get(w, w).split())
    return [w for w in words if _HAS_ALPHA.search(w)]


def alignable_chars(align_meta: dict) -> set:
    """Letters the aligner can emit: single-char dictionary keys that are
    letters or the apostrophe. For WAV2VEC2_ASR_BASE_960H the labels are
    ('-' blank, '|' space, A-Z, "'"), lower-cased by whisperx.load_align_model."""
    keys = align_meta.get('dictionary', {})
    return {c for c in keys if len(c) == 1 and (c.isalpha() or c == "'")}


def build_alignment_copy(tokens: Sequence[str], alphabet: set, normalise: bool = True
                         ) -> Tuple[List[str], List[Tuple[int, int]], int]:
    """Returns (copy_words, index_map, n_changed). index_map[i] = (a, b): token i
    is represented by copy_words[a:b] (empty range if nothing alignable).
    Raw mode passes tokens verbatim (stock whisperx behaviour)."""
    copy: List[str] = []
    index_map: List[Tuple[int, int]] = []
    n_changed = 0
    for tok in tokens:
        if normalise:
            ws = []
            for w in spell_token(tok):
                w = ''.join(c for c in w if c in alphabet)
                if _HAS_ALPHA.search(w):
                    ws.append(w)
            plain = ''.join(c for c in tok.lower() if c in alphabet)
            if ''.join(ws) != plain:
                n_changed += 1
        else:
            ws = [tok] if tok.strip() else []
        a = len(copy)
        copy.extend(ws)
        index_map.append((a, len(copy)))
    return copy, index_map, n_changed


# --------------------------------------------------------------------------- #
# Mapping aligned copy words back onto the original tokens
# --------------------------------------------------------------------------- #

def match_words(copy_words: Sequence[str], aligned: Sequence[dict]) -> Optional[List[dict]]:
    """whisperx.align returns one dict per non-empty space-separated word of the
    text, in order, 'word' being the exact characters. Match by text so a
    reordering or a dropped word cannot silently shift the map."""
    if not aligned:
        return None
    if len(aligned) == len(copy_words) and all(
            str(a.get('word', '')).lower() == w.lower() for a, w in zip(aligned, copy_words)):
        return list(aligned)
    out: List[dict] = []
    j = 0
    for w in copy_words:
        k = next((i for i in range(j, len(aligned))
                  if str(aligned[i].get('word', '')).lower() == w.lower()), None)
        if k is None:
            out.append({})
        else:
            out.append(aligned[k])
            j = k + 1
    return out if any(out) else None


def merge_alignment(tokens: Sequence[str], copy_words: Sequence[str],
                    index_map: Sequence[Tuple[int, int]], aligned: Sequence[dict],
                    t1: float, t2: float) -> Tuple[List[dict], dict]:
    """Assign every token a (start, end, p). Tokens whose copy words carry no
    timestamp are filled between neighbours, proportional to token length."""
    n = len(tokens)
    m = match_words(copy_words, aligned)
    times: List[Optional[Tuple[float, float]]] = [None] * n
    probs = [0.0] * n
    interp = [True] * n
    n_wild = 0
    if m is not None:
        for i, (a, b) in enumerate(index_map):
            got = [(m[j], copy_words[j]) for j in range(a, b)
                   if j < len(m) and 'start' in m[j] and 'end' in m[j]]
            if not got:
                continue
            times[i] = (min(float(x['start']) for x, _ in got),
                        max(float(x['end']) for x, _ in got))
            letters = [x for x, w in got if _HAS_ALPHA.search(w)]
            if letters:                              # real letter alignment
                probs[i] = round(statistics.fmean(float(x.get('score', 0.0)) for x in letters), 3)
                interp[i] = False
            else:                                    # whisperx wildcard (raw mode digits)
                n_wild += 1
    n_filled = sum(1 for t in times if t is None)
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else t1
        right = times[j][0] if j < n else t2
        right = max(right, left)
        lens = [max(1, len(tokens[k])) for k in range(i, j)]
        total = float(sum(lens))
        cur = left
        for k, ln in zip(range(i, j), lens):
            nxt = cur + (right - left) * ln / total
            times[k] = (round(cur, 3), round(nxt, 3))
            cur = nxt
        i = j
    words = []
    for i, tok in enumerate(tokens):
        s, e = times[i]
        w = {'w': ' ' + tok, 'start': s, 'end': max(s, e), 'p': probs[i]}
        if interp[i]:
            w['interp'] = True
        words.append(w)
    stats = {'n_filled': n_filled, 'n_wildcard': n_wild, 'failed': m is None,
             'filled': [tokens[i] for i in range(n) if interp[i]]}
    return words, stats


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def _no_merge_vad(args, vad_options: dict):
    """A VAD whose merge step yields one Whisper window per speech region
    instead of packing regions into <= chunk_size windows. Goes through the
    documented load_model(vad_model=...) hook; only merge_chunks is overridden.
    Pyannote.merge_chunks / Silero.merge_chunks verified in
    https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/vads/pyannote.py
    and .../vads/silero.py; transcribe() only reads seg['start'] / seg['end']."""
    import torch
    from whisperx.vads import Pyannote, Silero
    from whisperx.vads.pyannote import Binarize

    if args.vad_method == 'silero':
        class SileroNoMerge(Silero):
            @staticmethod
            def merge_chunks(segments_list, chunk_size, onset=0.5, offset=None):
                return [{'start': s.start, 'end': s.end, 'segments': [(s.start, s.end)]}
                        for s in segments_list]
        return SileroNoMerge(**vad_options)

    class PyannoteNoMerge(Pyannote):
        @staticmethod
        def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
            # same binarisation as stock (min-cut at chunk_size), no packing
            binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
            turns = binarize(segments).get_timeline()
            return [{'start': t.start, 'end': t.end, 'segments': [(t.start, t.end)]} for t in turns]

    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    return PyannoteNoMerge(device, token=None, **vad_options)


def load_models(args):
    import whisperx  # lazy: heavy import, absent on the laptop

    vad_options = {'vad_onset': args.vad_onset, 'vad_offset': args.vad_offset,
                   'chunk_size': args.chunk_size}
    asr_options = {'beam_size': args.beam, 'suppress_numerals': args.suppress_numerals}
    if args.initial_prompt:
        asr_options['initial_prompt'] = args.initial_prompt
    vad_model = _no_merge_vad(args, vad_options) if args.no_vad_merge else None
    # https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/asr.py
    # load_model(whisper_arch, device, device_index=0, compute_type='default', asr_options=None,
    #            language=None, vad_model=None, vad_method='pyannote', vad_options=None, model=None,
    #            task='transcribe', download_root=None, local_files_only=False, threads=4, use_auth_token=None)
    asr = whisperx.load_model(
        args.model, args.device, compute_type=args.compute_type, asr_options=asr_options,
        language='en', vad_model=vad_model, vad_method=args.vad_method, vad_options=vad_options,
        download_root=args.download_root)
    # https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py
    # load_align_model(language_code, device, model_name=None, model_dir=None, model_cache_only=False)
    #   -> (model, {'language', 'dictionary', 'type'})
    align_model, align_meta = whisperx.load_align_model(
        'en', args.device, model_name=args.align_model, model_dir=args.align_dir)
    return asr, align_model, align_meta


def align_chunks(chunks: Sequence[dict], audio, align_model, align_meta, alphabet: set, args
                 ) -> Tuple[List[dict], dict]:
    import whisperx

    segments: List[dict] = []
    vad_chunks: List[dict] = []
    filled_words: List[str] = []
    n_words = n_changed = n_filled = n_wild = n_failed = 0
    for ch in chunks:
        # whisperx.FasterWhisperPipeline.transcribe() only unwraps 'text' and
        # 'avg_logprob' from single-item lists to scalars when batch_size is in
        # (0, 1, None); for any other batch_size (the runner's own default 16)
        # both fields stay lists on every chunk dict in res['segments'].
        # https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/asr.py
        text = ch.get('text', '')
        if isinstance(text, list):
            text = text[0] if text else ''
        tokens = str(text).split()
        t1, t2 = float(ch['start']), float(ch['end'])
        info = {'start': t1, 'end': t2, 'n_words': len(tokens)}
        avg_logprob = ch.get('avg_logprob')
        if isinstance(avg_logprob, list):
            avg_logprob = avg_logprob[0] if avg_logprob else None
        if avg_logprob is not None:
            info['avg_logprob'] = round(float(avg_logprob), 4)
        vad_chunks.append(info)
        if not tokens:
            continue
        copy_words, index_map, changed = build_alignment_copy(
            tokens, alphabet, normalise=not args.no_normalise)
        aligned: List[dict] = []
        if copy_words:
            # https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/alignment.py
            # align(transcript, model, align_model_metadata, audio, device, interpolate_method='nearest',
            #       return_char_alignments=False, ...) -> {'segments': [...], 'word_segments':
            #       [{'word', 'start'?, 'end'?, 'score'?}, ...]}; words are text.split(' '), empties skipped.
            # One input segment per call so word_segments maps 1:1 onto copy_words.
            res = whisperx.align(
                [{'text': ' '.join(copy_words), 'start': t1, 'end': t2}],
                align_model, align_meta, audio, args.device,
                interpolate_method='nearest', return_char_alignments=False)
            aligned = list(res.get('word_segments', []))
        words, st = merge_alignment(tokens, copy_words, index_map, aligned, t1, t2)
        n_words += len(words)
        n_changed += changed
        n_filled += st['n_filled']
        n_wild += st['n_wildcard']
        n_failed += int(st['failed'])
        filled_words.extend(st['filled'])
        if args.chunk_segments:
            segments.append({'start': t1, 'end': t2,
                             'text': ''.join(w['w'] for w in words), 'words': words})
        else:
            segments.extend(words_to_segments(words))
    extra = {
        'n_words': n_words, 'n_changed': n_changed, 'n_filled': n_filled,
        'n_wildcard': n_wild, 'n_align_failed_chunks': n_failed,
        'filled_words': filled_words[:50], 'vad_chunks': vad_chunks,
    }
    return segments, extra


# --------------------------------------------------------------------------- #
# Caches: nothing under $HOME (bench/README.md 'Cluster'); --prefetch stages them
# --------------------------------------------------------------------------- #

# Where each library puts its downloads when the variable is unset:
#   huggingface_hub.constants  HF_HOME = $XDG_CACHE_HOME/huggingface (~/.cache); hub cache HF_HOME/hub
#   torch.hub._get_torch_home  $XDG_CACHE_HOME/torch (~/.cache); checkpoints TORCH_HOME/hub/checkpoints
#   nltk.downloader            default_download_dir: first writable NLTK_DATA entry, else ~/nltk_data
_CACHE_DEFAULTS: Dict[str, Tuple[str, str]] = {
    'HF_HOME': ('.cache', 'huggingface'), 'TORCH_HOME': ('.cache', 'torch'), 'NLTK_DATA': ('', 'nltk_data')}


def cache_root(var: str) -> Tuple[Path, bool]:
    """(directory the library will use, whether `var` is set). NLTK_DATA may be
    a os.pathsep list; the first entry is where nltk.download writes."""
    val = os.environ.get(var, '').split(os.pathsep)[0].strip()
    if val:
        return Path(val).expanduser(), True
    cache, sub = _CACHE_DEFAULTS[var]
    xdg = os.environ.get('XDG_CACHE_HOME', '').strip()
    if cache and xdg:
        return Path(xdg).expanduser() / sub, False
    return Path.home() / cache / sub, False


def check_caches(args) -> Dict[str, Path]:
    """Refuse to run when any model cache would land under the quota-limited
    $HOME. The login environment does not carry into an LSF job, so an unset
    variable there means a 3 GB download into ~/.cache mid-job that ends as
    'Disk quota exceeded' with no scheduler reason. --allow-home-cache skips it."""
    roots = {var: cache_root(var) for var in _CACHE_DEFAULTS}
    dirs = {var: d for var, (d, _) in roots.items()}
    try:
        home = Path.home().resolve()
    except RuntimeError:                     # no $HOME at all: nothing to protect
        return dirs
    bad = []
    for var, (d, explicit) in roots.items():
        try:
            under_home = d.resolve().is_relative_to(home)
        except OSError:
            under_home = False
        if under_home:
            bad.append(f'{var}={d if explicit else "(unset)"} -> {d}')
    if bad and not args.allow_home_cache:
        sys.exit('model caches under $HOME: ' + '; '.join(bad)
                 + '\nexport HF_HOME, TORCH_HOME and NLTK_DATA to blackhole (bench/README.md, '
                   'bench/hpc/asr_bench.lsf) or pass --allow-home-cache')
    return dirs


def effective_dirs(args, caches: Dict[str, Path]) -> Dict[str, Path]:
    """The directories a run reads from and --prefetch writes to. The defaults
    stay the libraries' own (HF_HOME/hub, TORCH_HOME/hub/checkpoints), which is
    where bench/hpc/env.sh prefetch put the weights; --download-root and
    --align-dir are passed through as snapshot_download(cache_dir=) and
    torch.hub model_dir= and therefore name the final directory themselves."""
    hub = os.environ.get('HF_HUB_CACHE', '').strip()
    return {
        'whisper': Path(args.download_root) if args.download_root
        else (Path(hub).expanduser() if hub else caches['HF_HOME'] / 'hub'),
        'align': Path(args.align_dir) if args.align_dir else caches['TORCH_HOME'] / 'hub' / 'checkpoints',
        'nltk': caches['NLTK_DATA'],
        'torch_hub': caches['TORCH_HOME'] / 'hub',
    }


def prefetch(args, dirs: Dict[str, Path]) -> int:
    """Stage every network fetch of a run on the login node (CPU, no GPU) so the
    job finds all of it in the caches. Idempotent: cached files are not refetched."""
    import nltk
    import whisperx
    # faster_whisper/utils.py: download_model(size_or_id, ..., cache_dir=None): _MODELS
    # (large-v3 -> Systran/faster-whisper-large-v3) then snapshot_download; the same
    # resolver WhisperModel.__init__ runs with cache_dir=download_root (verified).
    from faster_whisper.utils import download_model

    if os.environ.get('HF_HUB_OFFLINE', '').strip().lower() in ('1', 'true', 'yes'):
        print('WARN HF_HUB_OFFLINE is set: nothing can be fetched from the hub (unset it on the login node)',
              file=sys.stderr, flush=True)
    print(f'prefetch: whisper -> {dirs["whisper"]}, aligner -> {dirs["align"]}, '
          f'nltk -> {dirs["nltk"]}, torch hub -> {dirs["torch_hub"]}', flush=True)
    # 1. Whisper CTranslate2 weights
    if Path(args.model).is_dir():
        print(f'  whisper {args.model}: local directory, nothing to fetch', flush=True)
    else:
        p = download_model(args.model, cache_dir=args.download_root)
        print(f'  whisper {args.model} -> {p}', flush=True)
    # 2. Aligner: torchaudio bundle.get_model(dl_kwargs={'model_dir': model_dir}) for a
    #    pipeline name, Wav2Vec2ForCTC.from_pretrained (HF cache) for a repo id.
    _, meta = whisperx.load_align_model('en', 'cpu', model_name=args.align_model, model_dir=args.align_dir)
    print(f'  aligner {args.align_model or DEFAULT_ALIGN_EN} ok ({meta.get("type")}, '
          f'{len(alignable_chars(meta))} letters)', flush=True)
    # 3. punkt_tab: whisperx.align does nltk_load('tokenizers/punkt_tab/<lang>.pickle') and
    #    nltk.download('punkt_tab', quiet=True) on LookupError, which returns False offline.
    dirs['nltk'].mkdir(parents=True, exist_ok=True)
    if not nltk.download('punkt_tab', download_dir=str(dirs['nltk']), quiet=True):
        sys.exit(f'nltk punkt_tab download into {dirs["nltk"]} failed')
    print(f'  nltk punkt_tab -> {nltk.data.find("tokenizers/punkt_tab/english")}', flush=True)
    # 4. Silero VAD: whisperx/vads/silero.py fetches it with exactly this call.
    if args.vad_method == 'silero':
        import torch
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False,
                       onnx=False, trust_repo=True)
        print(f'  silero-vad -> {torch.hub.get_dir()}', flush=True)
    print('prefetch done', flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_args(ap) -> None:
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help='faster-whisper model name or CT2 repo (large-v3, large-v3-turbo, distil-large-v3)')
    ap.add_argument('--compute-type', default='default', help='default = float16 on cuda, float32 on cpu')
    ap.add_argument('--batch-size', type=int, default=16, help='VAD windows decoded per batch')
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--chunk-size', type=float, default=DEFAULT_CHUNK,
                    help='max VAD window length in s (min-cut and merge window), whisperx default 30')
    ap.add_argument('--no-vad-merge', action='store_true',
                    help='one Whisper window per VAD speech region (no packing to chunk-size)')
    ap.add_argument('--vad-method', default='pyannote', choices=['pyannote', 'silero'])
    ap.add_argument('--vad-onset', type=float, default=0.500)
    ap.add_argument('--vad-offset', type=float, default=0.363)
    ap.add_argument('--align-model', default=None,
                    help='torchaudio pipeline name or HF wav2vec2 repo; default WAV2VEC2_ASR_BASE_960H')
    ap.add_argument('--align-dir', default=None, help='torch.hub checkpoint dir for the aligner')
    ap.add_argument('--download-root', default=None, help='faster-whisper weights dir (else HF cache)')
    ap.add_argument('--suppress-numerals', action='store_true',
                    help='whisperx asr option: forbid digit tokens so Whisper spells numbers (+nonum)')
    ap.add_argument('--initial-prompt', default=None, help='Whisper initial_prompt, e.g. a drug list')
    ap.add_argument('--no-normalise', action='store_true',
                    help='align the raw text (stock whisperx; digits get wildcard times, marked p=0)')
    ap.add_argument('--chunk-segments', action='store_true',
                    help='keep VAD windows as segments instead of sentence segments')
    ap.add_argument('--selftest', action='store_true', help='CPU test of the normaliser and mapping')
    ap.add_argument('--prefetch', action='store_true',
                    help='login node, CPU: download whisper weights, aligner checkpoint, nltk punkt_tab '
                         '(and silero) into the caches, then exit')
    ap.add_argument('--allow-home-cache', action='store_true',
                    help='do not refuse HF_HOME/TORCH_HOME/NLTK_DATA unset or under $HOME (laptop)')


def derive_tag(args) -> str:
    clean = lambda s: re.sub(r'[^a-z0-9.-]+', '-', s.split('/')[-1].lower())  # noqa: E731
    tag = f'whisperx-{clean(args.model)}'
    if args.align_model:
        tag += '+' + clean(args.align_model)
    if args.no_normalise:
        tag += '+raw'
    if args.suppress_numerals:
        tag += '+nonum'
    if args.vad_method != 'pyannote':
        tag += f'+{args.vad_method}'
    if args.no_vad_merge:
        tag += '+nomerge'
    elif args.chunk_size != DEFAULT_CHUNK:
        tag += f'+chunk{args.chunk_size:g}'
    if args.chunk_segments:
        tag += '+chunkseg'
    return tag


def selftest() -> int:
    alphabet = set("abcdefghijklmnopqrstuvwxyz'")
    cases = {
        '100': ['one', 'hundred'], '2.5mg,': ['two', 'point', 'five', 'milligrams'],
        'Dr.': ['doctor'], '10:30': ['ten', 'thirty'], '1st': ['first'], "don't": ["don't"],
        'COVID-19': ['covid', 'nineteen'], '—': [], '...': [], 'B12': ['b', 'twelve'],
        '5%': ['five', 'percent'], '1,000': ['one', 'thousand'], '2019': ['twenty', 'nineteen'],
        'mg': ['milligrams'], '$20': ['twenty', 'dollars'], 'morning.': ['morning'],
        '0.5': ['zero', 'point', 'five'], '1/2': ['one', 'half'], '7:05': ['seven', 'oh', 'five'],
        'twice-daily': ['twice', 'daily'], 'am': ['am'], '2x': ['two', 'times'],
        '(paracetamol)': ['paracetamol'], 'café': ['cafe'], '3rd': ['third'],
    }
    bad = 0
    for tok, want in cases.items():
        got = spell_token(tok)
        ok = got == want
        bad += not ok
        print(f'  {"ok " if ok else "BAD"} {tok!r:16} -> {got}' + ('' if ok else f'   want {want}'))
    tokens = ['Take', '100', 'mg,', 'twice', '—', 'daily.', '2.5', 'ml']
    copy, idx, changed = build_alignment_copy(tokens, alphabet)
    print(f'  copy={copy}\n  map={idx} changed={changed}')
    assert copy == ['take', 'one', 'hundred', 'milligrams', 'twice', 'daily', 'two', 'point',
                    'five', 'millilitres'], copy
    # fake aligner output, 0.2 s per copy word, one word without a timestamp
    aligned = []
    for j, w in enumerate(copy):
        d = {'word': w, 'score': 0.9}
        if w != 'point':
            d.update(start=round(1.0 + 0.2 * j, 3), end=round(1.18 + 0.2 * j, 3))
        aligned.append(d)
    words, st = merge_alignment(tokens, copy, idx, aligned, 0.5, 4.0)
    for w in words:
        print(f'  {w}')
    assert words[1]['start'] == 1.2 and words[1]['end'] == 1.58 and 'interp' not in words[1]
    assert words[4].get('interp') and words[4]['p'] == 0.0
    assert words[3]['end'] <= words[4]['start'] <= words[4]['end'] <= words[5]['start']
    assert words[6]['start'] == 2.2 and words[6]['end'] == 2.78    # 'point' had no time, neighbours do
    assert st['n_filled'] == 1 and not st['failed']
    words, st = merge_alignment(tokens, copy, idx, [], 0.5, 4.0)  # whole window failed
    assert st['failed'] and all(w.get('interp') for w in words)
    assert words[0]['start'] == 0.5 and words[-1]['end'] == 4.0
    assert all(words[i]['end'] <= words[i + 1]['start'] + 1e-9 for i in range(len(words) - 1))
    segs = words_to_segments(words)
    assert len(segs) == 2 and segs[0]['text'].strip().endswith('daily.'), segs
    print(f'  selftest: {"FAILED" if bad else "ok"} ({len(cases)} spell cases, mapping ok)')
    return 1 if bad else 0


def main() -> int:
    args = standard_args(DEFAULT_TAG, extra=add_args)
    if args.selftest:
        return selftest()
    if not args.no_normalise and num2words is None:
        sys.exit('num2words is not installed: pip install num2words (or use --no-normalise)')
    if args.tag == DEFAULT_TAG:
        args.tag = derive_tag(args)
    dirs = effective_dirs(args, check_caches(args))      # before anything can download
    if args.prefetch:
        return prefetch(args, dirs)
    t0 = time.time()
    asr, align_model, align_meta = load_models(args)
    alphabet = alignable_chars(align_meta)
    try:
        import importlib.metadata as md
        version = md.version('whisperx')
    except Exception:
        version = 'unknown'
    print(f'loaded {args.model} + {args.align_model or DEFAULT_ALIGN_EN} (whisperx {version}) '
          f'in {time.time() - t0:.1f}s; tag {args.tag}; alphabet {"".join(sorted(alphabet))}', flush=True)
    print(f'caches: whisper {dirs["whisper"]}, aligner {dirs["align"]}, nltk {dirs["nltk"]}, '
          f'torch hub {dirs["torch_hub"]}', flush=True)

    total_audio = total_dt = 0.0
    n_done = n_failed_windows = 0
    failed: List[str] = []
    for path in audio_files(args):
        out = out_path(args, path)
        if out.exists() and not args.force:
            print(f'skip {path.name}', flush=True)
            continue
        try:
            audio, sr = load_audio(path)
            duration = len(audio) / sr
            with Timer() as t_all:
                with Timer() as t_tr:
                    # transcribe(audio: str|np.ndarray, batch_size=None, num_workers=0, language=None,
                    #            task=None, chunk_size=30, ...) -> {'segments': [{'text','start','end',
                    #            'avg_logprob'}], 'language'}   (whisperx/asr.py, verified)
                    res = asr.transcribe(audio, batch_size=args.batch_size, language='en',
                                         chunk_size=args.chunk_size)
                with Timer() as t_al:
                    segments, extra = align_chunks(res['segments'], audio, align_model, align_meta,
                                                   alphabet, args)
            extra.update({
                'runner': 'whisperx', 'whisperx_version': version, 'whisper_model': args.model,
                'align_model': args.align_model or DEFAULT_ALIGN_EN, 'vad_method': args.vad_method,
                'vad_merge': not args.no_vad_merge, 'chunk_size': args.chunk_size,
                'normalise': not args.no_normalise, 'suppress_numerals': args.suppress_numerals,
                'seconds_transcribe': round(t_tr.seconds, 3), 'seconds_align': round(t_al.seconds, 3),
            })
            # Written only once both stages succeeded, and removed again if the write
            # itself dies (disk quota): the skip rule above must never accept a half file.
            try:
                write_transcript(out, path, args.tag, duration, t_all.seconds, segments,
                                 word_timestamps=True, extra=extra)
            except BaseException:
                out.unlink(missing_ok=True)
                raise
        except Exception as e:  # noqa: BLE001  one bad file must not abort the tag run
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            if 'cuda' in str(e).lower() or 'out of memory' in str(e).lower():
                try:                                     # a clean allocator for the next file
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            continue
        total_audio += duration
        total_dt += t_all.seconds
        n_done += 1
        n_failed_windows += int(extra['n_align_failed_chunks'])
        print(f'{path.name}: {duration:6.1f}s audio, {t_all.seconds:5.1f}s '
              f'({t_tr.seconds:.1f} asr + {t_al.seconds:.1f} align), '
              f'{len(res["segments"])} windows, {len(segments)} segs, {extra["n_words"]} words, '
              f'{extra["n_changed"]} respelled, {extra["n_filled"]} filled, '
              f'{extra["n_align_failed_chunks"]} failed windows', flush=True)
    if n_done and total_audio:
        print(f'DONE {n_done} files, RTF {total_dt / total_audio:.3f}, '
              f'mean {total_dt / n_done:.1f}s per conversation', flush=True)
    if n_failed_windows:
        print(f'WARN {n_failed_windows} windows failed alignment across the run (their words carry '
              f'interpolated times, p=0, "interp": true; n_align_failed_chunks per transcript)',
              file=sys.stderr, flush=True)
    if failed:
        print(f'FAILED {len(failed)}: {" ".join(failed)}', file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
