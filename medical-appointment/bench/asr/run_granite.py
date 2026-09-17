"""IBM Granite Speech 4.1 runner: native word timestamps (plus) or punctuated text (base).

What it does
------------
Default mode (tag ``granite-4.1-2b-plus``): runs ``ibm-granite/granite-speech-4.1-2b-plus``
in its "Timestamps" prompt mode, parses the ``[T:N]`` tags into words with
start/end seconds and writes the bench transcript schema (bench/README.md).

``--punctuate`` (tag ``granite-4.1-2b-plus+punct``): additionally runs the base
model ``ibm-granite/granite-speech-4.1-2b`` (punctuation + capitalisation, better
WER) on the same audio and transfers its spelling/punctuation onto the
timestamped words with a difflib alignment, so ``common.words_to_segments`` can
cut sentence units at terminal punctuation. Times still come only from the plus
model. Two 2B models are resident at once (~10 GB bf16).

``--text-only`` (tag ``granite-4.1-2b``): reference-quality transcript from the
base model, ``word_timestamps: false``, one segment 0..duration, ``words: []``.
With ``--model ibm-granite/granite-speech-4.1-2b-plus --text-only`` the plus
model's plain ASR prompt is used instead (tag ``granite-4.1-2b-plus+text``).

Install (Linux venv, H100). Verified against the plus model card
(https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus, "supported
natively in transformers>=5.8"; its own line is
``pip install torchaudio datasets accelerate torchcodec``) and the transformers
source (modeling_granite_speech.py needs peft only when ``config.has_lora_adapter``
is true; both 4.1 config.json files say ``false``, so peft is not installed; the
feature extractor hard-requires torchaudio)::

    pip install "transformers>=5.8" torch torchaudio accelerate soundfile numpy librosa

librosa is NOT optional: every MP3 in data/audio is 44.1 kHz and
common.load_audio resamples to 16 kHz with librosa (its last fallback spawns
ffmpeg, which a compute node does not have on PATH). GraniteSpeechFeatureExtractor
assumes 16 kHz and never resamples. main() therefore decodes the first file
before loading any weights, so a missing resampler fails in seconds instead of
after two model loads, and asserts 16 kHz on every file.

Example::

    python bench/asr/run_granite.py --limit 2                       # plus, timestamps
    python bench/asr/run_granite.py --punctuate                     # plus times + base text
    python bench/asr/run_granite.py --text-only                     # base, reference text
    python bench/asr/run_granite.py --selftest                      # parser only, no GPU
    HF_HOME=/dtu/blackhole/1e/205502/hf python bench/asr/run_granite.py --force

Timestamp format (verified on the plus model card, quoted)
----------------------------------------------------------
* Prompt: ``<|audio|> Timestamps: Transcribe the speech. After each word, add a
  timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]``
* "the model adds timestamp tags after each word indicating the END of the word"
  -> tokens mark word ENDS only, never starts.
* "[T:N] where N is an integer number indicating the time in centiseconds";
  "only the last three digits of N are provided. This causes a rollover after
  10 seconds"; "N = round(t*100) mod 1000".
* "Silences are transcribed as ``_`` and a timestamp tag also indicates their end."
* "works well with audio segments up to 9 minutes long for ASR and SAA, and up
  to 3.5 minutes for timestamps" (our longest conversation is 3.5 min).
* "Unlike the base model, the plus model doesn't provide punctuation and
  capitalization" -> in default mode every word is lowercase and unpunctuated,
  so segments are cut at model-emitted silences >= --split-pause instead of at
  sentence punctuation (use --punctuate for punctuation-based units).

How word starts are derived (the model gives one boundary per token)
--------------------------------------------------------------------
start(word) = end of the previous token, where a token is a word or a ``_``
silence. A ``_`` therefore gives the next word a real start (the silence's end);
a word directly after another word starts where that word ended (i.e. zero
gap). The first token of the audio (or of a chunk) starts at 0.0 (chunk start).
Word ends are the model's tags unchanged. If the model emits several words
before one tag, the untagged words get ends linearly interpolated between the
previous end and the tag (counted in ``stats.n_untagged_words``). Words after
the final tag (only happens on truncation) get start = end = last known end
(counted in ``stats.n_trailing_untagged``). Nothing else is corrected here.

Rollover reconstruction: absolute_cs = N + 1000 * k. The card's reference loop
adds 10 s whenever a tag is smaller than the previous one. That turns a small
non-monotonic slip (e.g. 350 -> 340) into a +10 s error for the rest of the
file, so this runner only rolls over when the backward jump is at least
--rollover-min-jump centiseconds (default 100 = 1 s). A smaller backward jump
(or what is left of a larger one after the rollover) is counted in
``stats.n_regressions``: the model's value is kept as ``raw_cs`` on that token
and the token's end is clamped to the previous end (a zero-length word), so no
word or segment ever has end < start (span_ceiling.py takes a unit's end from
its last word and would score an inverted unit 0). ``--rollover-min-jump 1``
reproduces the card's rule exactly. A silence longer than 10 s that crosses a
rollover boundary is ambiguous for both rules (one cycle short).

Known caveats
-------------
* LLM decoder: one ``[T:N]`` tag per word roughly doubles the generated length;
  expect a few thousand tokens and 1-3 minutes per conversation with plain
  HF generate on one H100 (no vLLM here; vLLM >= 0.23 is the fast path).
* A third-party test reports drift "at segment boundaries, or after silences"
  (https://kappi-coval.github.io/); the offset stats in compare.py will show it.
* Generation that hits --max-new-tokens is flagged ``stats.truncated`` and the
  raw model output is stored under ``raw`` for inspection.
* --chunk-seconds N (default: auto) cuts the audio into fixed windows and adds
  the window start to every time; a word straddling a cut may be split or
  lost. Left at its default, a file estimated to exceed
  ``AUTO_CHUNK_DURATION_S`` (140 s, see next paragraph) is chunked at
  ``AUTO_CHUNK_SECONDS`` (120 s) automatically; passing --chunk-seconds
  explicitly (including ``0`` for "always whole file") disables the auto
  choice for the whole run. ``extra.stats.auto_chunked`` and
  ``extra.chunk_seconds`` record what was actually used per file.
* text_config.max_position_embeddings is 4096 (config.json, rope_type
  "default"). Timestamp mode costs ~7 output tokens per word (a ``[T:N]`` tag is
  5 tokens) plus one audio embedding per 0.1 s (2319 for the 3.5-minute file),
  so a conversation over ~140 s exceeds 4096 positions in one generate() call
  (the longest needs ~6.8k). generate() does not stop there and the card says
  timestamps work to 3.5 min, but accuracy past 4096 is unverified, which is
  why the runner auto-chunks past that length instead of silently degrading
  (see --chunk-seconds above). Every transcript stores ``stats.prompt_tokens``,
  ``stats.total_tokens`` and ``stats.max_ctx_tokens`` (prompt incl. audio
  embeddings + output, largest single call), the per-file line prints
  ``ctx N`` and flags ``CTX>4096`` (still possible on an explicit
  ``--chunk-seconds 0`` or on a chunk window that itself packs enough words).
  Check n_regressions/n_rollovers and the compare.py drift on the long files
  first; if they drift with auto-chunking already on, try a smaller
  --chunk-seconds and a distinct --tag (e.g. ``granite-4.1-2b-plus+chunk60``).
  --max-new-tokens 12000 is ample (expected maximum ~4.4k); hitting it would
  mean a generation loop.
* No ``[T:N]`` tag at all (the card: on a prompt it does not recognise "the
  model simply ignores it and performs transcription", e.g. a ``--prompt``
  without "Timestamps:") is written as a text-only transcript
  (``word_timestamps: false``, one segment 0..duration, ``words: []``,
  ``stats.no_tags``) with a WARNING, never as words all at 0.0-0.0. Fewer tags
  than half the words sets ``stats.sparse_tags`` with a WARNING.
* One failing file (decode error, CUDA OOM, ...) prints ``FAIL <name>`` with
  the traceback, empties the CUDA cache and the run continues; the run ends
  with ``FAILED n: <names>`` and exit status 1. Rerun without --force to redo
  only the missing files.
"""
from __future__ import annotations

import difflib
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SR, Timer, audio_files, load_audio, out_path,  # noqa: E402
                    standard_args, words_to_segments, write_transcript)

PLUS_ID = 'ibm-granite/granite-speech-4.1-2b-plus'
BASE_ID = 'ibm-granite/granite-speech-4.1-2b'
# text_config.max_position_embeddings in both config.json files (see Known caveats).
MAX_POS = 4096
# Known caveats: a whole-file call exceeds MAX_POS above ~140 s of audio (the
# longest training conversation, ~3.5 min, needs ~6.8k tokens). Past this
# length, auto-chunk at AUTO_CHUNK_SECONDS instead of silently generating past
# the model's trained context. --chunk-seconds set explicitly (any value,
# including 0) overrides this for the whole run.
AUTO_CHUNK_DURATION_S = 140.0
AUTO_CHUNK_SECONDS = 120.0

# Prompts quoted verbatim from the model cards.
# https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
TS_PROMPT = ('<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp '
             'tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]')
PLUS_ASR_PROMPT = '<|audio|> can you transcribe the speech into a written format?'
# https://huggingface.co/ibm-granite/granite-speech-4.1-2b
BASE_ASR_PROMPT = '<|audio|>transcribe the speech with proper punctuation and capitalization.'
# The plus card's example puts this fixed system message first; the base card's
# example uses no system message. Follow each card.
# https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
PLUS_SYSTEM_PROMPT = ('Knowledge Cutoff Date: April 2024.\nToday\'s Date: December 19, 2024.\n'
                      'You are Granite, developed by IBM. You are a helpful AI assistant')

TAG = re.compile(r'\[T:(\d+)\]')          # card: "[T:N]", N in centiseconds mod 1000
SILENCE = re.compile(r'^_+$')             # card: "Silences are transcribed as _"
_TRAILING_PUNCT = re.compile(r"[.!?,;:]+$")


# --------------------------------------------------------------------------- #
# Parsing (pure Python, testable without torch)
# --------------------------------------------------------------------------- #

def parse_timestamps(text: str, t_offset: float = 0.0, min_jump: int = 100,
                     stats: Optional[Counter] = None) -> Tuple[List[dict], List[dict]]:
    """Turn the model's timestamp-mode output into (words, silences).

    words:    [{'w': ' hello', 'start': s, 'end': s}, ...]   (leading space, model spelling)
    silences: [{'start': s, 'end': s}, ...]                   (the ``_`` tokens)
    All times in seconds, absolute (t_offset added). See the module docstring
    for the start-derivation and rollover rules.
    """
    stats = stats if stats is not None else Counter()
    parts = TAG.split(text)        # card parses with re.split(r"\[T:(\d+)\]", ts_text)
    words: List[dict] = []
    silences: List[dict] = []
    last_abs = 0                   # centiseconds within this chunk, after rollover
    rollover = 0
    prev_end = float(t_offset)     # end of the previous token in seconds

    def emit(tok: str, start: float, end: float, raw_cs: Optional[int] = None) -> None:
        item = {'start': start, 'end': end}
        if raw_cs is not None:
            item['raw_cs'] = raw_cs            # the model's tag when end had to be clamped
        if SILENCE.match(tok):
            silences.append(item)
        else:
            words.append({'w': ' ' + tok, **item})

    for i in range(0, len(parts) - 1, 2):
        toks = parts[i].split()
        n = int(parts[i + 1])
        stats['n_tags'] += 1
        cand = n + rollover
        if cand < last_abs and last_abs - cand >= min_jump:
            while cand < last_abs and last_abs - cand >= min_jump:
                rollover += 1000
                cand += 1000
            stats['n_rollovers'] += 1
        if not toks:
            stats['n_empty_tags'] += 1           # two tags with nothing between
            continue
        end = t_offset + cand / 100.0
        raw_cs = None
        if cand < last_abs:
            # Backward tag under min_jump (or the remainder after a rollover): keep
            # the model's value in raw_cs but clamp the end to the previous end so
            # no word or segment has end < start (compare.py would clamp it, but
            # span_ceiling.py and segment_words below would not).
            stats['n_regressions'] += 1
            raw_cs = n
            end = prev_end
        last_abs = max(last_abs, cand)
        k = len(toks)
        stats['n_untagged_words'] += k - 1
        cur = prev_end
        for j, tok in enumerate(toks):
            tok_end = end if j == k - 1 else prev_end + (end - prev_end) * (j + 1) / k
            emit(tok, cur, tok_end, raw_cs if j == k - 1 else None)
            cur = tok_end
        prev_end = end

    for tok in parts[-1].split():                # text after the final tag
        stats['n_trailing_untagged'] += 1
        emit(tok, prev_end, prev_end)
    stats['n_words'] += len(words)
    stats['n_silences'] += len(silences)
    return words, silences


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", '', tok.lower().replace('’', "'"))


def transfer_punctuation(words: List[dict], punct_text: str, stats: Counter) -> List[dict]:
    """Copy the base model's spelling, casing and punctuation onto the plus
    model's timestamped words, token by token where the two transcripts agree.
    Unequal runs keep the timestamped words; only terminal punctuation at the
    end of the base run is carried over so sentence boundaries survive."""
    ptoks = punct_text.split()
    a = [_norm(w['w']) for w in words]
    b = [_norm(t) for t in ptoks]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = [dict(w) for w in words]
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal' or (tag == 'replace' and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                out[i1 + k]['w'] = ' ' + ptoks[j1 + k]
            matched += i2 - i1 if tag == 'equal' else 0
        elif tag in ('replace', 'insert'):
            m = _TRAILING_PUNCT.search(ptoks[j2 - 1])
            target = i2 - 1 if tag == 'replace' else i1 - 1
            if m and target >= 0:
                out[target]['w'] = _TRAILING_PUNCT.sub('', out[target]['w']) + m.group(0)
        # 'delete': plus words absent from the base transcript stay as they are
    stats['punct_tokens'] = len(ptoks)
    stats['punct_matched'] = matched
    stats['punct_ratio'] = round(sm.ratio(), 4)
    return out


def segment_words(words: List[dict], split_pause: float, by_punct: bool) -> List[dict]:
    """Cut at gaps >= split_pause (a gap only exists where the model emitted a
    ``_`` silence), then, if by_punct, at terminal punctuation inside each run."""
    if not words:
        return []
    runs, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if split_pause > 0 and w['start'] - prev['end'] >= split_pause:
            runs.append(cur); cur = []
        cur.append(w)
    runs.append(cur)
    segments = []
    for run in runs:
        if by_punct:
            segments.extend(words_to_segments(run))
        else:
            segments.append({'start': run[0]['start'], 'end': run[-1]['end'],
                             'text': ''.join(w['w'] for w in run), 'words': list(run)})
    return segments


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class Granite:
    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        # https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus (usage example)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        self.device = device
        self.is_plus = model_id.endswith('-plus')
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        # card: AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, device_map=device, dtype=torch.bfloat16)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, device_map=device, dtype=getattr(torch, dtype))
        self.model.eval()
        print(f'loaded {model_id} in {time.time() - t0:.1f}s', flush=True)

    def generate(self, audio: np.ndarray, prompt: str, max_new_tokens: int) -> Tuple[str, int, int, bool]:
        """Returns (text, n_generated_tokens, n_prompt_tokens, truncated).
        n_prompt_tokens counts the expanded <|audio|> placeholders, i.e. it is the
        real context length the decoder sees before the first generated token."""
        import torch
        chat = [{'role': 'user', 'content': prompt}]
        if self.is_plus:
            chat.insert(0, {'role': 'system', 'content': PLUS_SYSTEM_PROMPT})
        # card: tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # Feature extractor accepts a float tensor; the cards pass torchaudio's (1, T).
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/feature_extraction_granite_speech.py
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            # card: processor(prompt_text, audio, device=device, return_tensors="pt").to(device)
            inputs = self.processor(prompt_text, wav, device=self.device, return_tensors='pt').to(self.device)
            # card: model.generate(**inputs, max_new_tokens=..., do_sample=False, num_beams=1)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, num_beams=1)
        n_in = inputs['input_ids'].shape[-1]
        new_tokens = outputs[0, n_in:]
        n_new = int(new_tokens.shape[-1])
        # card: tokenizer.decode(new_tokens, skip_special_tokens=True); the [T:N]
        # tags survive this in the card's own example, so they are not special tokens.
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), n_new, int(n_in), n_new >= max_new_tokens


def _count_tokens(stats: Counter, n_in: int, n_new: int, truncated: bool) -> None:
    """Per generate() call: prompt (incl. audio embeddings), output, and the
    largest single-call context seen, to compare with MAX_POS."""
    stats['prompt_tokens'] += n_in
    stats['gen_tokens'] += n_new
    stats['total_tokens'] += n_in + n_new
    stats['max_ctx_tokens'] = max(stats['max_ctx_tokens'], n_in + n_new)
    stats['truncated'] += int(truncated)


def _free_cuda(device: str) -> None:
    """After a failed file (e.g. OOM): release cached blocks so the next file gets them."""
    if not device.startswith('cuda'):
        return
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _extra(ap):
    ap.add_argument('--model', default=None, help='HF id; default depends on mode')
    ap.add_argument('--text-only', action='store_true',
                    help='punctuated text, no word timestamps (base model by default)')
    ap.add_argument('--punctuate', action='store_true',
                    help='also run the base model and transfer its punctuation onto the plus words')
    ap.add_argument('--prompt', default=None, help='override the instruction (must contain <|audio|>)')
    ap.add_argument('--max-new-tokens', type=int, default=12000,
                    help='card uses 10000 for a 1-minute timestamp example')
    ap.add_argument('--split-pause', type=float, default=0.6,
                    help='segment break at model-emitted silences >= this many seconds (0 = off)')
    ap.add_argument('--rollover-min-jump', type=int, default=100,
                    help='centiseconds a tag must jump backwards to count as a 10 s rollover; 1 = card rule')
    ap.add_argument('--chunk-seconds', type=float, default=None,
                    help='cut audio into fixed windows for timestamp mode (0 = whole file); '
                         f'default is auto: whole file unless duration > {AUTO_CHUNK_DURATION_S:g}s, '
                         f'then {AUTO_CHUNK_SECONDS:g}s windows (pass a value, including 0, to force it)')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--selftest', action='store_true', help='run the parser on a synthetic string and exit')


def default_tag(model_id: str, text_only: bool, punctuate: bool) -> str:
    short = 'granite-' + model_id.split('granite-speech-', 1)[-1]   # granite-4.1-2b[-plus]
    if text_only and short.endswith('-plus'):
        return short + '+text'
    if punctuate:
        return short + '+punct'
    return short


def selftest() -> int:
    stats = Counter()
    s = 'hello [T:45] world [T:82] _ [T:150] how are [T:190] you [T:995] fine [T:20] _ [T:70] ok [T:60] tail'
    words, sil = parse_timestamps(s, min_jump=100, stats=stats)
    got = [(w['w'], round(w['start'], 2), round(w['end'], 2)) for w in words]
    exp = [(' hello', 0.0, 0.45), (' world', 0.45, 0.82), (' how', 1.5, 1.7), (' are', 1.7, 1.9),
           (' you', 1.9, 9.95), (' fine', 9.95, 10.2), (' ok', 10.7, 10.7), (' tail', 10.7, 10.7)]
    assert got == exp, got
    # 'ok [T:60]' after '_ [T:70]' is a 0.1 s regression: end clamped to 10.7, tag kept
    assert words[6].get('raw_cs') == 60 and 'raw_cs' not in words[5], words[5:7]
    assert all(w['end'] >= w['start'] for w in words), words
    assert [(x['start'], x['end']) for x in sil] == [(0.82, 1.5), (10.2, 10.7)], sil
    assert stats['n_rollovers'] == 1 and stats['n_regressions'] == 1 and stats['n_untagged_words'] == 1 \
        and stats['n_trailing_untagged'] == 1, dict(stats)
    p = transfer_punctuation(words, 'Hello, world. How are you? Fine. OK tail.', Counter())
    assert ''.join(w['w'] for w in p) == ' Hello, world. How are you? Fine. OK tail.', p
    segs = segment_words(p, split_pause=0.6, by_punct=True)
    assert [x['text'] for x in segs] == [' Hello, world.', ' How are you?', ' Fine.', ' OK tail.'], segs
    segs = segment_words(words, split_pause=0.6, by_punct=False)   # 0.68 s gap splits, 0.5 s gap does not
    assert [x['text'] for x in segs] == [' hello world', ' how are you fine ok tail'], segs
    segs = segment_words(words, split_pause=0.4, by_punct=False)
    assert [x['text'] for x in segs] == [' hello world', ' how are you fine', ' ok tail'], segs
    # chunk offset
    w2, _ = parse_timestamps('a [T:100]', t_offset=30.0)
    assert (w2[0]['start'], w2[0]['end']) == (30.0, 31.0)
    # plain transcription (no tags): every word 0.0-0.0 and n_tags 0; main() writes text-only
    st = Counter()
    w3, _ = parse_timestamps('hello world there', stats=st)
    assert st['n_tags'] == 0 and [(w['start'], w['end']) for w in w3] == [(0.0, 0.0)] * 3, (dict(st), w3)
    print('selftest ok', dict(stats))
    return 0


def main() -> int:
    args = standard_args('granite-4.1-2b-plus', _extra)
    if args.selftest:
        return selftest()
    if args.text_only and args.punctuate:
        raise SystemExit('--text-only and --punctuate are exclusive')
    model_id = args.model or (BASE_ID if args.text_only else PLUS_ID)
    if args.tag == 'granite-4.1-2b-plus':
        args.tag = default_tag(model_id, args.text_only, args.punctuate)
    is_plus = model_id.endswith('-plus')
    if args.text_only:
        prompt = args.prompt or (PLUS_ASR_PROMPT if is_plus else BASE_ASR_PROMPT)
    else:
        if not is_plus:
            raise SystemExit('timestamp mode needs the -plus model (use --text-only for the base model)')
        prompt = args.prompt or TS_PROMPT

    files = audio_files(args)
    if files:
        # Pre-flight: decode one file before loading ~5-10 GB of weights, so a
        # missing resampler (librosa) fails in seconds, not after the model load.
        _, sr = load_audio(files[0])
        if sr != SR:
            raise SystemExit(f'load_audio returned {sr} Hz; the Granite feature extractor assumes {SR}')

    main_model = Granite(model_id, args.device, args.dtype)
    punct_model = Granite(BASE_ID, args.device, args.dtype) if args.punctuate else None
    print(f'tag {args.tag}  prompt {prompt!r}', flush=True)

    total_audio = total_dt = 0.0
    n_done = 0
    failed: List[str] = []
    for path in files:
        out = out_path(args, path)
        if out.exists() and not args.force:
            print(f'skip {path.name}', flush=True); continue
        try:
            audio, sr = load_audio(path)
            assert sr == SR, f'load_audio returned {sr} Hz, the Granite feature extractor assumes {SR}'
            duration = len(audio) / sr
            stats = Counter()
            extra = {'hf_model': model_id, 'prompt': prompt}
            word_timestamps = not args.text_only

            if args.text_only:
                with Timer() as t:
                    text, n_new, n_in, trunc = main_model.generate(audio, prompt, args.max_new_tokens)
                segments = [{'start': 0.0, 'end': duration, 'text': text, 'words': []}]
                _count_tokens(stats, n_in, n_new, trunc)
                extra.update({'raw': text, 'stats': dict(stats)})
            else:
                raws: List[str] = []
                if args.chunk_seconds is None:
                    auto_chunked = duration > AUTO_CHUNK_DURATION_S
                    chunk_seconds = AUTO_CHUNK_SECONDS if auto_chunked else 0.0
                else:
                    auto_chunked = False
                    chunk_seconds = args.chunk_seconds
                with Timer() as t:
                    words: List[dict] = []
                    silences: List[dict] = []
                    step = int(chunk_seconds * sr) if chunk_seconds > 0 else len(audio)
                    for off in range(0, len(audio), max(step, 1)):
                        piece = audio[off:off + step]
                        text, n_new, n_in, trunc = main_model.generate(piece, prompt, args.max_new_tokens)
                        raws.append(text)
                        _count_tokens(stats, n_in, n_new, trunc)
                        w, s = parse_timestamps(text, t_offset=off / sr, min_jump=args.rollover_min_jump,
                                                stats=stats)
                        words.extend(w); silences.extend(s)
                    no_tags = stats['n_tags'] == 0
                    if punct_model is not None and not no_tags:
                        ptext, n_new, n_in, trunc = punct_model.generate(audio, BASE_ASR_PROMPT,
                                                                         args.max_new_tokens)
                        stats['punct_gen_tokens'] = n_new
                        stats['punct_prompt_tokens'] = n_in
                        stats['punct_truncated'] = int(trunc)
                        extra['raw_punct'] = ptext
                        words = transfer_punctuation(words, ptext, stats)
                if no_tags:
                    # Card: on a prompt it does not recognise "the model simply ignores
                    # it and performs transcription". Without tags every word would sit
                    # at 0.0-0.0 and look like valid timing downstream; write text-only.
                    stats['no_tags'] = 1
                    word_timestamps = False
                    print(f'WARNING {path.name}: no [T:N] tags in {stats["n_words"]} words; '
                          f'writing a text-only transcript (word_timestamps false)',
                          file=sys.stderr, flush=True)
                    segments = [{'start': 0.0, 'end': duration, 'text': ' '.join(raws).strip(),
                                 'words': []}]
                else:
                    if stats['n_tags'] < 0.5 * stats['n_words']:
                        stats['sparse_tags'] = 1
                        print(f'WARNING {path.name}: only {stats["n_tags"]} tags for '
                              f'{stats["n_words"]} words; most times are interpolated',
                              file=sys.stderr, flush=True)
                    segments = segment_words(words, args.split_pause, by_punct=punct_model is not None)
                stats['auto_chunked'] = int(auto_chunked)
                extra.update({'raw': '\n'.join(raws), 'silences': silences, 'stats': dict(stats),
                              'chunk_seconds': chunk_seconds, 'split_pause': args.split_pause,
                              'rollover_min_jump': args.rollover_min_jump})
            write_transcript(out, path, args.tag, duration, t.seconds, segments,
                             word_timestamps=word_timestamps, extra=extra)
        except Exception as e:  # noqa: BLE001  one bad file must not abort the tag run
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            _free_cuda(args.device)
            continue

        total_audio += duration; total_dt += t.seconds; n_done += 1
        flags = ' TRUNCATED' if stats.get('truncated') else ''
        if stats.get('auto_chunked'):
            flags += f' AUTO_CHUNK={AUTO_CHUNK_SECONDS:g}s'
        if stats.get('max_ctx_tokens', 0) > MAX_POS:
            flags += f' CTX>{MAX_POS}'
        if stats.get('no_tags'):
            flags += ' NO_TAGS'
        elif stats.get('sparse_tags'):
            flags += ' SPARSE_TAGS'
        print(f'{path.name}: {duration:6.1f}s audio, {t.seconds:6.1f}s, {len(segments)} segs, '
              f'{stats.get("n_words", 0)} words, {stats.get("n_silences", 0)} sil, '
              f'{stats.get("gen_tokens", 0)} tok, ctx {stats.get("max_ctx_tokens", 0)}, '
              f'roll {stats.get("n_rollovers", 0)}, regr {stats.get("n_regressions", 0)}{flags}',
              flush=True)
    if n_done and total_audio:
        print(f'DONE {n_done} files, RTF {total_dt / total_audio:.3f}, '
              f'mean {total_dt / n_done:.1f}s per conversation', flush=True)
    if failed:
        print(f'FAILED {len(failed)}: {" ".join(failed)}', file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
