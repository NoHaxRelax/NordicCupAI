"""HF transformers Whisper runner: word timestamps from the cross-attention
alignment heads (DTW), optional speculative decoding with a distil-whisper style
assistant, sequential long-form generation. Writes the transcript schema from
bench/README.md (one JSON per conversation per tag under transcripts/).

Tags (derived unless --tag is given):
    whisper-large-v3                          openai/whisper-large-v3, greedy, word timestamps
    whisper-large-v3+spec-distil              + --assistant distil-whisper/distil-large-v3
    whisper-large-v3+spec-distil-large-v3.5   + --assistant distil-whisper/distil-large-v3.5
    whisper-large-v3+spec-large-v3-turbo      + --assistant openai/whisper-large-v3-turbo
    <tag>+beam5                               --beam 5 (never together with --assistant)
    <tag>+segts                               --timestamps segment: segment times only,
                                              words: [], top-level word_timestamps=false;
                                              also where a file lands when assistant + word
                                              timestamps fails and the segment retry succeeds

Install (Linux venv, CUDA 12.x, H100; the "hf" dependency family in bench/hpc/env.sh):
    python3.11 -m venv "$VENV/hf" && . "$VENV/hf/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install "transformers>=5.17" accelerate numpy soundfile librosa
    export HF_HOME=/dtu/blackhole/1e/205502/hf      # weights land on blackhole, not $HOME

transformers 5.2.0 is verified BROKEN for --assistant (measured 2026-09-17 by
running this runner on CPU with openai/whisper-tiny on conversation_sample_4.mp3):
a decoder-only assistant dies on every file with `UnboundLocalError: is_updated`
in WhisperAttention.forward (modeling_whisper.py:325; 5.17.0 adds `is_updated =
False` at :313), and a seq2seq assistant cannot emit EOS (regression fixed by
PR #48108, worked around in 5.17.0's generate_with_fallback), which made the
segment-mode retry write 6 fragments / 100 characters for a 106 s file. Plain
runs (no assistant) are fine on 5.2.0. load_models() refuses --assistant below
5.17.0 (MIN_TRANSFORMERS_ASSISTED). bench/hpc/env.sh installs "transformers>=5",
which resolved to 5.17.0 on 2026-09-17, so the LSF hf-whisper-spec step is on
the working version.

Run (from medical-appointment/):
    python bench/asr/run_hf_whisper.py                                             # whisper-large-v3
    python bench/asr/run_hf_whisper.py --assistant distil-whisper/distil-large-v3   # +spec-distil
    python bench/asr/run_hf_whisper.py --assistant distil-whisper/distil-large-v3 --timestamps segment
    python bench/asr/run_hf_whisper.py --model openai/whisper-tiny --device cpu --limit 1   # CPU smoke test

Timing: `seconds` in the transcript is feature extraction + generate() for that
file only (CUDA-synchronised), model loading excluded, so the plain tag and the
+spec-* tag can be compared directly. A warm-up generate runs first (--no-warmup
to skip) so the first file does not pay CUDA initialisation.

How it works (line numbers are from the transformers v5.2.0 source installed
locally; every function cited was diffed against the v5.17.0 wheel on
2026-09-17 and is byte-identical unless noted, so the reading applies to both;
URLs next to each call):

  * Long-form: audio longer than 30 s is featurised with truncation=False,
    padding="longest", return_attention_mask=True and generate() runs OpenAI's
    sequential sliding-window algorithm (return_timestamps=True is mandatory
    there). We deliberately do not use the pipeline's chunk_length_s mode: it
    transcribes overlapping chunks independently and stitches them, which is
    faster but is documented as less accurate and its word times come from the
    stitched token stream. Sequential keeps one time origin per 30 s window and
    generate() returns `segments` with absolute start/end and per-token times.
  * Word timestamps: return_token_timestamps=True makes generate() collect
    encoder-decoder cross-attentions from generation_config.alignment_heads and
    run DTW (WhisperGenerationMixin._extract_token_timestamps). v5.2.0 forces
    eager attention itself when this flag is set (generation_whisper.py:704-707),
    so sdpa/flash-attn cannot be used for the timestamped run; that is a known
    slowdown, not something to fix here. Per segment, token_timestamps[j] is
    the END of token j (= start of token j+1); a word spanning token indices
    a..b therefore starts at token_timestamps[a-1] and ends at
    token_timestamps[b], exactly what transformers' own _decode_asr does
    (tokenization_whisper.py:1089-1095) and what faster-whisper/openai-whisper
    report (jump time at the word's first boundary to jump time at the next),
    so the offsets measured by span_ceiling.py are comparable across tags.
    Token to word grouping mirrors _split_tokens_on_spaces/_merge_punctuations
    (new word at a leading space or a punctuation token; opening punctuation
    attaches to the next word, closing to the previous). Times are rounded to
    the 0.02 s grid the model already uses, nothing else is adjusted.
  * Speculative decoding: --assistant loads a WhisperForCausalLM (decoder only)
    that reuses the main model's encoder outputs; that is the documented setup
    for distil-whisper/distil-large-v3 and distil-large-v3.5 (both keep the
    large-v3 encoder frozen) and it applies to openai/whisper-large-v3-turbo
    too (its card: "the exact same model, except that the number of decoding
    layers have reduced from 32 to 4"). --assistant-arch seq2seq (a full
    WhisperForConditionalGeneration with its own encoder, e.g. whisper-tiny, as
    in transformers' test_speculative_decoding_non_distil) is REFUSED by main():
    measured broken in both versions, 5.2.0 raises TypeError 'NoneType' object
    is not subscriptable in generation_whisper.py:1165 (split_by_batch_index
    over the assistant's None cross_attentions) and 5.17.0 raises IndexError
    index 448 is out of bounds in the assistant's embed_positions. The choice
    stays in argparse for when upstream fixes it. Assisted generation is
    batch_size = 1 and greedy/sample only (generation/utils.py:3649 in 5.2.0;
    configuration_utils.py get_generation_mode), so this runner always
    transcribes one file at a time and refuses --beam > 1 with an assistant.

Word timestamps together with assistant_model (the CRITICAL question), MEASURED
2026-09-17, not just code-read: openai/whisper-tiny as the main model and the
same checkpoint as a decoder-only assistant (WhisperForCausalLM), CPU,
conversation_sample_4.mp3 (106 s), transformers 5.17.0:
  * plain word mode and --assistant word mode both run and give 257 words with
    identical text and identical times (35 segments, 1384 characters);
    --assistant --timestamps segment gives the same 35 segments. On 5.2.0 the
    same runs crash (see Install above).
  * Mechanism: _assisted_decoding stores cross-attentions one tuple entry per
    accepted token via _split_model_outputs (utils.py:3780 in 5.2.0, :3991 in
    5.17.0; the function is byte-identical) and _extract_token_timestamps
    concatenates the entries along the token axis, so the tensor shapes match
    the greedy path. Not covered by transformers' tests (test_modeling_whisper.py
    tests assisted decoding without timestamps).
  * One real bug in _split_model_outputs (both versions, confirmed by running
    it on row-labelled tensors): on the FIRST assisted iteration of each 30 s
    window the model output still holds the P prompt rows, and after emitting
    the prompt block [0:P] the loop slices rows [i:i+1] from row 0, i.e. the
    prompt's rows again, instead of the candidates' rows [P+i:P+i+1]. The entry
    count is right, so nothing raises; the first n_matches accepted tokens of
    every window get someone else's cross-attention rows in the DTW, and the
    DTW output is monotonic anyway, so words_non_monotonic cannot catch it. It
    did not fire in the measurement above because the assistant's first
    candidate in every window was a timestamp token the main model rejected
    (n_matches = 0 in 4 of 4 windows), which is why plain and +spec times were
    identical; with distil-large-v3 as assistant a first-token match is likely
    and the corruption would be silent. patch_split_model_outputs() therefore
    self-tests the installed function on labelled rows and, if it shows this
    bug, replaces it with the corrected copy _split_model_outputs_fixed before
    any generate() with an assistant (the name is a module global looked up
    inside _assisted_decoding, so the patch takes effect); the outcome is
    recorded under runner.split_model_outputs in every transcript. Still to do
    on the cluster: check that +spec-distil word times equal the
    whisper-large-v3 times wherever the words match (greedy decoding is
    deterministic apart from begin_suppress_tokens), and file the upstream
    issue.
  * If generate() raises with assistant + word timestamps the file is retried
    with segment timestamps only and the result is written under <tag>+segts
    (words: [], word_timestamps=false, "timestamps": "segment" and
    "fallback_from_word": true under the "runner" key), NEVER under the word
    tag: the word-tag file stays absent, is retried on the next run, and
    compare.py's rows stay honest. --timestamps segment forces segment mode for
    every file (tag suffix +segts). The speed comparison is valid in either mode.
  * Plausibility guard: a transcript with fewer than MIN_CHARS_PER_SEC characters
    per second of audio, or in word mode fewer than MIN_WORDS_PER_SEC words per
    second, is not written and counts as FAILED (this corpus runs about 13
    characters/s and 2.4 words/s; the 5.2.0 seq2seq fallback produced 0.9
    characters/s and was previously written as a normal DONE file, which
    skip-if-exists then cemented).
  * Post-processing (to_segments, write_transcript) is guarded like generate():
    one odd file is reported as FAILED and the tag continues; a write that
    fails midway is unlinked so skip-if-exists cannot keep a truncated JSON.

Other caveats:
  * `p` (word probability) is not written; the schema marks it optional.
  * --fallback enables OpenAI's temperature fallback (0.0..1.0, compression
    ratio 1.35, logprob -1.0, no-speech 0.6). Off by default: this audio is
    clean, and fallback+assistant is UNVERIFIED (assisted sampling exists but
    Whisper's fallback loop with an assistant is untested here).
  * openai/whisper-large-v3-turbo as the MAIN model has documented broken word
    timestamps (transformers issue #37248); as an assistant it is fine, since
    timestamps come from the main model's cross-attention.
  * Runs one file per generate() call (batch 1). Fine for 39 files on an H100.

Sources verified 2026-09-17:
  https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/whisper
  https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/whisper/generation_whisper.py
  https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/whisper/tokenization_whisper.py
  https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/generation/utils.py
  https://github.com/huggingface/transformers/blob/v5.2.0/tests/models/whisper/test_modeling_whisper.py
  https://huggingface.co/openai/whisper-large-v3 (+ raw/main/generation_config.json: alignment_heads)
  https://huggingface.co/distil-whisper/distil-large-v3
  https://huggingface.co/distil-whisper/distil-large-v3.5
  https://huggingface.co/openai/whisper-large-v3-turbo
  https://huggingface.co/blog/whisper-speculative-decoding
  https://github.com/huggingface/transformers/issues/28977 (long-form + token timestamps, fixed by PR #29148)
  https://github.com/huggingface/transformers/pull/48108 (Whisper speculative decoding: the assistant could not
      emit EOS after PR #42702; merged 2026-08-20; the workaround is in 5.17.0's generation_whisper.py:1036)
  transformers 5.17.0 wheel, diffed against 5.2.0 on 2026-09-17: modeling_whisper.py:313 adds `is_updated = False`;
      generation/utils.py:4227 _split_model_outputs, generation_whisper.py _retrieve_segment /
      _extract_token_timestamps / generate() kwargs, tokenization_whisper.py _decode_asr, the feature extractor
      __call__ / n_samples and from_pretrained(dtype=, use_safetensors=) are identical.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, audio_files, load_audio, out_path, standard_args, write_transcript  # noqa: E402

# Windows laptop only (torch + MKL each ship an OpenMP runtime); harmless on Linux.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

DEFAULT_MODEL = 'openai/whisper-large-v3'

# Same character sets as transformers' _combine_tokens_into_words defaults and
# _split_tokens_on_spaces (tokenization_whisper.py:1289-1368, v5.2.0).
PREPEND_PUNCT = "\"'“¡¿([{-"
APPEND_PUNCT = "\"'.。,，!！?？:：”)]}、"
SPLIT_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# OpenAI's fallback settings, as in the transformers Whisper docs long-form example
# https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/whisper
FALLBACK_KWARGS = dict(
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=1.35,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
)

# transformers < 5.17.0 lacks the `is_updated = False` init in WhisperAttention.forward
# (modeling_whisper.py:313 in 5.17.0) and --assistant crashes on every file with
# UnboundLocalError: is_updated (see module docstring, "Install").
MIN_TRANSFORMERS_ASSISTED = (5, 17, 0)

# Plausibility guard (module docstring): this corpus runs about 13 characters/s and
# 2.4 words/s; the 5.2.0 seq2seq-assistant fallback produced 0.9 characters/s and was
# previously written as a normal DONE file, which skip-if-exists then cemented.
MIN_CHARS_PER_SEC = 3.0
MIN_WORDS_PER_SEC = 0.5


def _version_tuple(v: str) -> Tuple[int, int, int]:
    """'5.17.0' / '5.17.0.dev0' -> (5, 17, 0). No `packaging` dependency needed."""
    parts = []
    for p in v.split('.')[:3]:
        digits = ''.join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_args(ap) -> None:
    ap.add_argument('--model', default=DEFAULT_MODEL, help='HF id of the main Whisper checkpoint')
    ap.add_argument('--assistant', default=None,
                    help='HF id of the assistant for speculative decoding, e.g. distil-whisper/distil-large-v3')
    ap.add_argument('--assistant-arch', choices=['decoder', 'seq2seq'], default='decoder',
                    help='decoder: WhisperForCausalLM sharing the main encoder (distil-*, large-v3-turbo); '
                         'seq2seq: full model with its own encoder (e.g. openai/whisper-tiny)')
    ap.add_argument('--timestamps', choices=['word', 'segment'], default='word',
                    help='word: cross-attention DTW word times; segment: timestamp tokens only (words: [])')
    ap.add_argument('--beam', type=int, default=1, help='num_beams (must be 1 with --assistant)')
    ap.add_argument('--fallback', action='store_true',
                    help='OpenAI temperature fallback + compression/logprob/no-speech thresholds')
    ap.add_argument('--condition-on-prev', action='store_true',
                    help='condition each 30 s window on the previous window text (off: HF default)')
    ap.add_argument('--dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto',
                    help='auto = float16 on cuda, float32 on cpu')
    ap.add_argument('--no-warmup', action='store_true', help='skip the untimed warm-up generate()')


def derive_tag(args) -> str:
    """README naming: HF name without the org, plus a suffix per variant."""
    tag = args.model.split('/')[-1].lower()
    if args.assistant:
        short = args.assistant.split('/')[-1].lower()
        if short.startswith('whisper-'):
            short = short[len('whisper-'):]
        if short == 'distil-large-v3':
            short = 'distil'                     # README example: large-v3+spec-distil
        tag += f'+spec-{short}'
    if args.beam > 1:
        tag += f'+beam{args.beam}'
    if args.timestamps == 'segment':
        tag += '+segts'
    return tag


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def load_models(args):
    import torch
    import transformers
    # Class names and from_pretrained(dtype=..., use_safetensors=...) as used in transformers v5.2.0's own
    # tests/models/whisper/test_modeling_whisper.py::test_speculative_decoding_distil / _non_distil.
    # `dtype` replaces the deprecated `torch_dtype` in v5 (modeling_utils.py:1444-1446).
    from transformers import WhisperForCausalLM, WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device(args.device)
    if args.dtype == 'auto':
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    t0 = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype, use_safetensors=True)
    model.to(device).eval()
    processor = WhisperProcessor.from_pretrained(args.model)
    if args.timestamps == 'word' and getattr(model.generation_config, 'alignment_heads', None) is None:
        # generate() raises the same way (generation_whisper.py:_set_num_frames); fail before loading anything else
        raise SystemExit(f'{args.model}: generation_config has no alignment_heads, word timestamps impossible; '
                         f'use --timestamps segment or another checkpoint')
    print(f'loaded {args.model} ({dtype}) on {device} in {time.time() - t0:.1f}s', flush=True)

    assistant = None
    if args.assistant:
        if _version_tuple(transformers.__version__) < MIN_TRANSFORMERS_ASSISTED:
            min_str = '.'.join(str(x) for x in MIN_TRANSFORMERS_ASSISTED)
            raise SystemExit(f'--assistant needs transformers>={min_str} (is_updated bug in '
                             f'WhisperAttention.forward below that version), have {transformers.__version__}')
        t0 = time.time()
        if args.assistant_arch == 'decoder':
            # Decoder-only assistant reusing the main encoder's outputs
            # (https://huggingface.co/distil-whisper/distil-large-v3 "Speculative Decoding";
            #  generation/candidate_generator.py:151-152 passes encoder_outputs to the assistant).
            assistant = WhisperForCausalLM.from_pretrained(args.assistant, dtype=dtype, use_safetensors=True)
        else:
            assistant = WhisperForConditionalGeneration.from_pretrained(args.assistant, dtype=dtype,
                                                                        use_safetensors=True)
        assistant.to(device).eval()
        print(f'loaded assistant {args.assistant} ({args.assistant_arch}) in {time.time() - t0:.1f}s', flush=True)

    print(f'transformers {transformers.__version__}, torch {torch.__version__}', flush=True)
    return model, processor, assistant, device, dtype


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

def transcribe(model, processor, assistant, device, dtype, audio: np.ndarray, args, want_words: bool):
    """One generate() call for one file. Returns the dict generate() gives when
    return_segments=True: {'sequences', ['token_timestamps'], 'segments'}."""
    import torch
    fe = processor.feature_extractor
    if len(audio) > fe.n_samples:
        # Long-form (> 30 s): the documented sequential recipe
        # https://huggingface.co/docs/transformers/v5.2.0/en/model_doc/whisper ("Long-form transcription")
        feats = fe(audio, sampling_rate=SR, return_tensors='pt', truncation=False, padding='longest',
                   return_attention_mask=True)
    else:
        # Short-form: the encoder needs exactly 3000 mel frames (default padding='max_length')
        feats = fe(audio, sampling_rate=SR, return_tensors='pt', return_attention_mask=True)

    # WhisperGenerationMixin.generate kwargs, generation_whisper.py:383-412 (v5.2.0)
    kw = dict(
        return_timestamps=True,          # mandatory for long-form; gives segment start/end from timestamp tokens
        return_segments=True,            # returns the per-window segment dicts instead of flat token ids
        language='en',
        task='transcribe',
        num_beams=args.beam,
        condition_on_prev_tokens=args.condition_on_prev,
    )
    if want_words:
        kw['return_token_timestamps'] = True   # DTW over alignment_heads; forces eager attention internally
    if args.fallback:
        kw.update(FALLBACK_KWARGS)
    if assistant is not None:
        kw['assistant_model'] = assistant       # generate() sets force_unique_generate_call on it itself

    input_features = feats.input_features.to(device, dtype)
    attention_mask = feats.attention_mask.to(device)   # needed for num_frames -> DTW crop of the last window
    with torch.inference_mode():
        out = model.generate(input_features=input_features, attention_mask=attention_mask, **kw)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return out


def words_from_tokens(tokenizer, toks: List[int], tts: List[float], seg_start: float, seg_end: float,
                      eos_id: int) -> List[dict]:
    """Group a segment's token ids into words and time them.

    toks[j] / tts[j] are aligned; tts[j] is the end of token j (start of token
    j+1), see the module docstring. Special tokens (id >= eos_token_id: EOS,
    language/task markers, <|notimestamps|>, <|t.tt|>) carry no text and are
    skipped but their times are kept as neighbours."""
    # 1. tokens -> subwords at valid unicode boundaries (_split_tokens_on_unicode)
    subwords: List[Tuple[str, List[int]]] = []
    cur: List[int] = []
    cur_idx: List[int] = []
    for j, t in enumerate(toks):
        if t >= eos_id:
            continue
        cur.append(t)
        cur_idx.append(j)
        s = tokenizer.decode(cur)                 # public API; partial UTF-8 shows up as U+FFFD
        if '�' not in s:
            subwords.append((s, cur_idx))
            cur, cur_idx = [], []
    if cur:
        subwords.append((tokenizer.decode(cur), cur_idx))

    # 2. subwords -> words at a leading space or a punctuation token (_split_tokens_on_spaces)
    words: List[List] = []
    for s, idx in subwords:
        if s.startswith(' ') or s.strip() in SPLIT_PUNCT or not words:
            words.append([s, list(idx)])
        else:
            words[-1][0] += s
            words[-1][1].extend(idx)

    # 3. opening punctuation joins the next word, closing punctuation the previous (_merge_punctuations)
    merged: List[List] = []
    prefix: Optional[Tuple[str, List[int]]] = None
    for s, idx in words:
        if s.startswith(' ') and s.strip() in PREPEND_PUNCT:
            prefix = (s, idx) if prefix is None else (prefix[0] + s, prefix[1] + idx)
            continue
        if prefix is not None:
            s, idx = prefix[0] + s, prefix[1] + idx
            prefix = None
        if merged and s in APPEND_PUNCT and not merged[-1][0].endswith(' '):
            merged[-1][0] += s
            merged[-1][1].extend(idx)
        else:
            merged.append([s, idx])
    if prefix is not None:
        merged.append([prefix[0], prefix[1]])

    # 4. times: start = end of the previous token, end = end of the last token
    out: List[dict] = []
    for s, idx in merged:
        if not s.strip():
            continue
        start = tts[idx[0] - 1] if idx[0] > 0 else seg_start
        end = tts[idx[-1]] if idx[-1] < len(tts) else seg_end
        start = round(float(start), 2)
        end = round(max(float(end), start), 2)
        out.append({'w': s, 'start': start, 'end': end})
    return out


def to_segments(gen_out, tokenizer, eos_id: int, want_words: bool) -> List[dict]:
    """generate() segments (batch item 0) -> README segment dicts.

    Each segment dict from generation_whisper.py:_retrieve_segment has
    'start'/'end' (0-d float tensors, absolute seconds), 'tokens' (1-d tensor,
    prompt stripped, timestamp tokens included) and, with
    return_token_timestamps, 'token_timestamps' (same length, absolute)."""
    segments: List[dict] = []
    for seg in gen_out['segments'][0]:
        toks = [int(t) for t in seg['tokens'].tolist()]
        s0, s1 = float(seg['start']), float(seg['end'])
        if want_words:
            tts = [float(x) for x in seg['token_timestamps'].tolist()]
            if len(tts) < len(toks):                                # never expected; keep alignment sane
                tts = tts + [tts[-1] if tts else s1] * (len(toks) - len(tts))
            words = words_from_tokens(tokenizer, toks, tts, s0, s1, eos_id)
            if not words:
                continue
            segments.append({'start': round(s0, 2), 'end': round(s1, 2),
                             'text': ''.join(w['w'] for w in words), 'words': words})
        else:
            text = tokenizer.decode(toks, skip_special_tokens=True)
            if not text.strip():
                continue
            segments.append({'start': round(s0, 2), 'end': round(s1, 2), 'text': text, 'words': []})
    return segments


def non_monotonic_words(segments: List[dict]) -> int:
    """How many words start before the previous word: a cheap red flag for
    broken DTW rows (e.g. the assisted-decoding concern in the docstring)."""
    n, prev = 0, -1.0
    for s in segments:
        for w in s['words']:
            if w['start'] < prev - 1e-6:
                n += 1
            prev = w['start']
    return n


# --------------------------------------------------------------------------- #
# _split_model_outputs prompt-row bug (module docstring): self-test + patch
# --------------------------------------------------------------------------- #

def _split_model_outputs_fixed(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """Corrected copy of transformers.generation.utils._split_model_outputs
    (byte-identical in 5.2.0 and 5.17.0). On the first assisted step of a
    window (len(outputs) == 0) the function consumes the prompt block
    [0:cur_len] from `new_outputs`, bumps `cur_len` by one and shrinks
    `added_len` accordingly -- but then the shared loop below re-reads
    `new_outputs` from row 0 (`layer[..., i:i+1, :]`) instead of continuing
    from the row the prompt block left off at, so the first n_matches accepted
    tokens of every window silently get the prompt's own cross-attention rows.
    Fixed by tracking that row offset explicitly; the loop is a no-op change
    (`row_offset` is 0) on every call that does not go through the first-step
    branch, i.e. everything upstream already gets right."""
    row_offset = 0
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        cur_len += 1
        added_len -= cur_len
        row_offset = cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            r = row_offset + i
            new_tuple += (layer[..., r:r + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs


def patch_split_model_outputs() -> bool:
    """Self-test the installed transformers.generation.utils._split_model_outputs
    on row-labelled tensors and replace it with _split_model_outputs_fixed if
    it shows the prompt-row bug described above. Must run before the first
    generate() call with an assistant: _assisted_decoding looks up the
    function by its module-global name, so patching that name takes effect.
    Returns True if the bug was present (and the patch was applied), False if
    the installed function was already correct (nothing patched)."""
    import torch
    import transformers.generation.utils as gen_utils

    prompt_len, extra = 3, 3   # rows 0..2 = prompt, 3 = first generated token, 4-5 = 2 more candidates
    rows = torch.arange(prompt_len + extra, dtype=torch.float32).view(1, prompt_len + extra, 1)
    out = gen_utils._split_model_outputs((), (rows,), cur_len=prompt_len, added_len=prompt_len + extra,
                                         is_decoder_attention=False)
    # out[0] = combined prompt + first-token block (rows 0..prompt_len); out[1] should
    # be the next candidate row, i.e. row prompt_len + 1, not row 0 again.
    first_candidate_row = float(out[1][0][..., 0, 0])
    buggy = first_candidate_row != float(prompt_len + 1)
    if buggy:
        gen_utils._split_model_outputs = _split_model_outputs_fixed
    return buggy


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = standard_args('', extra=add_args)
    if args.assistant and args.beam > 1:
        sys.exit('assisted generation is greedy/sample only in transformers 5.2.0: use --beam 1 with --assistant')
    if args.assistant and args.assistant_arch == 'seq2seq':
        sys.exit('--assistant-arch seq2seq is broken in transformers 5.2.0/5.17.0 (no cross_attentions / '
                 'EOS never emitted); use --assistant-arch decoder')
    if not args.tag:
        args.tag = derive_tag(args)
    files = audio_files(args)
    todo = [f for f in files if args.force or not out_path(args, f).exists()]
    print(f'tag {args.tag}: {len(todo)} of {len(files)} files to transcribe', flush=True)
    if not todo:
        return 0

    import torch
    import transformers
    model, processor, assistant, device, dtype = load_models(args)
    tokenizer = processor.tokenizer
    eos_id = int(tokenizer.eos_token_id)
    want_words = args.timestamps == 'word'
    runner_info = {
        'hf_model': args.model, 'assistant': args.assistant,
        'assistant_arch': args.assistant_arch if args.assistant else None,
        'num_beams': args.beam, 'fallback': args.fallback, 'condition_on_prev_tokens': args.condition_on_prev,
        'long_form': 'sequential', 'dtype': str(dtype).replace('torch.', ''), 'device': str(device),
        'transformers': transformers.__version__, 'torch': torch.__version__,
    }

    if not args.no_warmup:
        t0 = time.time()
        try:
            warm = np.zeros(SR * 8, dtype=np.float32)
            warm[SR:SR * 7] = (np.random.default_rng(0).standard_normal(SR * 6) * 0.01).astype(np.float32)
            transcribe(model, processor, assistant, device, dtype, warm, args, want_words)
            print(f'warm-up generate in {time.time() - t0:.1f}s', flush=True)
        except Exception:
            traceback.print_exc()
            print('warm-up failed (continuing; the real files will show the actual error)', flush=True)

    total_audio = total_dt = 0.0
    n_done, n_fallback, failed = 0, 0, []
    for f in files:
        out = out_path(args, f)
        if out.exists() and not args.force:
            print(f'skip {f.name}', flush=True)
            continue
        audio, sr = load_audio(f)
        duration = len(audio) / sr
        mode = args.timestamps
        fallback_from_word = False
        try:
            t0 = time.time()
            gen = transcribe(model, processor, assistant, device, dtype, audio, args, want_words)
            dt = time.time() - t0
        except Exception:
            traceback.print_exc()
            if not (want_words and assistant is not None):
                failed.append(f.name)
                print(f'FAILED {f.name}', flush=True)
                continue
            # The documented fallback: assistant + word timestamps did not run -> segment timestamps only.
            print(f'{f.name}: word timestamps with assistant failed, retrying with segment timestamps', flush=True)
            try:
                t0 = time.time()
                gen = transcribe(model, processor, assistant, device, dtype, audio, args, want_words=False)
                dt = time.time() - t0
            except Exception:
                traceback.print_exc()
                failed.append(f.name)
                print(f'FAILED {f.name}', flush=True)
                continue
            mode, fallback_from_word, n_fallback = 'segment', True, n_fallback + 1

        segments = to_segments(gen, tokenizer, eos_id, want_words=(mode == 'word'))
        n_words = sum(len(s['words']) for s in segments)
        info = dict(runner_info, timestamps=mode, fallback_from_word=fallback_from_word,
                    n_segments=len(segments), n_words=n_words,
                    words_non_monotonic=non_monotonic_words(segments) if mode == 'word' else None)
        write_transcript(out, f, args.tag, duration, dt, segments,
                         word_timestamps=(mode == 'word'), extra={'runner': info})
        total_audio += duration
        total_dt += dt
        n_done += 1
        flag = '' if not info['words_non_monotonic'] else f', {info["words_non_monotonic"]} non-monotonic words'
        print(f'{f.name}: {duration:6.1f}s audio, {dt:5.1f}s, {len(segments)} segs, {n_words} words '
              f'[{mode}]{flag}', flush=True)

    if n_done:
        print(f'DONE {n_done} files, RTF {total_dt / total_audio:.3f}, mean {total_dt / n_done:.1f}s per '
              f'conversation, tag {args.tag}', flush=True)
    if n_fallback:
        print(f'NOTE {n_fallback} files fell back to segment timestamps (word_timestamps=false)', flush=True)
    if failed:
        print(f'FAILED {len(failed)}: {", ".join(failed)}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
