"""Qwen3-ASR-1.7B transcription with Qwen3-ForcedAligner-0.6B word timestamps,
and re-timing of any cached transcript with the aligner alone.

Two modes, one output schema (bench/README.md):

  ASR mode (default, tag `qwen3-asr-1.7b+fa`)
      audio -> Qwen3-ASR-1.7B text -> Qwen3-ForcedAligner-0.6B word times, in one
      `transcribe(..., return_time_stamps=True)` call of the official qwen-asr
      toolkit. The toolkit cuts audio longer than 180 s at a low-energy point
      near the boundary, aligns each piece, and offsets the times back.

  Align-only mode (`--align-only <tag>`, output tag `<tag>+qfa`)
      skip ASR; read transcripts/<stem>.<tag>.json, join its words (or its
      segment texts when it is a text-only transcript), run only the forced
      aligner on the audio + that text. This re-times any other model's words
      (faster-whisper, Parakeet, Canary-Qwen silver text, ...) on the aligner's
      80 ms grid. Audio longer than --max-align-seconds is cut at the widest
      word gap near the boundary using the source's own timestamps; a text-only
      source without usable times is aligned in one call (model card: <= 5 min).

Install (its own Linux venv, Python 3.11/3.12). qwen-asr 0.0.6 (current on PyPI)
hard-pins transformers==4.57.6, accelerate==1.12.0, nagisa==0.2.11, soynlp==0.0.493
and drags in gradio, flask, sox and qwen-omni-utils (pyproject.toml, verified
2026-09-17), so it must not share the whisper/whisperx venv, which it would downgrade:

    python3.12 -m venv "$VENVS/qwen" && source "$VENVS/qwen/bin/activate" && pip install -U pip
    pip install -U qwen-asr soundfile librosa        # transformers backend; torch arrives via accelerate;
                                                     # soundfile/librosa for common.load_audio (ffmpeg on PATH as fallback)
    pip install -U flash-attn --no-build-isolation   # optional, only once torch is installed: --attn flash_attention_2
    pip install -U "qwen-asr[vllm]"                  # optional: --backend vllm; vllm==0.14.0 re-pins torch and may
                                                     # fight transformers==4.57.6, so leave it out of a first run

The runner sets neither HF_HOME nor TMPDIR. On the cluster export both to blackhole
before running (`export HF_HOME=/dtu/blackhole/1e/205502/hf
TMPDIR=/dtu/blackhole/1e/205502/tmp`, bench/README.md "Cluster"), else the ~4.6 GB of
weights land in the quota-limited $HOME. bench/hpc/env.sh and asr_bench.lsf are the
intended home for this; a runner exit code of 1 means at least one file failed (see
main), which the job script should tolerate so the later runners still run.

Examples:

    python bench/asr/run_qwen3_asr.py                                  # all files, tag qwen3-asr-1.7b+fa
    python bench/asr/run_qwen3_asr.py --backend vllm --limit 3          # vLLM for the 1.7B, aligner stays on transformers
    python bench/asr/run_qwen3_asr.py --align-only large-v3             # writes <stem>.large-v3+qfa.json
    python bench/asr/run_qwen3_asr.py --align-only parakeet-tdt-0.6b-v2 --max-align-seconds 0   # one aligner call per file
    python bench/asr/run_qwen3_asr.py --context-file bench/ref/drug_names.txt --tag qwen3-asr-1.7b+fa+ctx

Output words keep the ASR's own spelling and punctuation: the aligner only sees
letters, digits and apostrophes (it drops everything else and splits on
whitespace), so its items are mapped back onto the original text by the
position of those kept characters. That mapping also repairs the toolkit's
chunk join (chunk texts are concatenated with "" and not " "). If the kept
character streams ever disagree, the aligner's own token text is used and
`spelling_mapped` is false in the output; sentence splitting then degrades to
one segment.

Known bug, issue #197 (github.com/QwenLM/Qwen3-ASR/issues/197): about 2.5 % of
words come back with start_time == end_time (about 35 % of clips have at least
one). They are kept in place, given `p: 0.0` (other words carry no `p`, the
aligner has no confidence), counted in `zero_duration_words` and logged.

Caveats:
  - The toolkit's internal chunk limit for timestamps is 180 s
    (MAX_FORCE_ALIGN_INPUT_SECONDS in qwen_asr/inference/utils.py), the model
    card says up to 5 min. --max-align-seconds defaults to 180 to match the
    toolkit in both modes; set 0 to align a whole file in one call.
  - --max-new-tokens defaults to 2048 (toolkit default 512): a 180 s chunk of
    dialogue is ~500 words, and truncation would silently drop the tail.
  - The vLLM backend must be created under `if __name__ == "__main__"` (it is).
  - Timestamps are on an 80 ms grid (config timestamp_segment_time = 80).
  - GPU only in practice; the laptop is for py_compile and the pure helpers.

Verified 2026-09-17 against:
  https://github.com/QwenLM/Qwen3-ASR (README: classes, from_pretrained/LLM/transcribe/align signatures, 5 min limit, backends)
  https://huggingface.co/Qwen/Qwen3-ASR-1.7B and https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B (model cards)
  https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/qwen3_forced_aligner.py (ForcedAlignResult/Item, tokenizer, seconds)
  https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/qwen3_asr.py (ASRTranscription, chunking, "".join)
  https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/utils.py (SAMPLE_RATE, MAX_FORCE_ALIGN_INPUT_SECONDS = 180)
  https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/raw/main/config.json (timestamp_segment_time = 80)
"""
from __future__ import annotations

import sys
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from common import (Timer, audio_files, flat_words, load_audio, out_path,
                    read_transcript, standard_args, words_to_segments,
                    write_transcript)

DEFAULT_TAG = 'qwen3-asr-1.7b+fa'
QFA_SUFFIX = '+qfa'
ASR_ID = 'Qwen/Qwen3-ASR-1.7B'
FA_ID = 'Qwen/Qwen3-ForcedAligner-0.6B'


def extra_args(ap) -> None:
    ap.add_argument('--align-only', metavar='TAG', default=None,
                    help='skip ASR; align the text of transcripts/<stem>.<TAG>.json, write <TAG>+qfa')
    ap.add_argument('--asr-model', default=ASR_ID, help='HF id or local dir (Qwen/Qwen3-ASR-0.6B also works)')
    ap.add_argument('--aligner-model', default=FA_ID)
    ap.add_argument('--backend', choices=['transformers', 'vllm'], default='transformers',
                    help='ASR backend; the aligner always runs on transformers')
    ap.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    ap.add_argument('--attn', default=None, help='attn_implementation, e.g. flash_attention_2 or sdpa')
    ap.add_argument('--language', default='English', help='"English" etc.; passed to ASR and aligner')
    ap.add_argument('--max-new-tokens', type=int, default=2048)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.7, help='vLLM backend only')
    ap.add_argument('--context-file', default=None,
                    help='text file passed as `context` to the ASR (keyword biasing, e.g. drug names)')
    ap.add_argument('--max-align-seconds', type=float, default=180.0,
                    help='align-only: cut longer audio at a word gap near this length; 0 = never cut')
    ap.add_argument('--align-search-seconds', type=float, default=15.0,
                    help='align-only: look this far before the boundary for the widest word gap')


# --------------------------------------------------------------------------- #
# Pure helpers (no torch): testable on the laptop
# --------------------------------------------------------------------------- #

def _kept(ch: str) -> bool:
    """Replica of Qwen3ForcedAligner.is_kept_char: apostrophe, letters (L*), numbers (N*).
    https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/qwen3_forced_aligner.py"""
    return ch == "'" or unicodedata.category(ch)[0] in 'LN'


def item_fields(it: Any) -> Tuple[str, float, float]:
    """(text, start_s, end_s) of a ForcedAlignItem (dataclass with .text/.start_time/.end_time,
    seconds rounded to 3 decimals) or of the equivalent dict."""
    if isinstance(it, dict):
        return str(it['text']), float(it['start_time']), float(it['end_time'])
    return str(it.text), float(it.start_time), float(it.end_time)


def attach_spelling(text: str, items: Sequence[Any]) -> Optional[List[dict]]:
    """Give each aligner item the original spelling from `text`.

    The aligner tokenizes `text` on whitespace and keeps only letters, digits and
    apostrophes, so the concatenation of its item texts equals the stream of kept
    characters of `text`. Item k therefore owns the slice of `text` from its
    first kept character to the first kept character of item k+1 (item 0 also
    takes everything before it), trailing whitespace stripped, one leading space
    added, faster-whisper style. Returns None when the streams disagree.
    """
    kept = [i for i, ch in enumerate(text) if _kept(ch)]
    stream = ''.join(text[i] for i in kept).casefold()
    fields = [item_fields(it) for it in items]
    if ''.join(t for t, _, _ in fields).casefold() != stream:
        return None
    words: List[dict] = []
    pos = 0                                   # index into `kept`
    n_items = len(fields)
    for k, (t, s, e) in enumerate(fields):
        if not t:
            continue                          # the aligner never emits these; harmless guard
        begin = 0 if not words else kept[pos]
        pos += len(t)
        end = kept[pos] if pos < len(kept) else len(text)
        # `end` is the next kept character; a following item with empty text
        # cannot exist (kept-char stream), so this slice is exactly item k's.
        words.append({'w': ' ' + text[begin:end].strip(), 'start': s, 'end': e})
    assert pos == len(kept) or n_items == 0
    return words


def fallback_words(items: Sequence[Any]) -> List[dict]:
    """Aligner tokens as they are (no punctuation) when the mapping failed."""
    return [{'w': ' ' + t, 'start': s, 'end': e} for t, s, e in map(item_fields, items)]


def mark_zero_duration(words: List[dict]) -> int:
    """Issue #197: keep zero-length (or reversed) words, mark p=0, return how many."""
    n = 0
    for w in words:
        if w['end'] <= w['start']:
            w['p'] = 0.0
            n += 1
    return n


def source_units(data: dict) -> List[dict]:
    """Timed text units of a cached transcript: its words when it has them,
    else its segments (text-only transcript, README rule)."""
    words = flat_words(data)
    if words:
        return [{'text': w['w'].strip(), 'start': float(w['start']), 'end': float(w['end'])}
                for w in words if w['w'].strip()]
    return [{'text': s.get('text', '').strip(), 'start': float(s.get('start', 0.0)),
             'end': float(s.get('end', 0.0))}
            for s in data.get('segments', []) if s.get('text', '').strip()]


def plan_chunks(units: List[dict], duration: float, max_sec: float, search_sec: float
                ) -> List[Tuple[float, float, int, int]]:
    """Cut [0, duration] into pieces of at most max_sec at the widest gap between
    consecutive units whose start lies in [boundary - search_sec, boundary].
    Returns [(t0, t1, i0, i1)]: units[i0:i1] are spoken inside [t0, t1). When no
    unit starts inside the window (text-only source with 0/duration times) the
    remainder stays one piece, which the caller reports."""
    if max_sec <= 0 or duration <= max_sec or len(units) < 2:
        return [(0.0, duration, 0, len(units))]
    chunks: List[Tuple[float, float, int, int]] = []
    t0, i0 = 0.0, 0
    while duration - t0 > max_sec:
        boundary = t0 + max_sec
        best = None
        for j in range(i0 + 1, len(units)):
            s = units[j]['start']
            if s > boundary:
                break
            if s < boundary - search_sec:
                continue
            gap = s - units[j - 1]['end']
            if best is None or gap > best[0]:
                best = (gap, j)
        if best is None:
            break
        j = best[1]
        cut = (units[j - 1]['end'] + units[j]['start']) / 2.0
        cut = max(t0, min(cut, duration))
        chunks.append((t0, cut, i0, j))
        t0, i0 = cut, j
    chunks.append((t0, duration, i0, len(units)))
    return chunks


# --------------------------------------------------------------------------- #
# Models (imported lazily so the helpers above load without torch)
# --------------------------------------------------------------------------- #

def _torch_dtype(name: str):
    import torch
    return getattr(torch, name)


def load_asr(args):
    """Qwen3ASRModel with the forced aligner attached.
    https://github.com/QwenLM/Qwen3-ASR README and https://huggingface.co/Qwen/Qwen3-ASR-1.7B"""
    from qwen_asr import Qwen3ASRModel
    dtype = _torch_dtype(args.dtype)
    fa_kwargs = dict(dtype=dtype, device_map=args.device)
    if args.attn:
        fa_kwargs['attn_implementation'] = args.attn
    if args.backend == 'vllm':
        # README "vLLM backend": Qwen3ASRModel.LLM(model=..., gpu_memory_utilization=...,
        # max_new_tokens=..., forced_aligner=..., forced_aligner_kwargs=...); **kwargs -> vllm.LLM
        return Qwen3ASRModel.LLM(
            model=args.asr_model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            forced_aligner=args.aligner_model,
            forced_aligner_kwargs=fa_kwargs,
        )
    # README "Transformers backend": Qwen3ASRModel.from_pretrained(id, dtype=, device_map=,
    # attn_implementation=, max_inference_batch_size=, max_new_tokens=, forced_aligner=, forced_aligner_kwargs=)
    kwargs = dict(dtype=dtype, device_map=args.device, max_new_tokens=args.max_new_tokens,
                  forced_aligner=args.aligner_model, forced_aligner_kwargs=fa_kwargs)
    if args.attn:
        kwargs['attn_implementation'] = args.attn
    return Qwen3ASRModel.from_pretrained(args.asr_model, **kwargs)


def load_aligner(args):
    """Standalone Qwen3ForcedAligner (transformers only; the toolkit exposes no vLLM aligner).
    https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B"""
    from qwen_asr import Qwen3ForcedAligner
    kwargs = dict(dtype=_torch_dtype(args.dtype), device_map=args.device)
    if args.attn:
        kwargs['attn_implementation'] = args.attn
    return Qwen3ForcedAligner.from_pretrained(args.aligner_model, **kwargs)


def run_asr(model, audio: np.ndarray, sr: int, language: str, context: str
            ) -> Tuple[List[dict], str, str, bool]:
    """One conversation through ASR + aligner. Returns (words, text, detected language, mapped)."""
    # README "Transcribe": transcribe(audio=(np.ndarray, sr) | path | list, context="", language=None|"English",
    # return_time_stamps=False) -> List[ASRTranscription(language, text, time_stamps: ForcedAlignResult)]
    # https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/qwen3_asr.py
    res = model.transcribe(audio=[(audio, sr)], context=context, language=language,
                           return_time_stamps=True)[0]
    if res.time_stamps is None:
        # _merge_align_results returns None when every chunk decoded to empty text (a missing
        # aligner raises ValueError inside transcribe() instead), so this is a per-file failure.
        raise RuntimeError('transcribe() returned no time_stamps (every chunk decoded to empty text)')
    items = list(res.time_stamps)             # ForcedAlignResult is iterable over ForcedAlignItem
    words = attach_spelling(res.text, items)
    mapped = words is not None
    if words is None:
        words = fallback_words(items)
    return words, res.text, str(res.language or ''), mapped


def align_units(aligner, audio: np.ndarray, sr: int, units: List[dict], language: str,
                max_sec: float, search_sec: float) -> Tuple[List[dict], List[Tuple[float, float, int, int]], bool]:
    """Forced-align `units` (text with times from another model) onto `audio`.
    Returns (words, chunk plan, mapped)."""
    duration = len(audio) / sr
    if not units:
        return [], [(0.0, duration, 0, 0)], True
    plan = plan_chunks(units, duration, max_sec, search_sec)
    audios, texts = [], []
    for (t0, t1, i0, i1) in plan:
        audios.append((audio[int(round(t0 * sr)):int(round(t1 * sr))], sr))
        texts.append(' '.join(u['text'] for u in units[i0:i1]))
    # align(audio: AudioLike | List, text: str | List[str], language: str | List[str]) -> List[ForcedAlignResult]
    # https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/inference/qwen3_forced_aligner.py
    results = aligner.align(audio=audios, text=texts, language=[language] * len(audios))
    words: List[dict] = []
    mapped = True
    for (t0, _, _, _), text, res in zip(plan, texts, results):
        items = list(res)
        ws = attach_spelling(text, items)
        if ws is None:
            mapped = False
            ws = fallback_words(items)
        if t0:
            for w in ws:                       # same offsetting the toolkit does for its own chunks
                w['start'] = round(w['start'] + t0, 3)
                w['end'] = round(w['end'] + t0, 3)
        words.extend(ws)
    return words, plan, mapped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version('qwen-asr')
    except Exception:
        return None


def main() -> int:
    args = standard_args(DEFAULT_TAG, extra_args)
    if args.align_only and args.tag == DEFAULT_TAG:
        args.tag = args.align_only + QFA_SUFFIX
    context = Path(args.context_file).read_text(encoding='utf-8').strip() if args.context_file else ''
    if args.context_file and args.tag == DEFAULT_TAG:
        sys.exit('--context-file requires a distinct --tag (see file header)')

    files = audio_files(args)
    todo = [p for p in files if args.force or not out_path(args, p).exists()]
    if not todo:
        print(f'nothing to do: {len(files)} files already have tag {args.tag}', flush=True)
        return 0

    t_load = time.time()
    if args.align_only:
        aligner, model = load_aligner(args), None
        print(f'loaded {args.aligner_model} in {time.time() - t_load:.1f}s (align-only from tag {args.align_only})', flush=True)
    else:
        model, aligner = load_asr(args), None
        print(f'loaded {args.asr_model} + {args.aligner_model} [{args.backend}] in {time.time() - t_load:.1f}s', flush=True)

    total_audio = total_dt = 0.0
    total_words = total_zero = 0
    n_done = 0
    failed: List[str] = []
    for path in files:
        out = out_path(args, path)
        if out.exists() and not args.force:
            print(f'skip {path.name}', flush=True)
            continue
        try:
            audio, sr = load_audio(path)
            duration = len(audio) / sr
            extra: dict = {'aligner_model': args.aligner_model, 'language': args.language,
                           'qwen_asr_version': _version()}
            if args.align_only:
                src = Path(args.out_dir) / f'{path.stem}.{args.align_only}.json'
                if not src.exists():
                    print(f'{path.name}: no source transcript {src.name}, skipped', flush=True)
                    continue
                units = source_units(read_transcript(src))
                with Timer() as t:
                    words, plan, mapped = align_units(aligner, audio, sr, units, args.language,
                                                      args.max_align_seconds, args.align_search_seconds)
                longest = max((t1 - t0 for t0, t1, _, _ in plan), default=duration)
                if args.max_align_seconds and longest > args.max_align_seconds + 1e-6:
                    print(f'  warning: a {longest:.0f}s piece exceeds --max-align-seconds '
                          f'(no timed word gap to cut at; model card allows 5 min)', flush=True)
                extra.update(source_tag=args.align_only, align_chunks=len(plan),
                             align_chunk_bounds=[[round(t0, 3), round(t1, 3)] for t0, t1, _, _ in plan],
                             source_words=len(units))
            else:
                with Timer() as t:
                    words, text, lang, mapped = run_asr(model, audio, sr, args.language, context)
                extra.update(asr_model=args.asr_model, backend=args.backend,
                             detected_language=lang, context_chars=len(context))
            zero = mark_zero_duration(words)
            segments = words_to_segments(words)
            extra.update(n_words=len(words), zero_duration_words=zero, spelling_mapped=mapped)
            write_transcript(out, path, args.tag, duration, t.seconds, segments, word_timestamps=True, extra=extra)
        except Exception as e:  # noqa: BLE001  one bad file (CUDA OOM on a long whole-file align, an HF hub
            # hiccup, empty ASR text, a corrupt source JSON) must not abort the tag run; KeyboardInterrupt
            # is not an Exception and still stops it. Same convention as run_faster_whisper.py.
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            continue
        total_audio += duration
        total_dt += t.seconds
        total_words += len(words)
        total_zero += zero
        n_done += 1
        note = '' if mapped else '  (spelling NOT mapped, aligner tokens written)'
        print(f'{path.name}: {duration:6.1f}s audio, {t.seconds:5.1f}s, {len(words)} words, '
              f'{len(segments)} segs, {zero} zero-duration{note}', flush=True)

    if n_done:
        share = 100.0 * total_zero / total_words if total_words else 0.0
        rtf = total_dt / total_audio if total_audio else float('nan')
        print(f'DONE {n_done} files, {len(failed)} failed, RTF {rtf:.3f}, mean {total_dt / n_done:.1f}s per '
              f'conversation, zero-duration words {total_zero}/{total_words} ({share:.2f}%)', flush=True)
    if failed:
        print(f'FAILED {len(failed)}: {" ".join(failed)}', file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
