"""Parakeet-TDT runner (NVIDIA NeMo): native word and segment timestamps.

Transcribes every MP3 under data/audio with nvidia/parakeet-tdt-0.6b-v2
(default, English only) or nvidia/parakeet-tdt-0.6b-v3 (25 European
languages, auto-detected, no language argument) and writes
transcripts/<stem>.<tag>.json in the schema of bench/README.md. The TDT
decoder emits word / segment / char timestamps in the same forward pass. Word
boundaries sit on the 80 ms encoder frame grid: NeMo computes
start = start_offset * window_stride(0.01) * subsampling(8)
(nemo/collections/asr/parts/utils/timestamp_utils.py, process_timestamp_outputs).

Tags produced: parakeet-tdt-0.6b-v2 (default), parakeet-tdt-0.6b-v3 (--model v3).

Install (Linux, its own venv: NeMo pins torch / lhotse / numpy versions that
conflict with the whisperx + pyannote venv; see bench/hpc/env.sh):

    python -m venv $VENVS/nemo && source $VENVS/nemo/bin/activate
    pip install -U pip
    pip install -U "nemo_toolkit[asr,speechlm2]"   # asr: this runner (card Usage). speechlm2: peft etc.
                                                   # for run_canary_qwen.py, which shares this venv;
                                                   # the asr extra alone cannot import SALM (3.0.0)
    pip install soundfile librosa          # common.load_audio (mp3 decode); ffmpeg on PATH is the fallback
    export HF_HOME=/dtu/blackhole/1e/205502/hf TMPDIR=/dtu/blackhole/1e/205502/tmp \
           NEMO_CACHE_DIR=/dtu/blackhole/1e/205502/nemo_cache
    # pre-warm the weights from the login node (network) so the GPU job runs offline-safe
    # (loads on CPU there; ~2.4 GB fp32 per model):
    python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')"
    python -c "import nemo.collections.asr as a; a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v3')"

Run (from medical-appointment/):

    python bench/asr/run_parakeet.py                          # v2  -> tag parakeet-tdt-0.6b-v2
    python bench/asr/run_parakeet.py --model v3               # v3  -> tag parakeet-tdt-0.6b-v3
    python bench/asr/run_parakeet.py --limit 2 --force
    python bench/asr/run_parakeet.py --chunk-s 600 --local-attn   # long-audio path only (audio >> 24 min)

Caveats:

- Whole clip in ONE transcribe() call by default. The cards state 24 min of
  audio in a single pass with full attention on an 80 GB card, so the 1-3.5 min
  conversations never need chunking. --chunk-s is the escape hatch for the
  long-audio path: it cuts the sample array at the quietest 20 ms inside the
  last --search-s seconds of every window and offsets the timestamps by the
  chunk start. --local-attn applies the v3 card's long-form recipe
  (rel_pos_local_attn, context [256, 256]); it changes the encoder and hence
  the timestamps, so keep it off for the bench.
- Trailing-silence bug, NeMo/Speech issue #15757 (open, "waiting on
  maintainers"): a short utterance (2.2 s) followed by ~400 ms of appended
  zeros decodes to '' with parakeet-tdt-0.6b-v3 (nemo-toolkit 2.7.3), because
  the preprocessor normalises the log-mel over the silence as if it were
  speech. Full 1-3.5 min clips are far from that regime, but the runner still
  retries once with trailing near-silence trimmed when a decode comes back
  empty, and prints a warning. This matters the day someone VAD-chunks.
- Word spelling: NeMo words carry their punctuation ("Fabricius.") and have no
  leading space. The runner prepends one space to every word so that
  ''.join(w['w']) reproduces the text, which is the convention
  transcribe_cache.py, model.py and span_ceiling.py rely on (Whisper tokens
  carry that space themselves). Consumers .strip() unit text anyway.
- Segments: the model's own segments are sentence chunks split at . ? !
  (NeMo segment_seperators default). Words are assigned to a segment by end
  time; leftovers become extra sentence segments via common.words_to_segments.
  --sentence-segments ignores the model segments and splits at terminal
  punctuation only (identical in practice, useful if a NeMo release changes
  the segment rule). research/01 warns not to build evidence spans from
  segment edges; the consumers use word times, so this is moot here.
- No word probabilities ('p' omitted; TDT does not expose per-word confidence
  through transcribe()). Digits vs number words are model-dependent
  (research/01-asr-timestamped.md); nothing is normalised in a runner.
- Weights: from_pretrained fetches the .nemo through huggingface_hub (HF_HOME)
  and unpacks it into a temp dir (TMPDIR); both must point at blackhole, $HOME
  is quota-limited. NeMo also has a cache of its own: nemo/constants.py defines
  NEMO_ENV_CACHE_DIR = "NEMO_CACHE_DIR" and resolve_cache_dir() defaults to
  ~/.cache/torch/NeMo/... (used for snapshot-style downloads, hf_hub_cache/<model>),
  so export NEMO_CACHE_DIR as well, inside the LSF job too, or that download
  lands in the quota-limited $HOME and fails silently. The first run needs
  outbound network from the GPU node unless the caches were pre-warmed from the
  login node (Install above).
- Errors: one bad file (mp3 decode error, CUDA OOM on one clip, a decode that
  stays empty after the #15757 retry) is logged as FAIL and skipped, nothing is
  written for it, and the run continues; the exit code is 1 at the end if any
  file failed, so the LSF job still produces every other transcript.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODELS = {
    'v2': 'nvidia/parakeet-tdt-0.6b-v2',   # https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
    'v3': 'nvidia/parakeet-tdt-0.6b-v3',   # https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
}
DEFAULT_MODEL = MODELS['v2']
DEFAULT_TAG = 'parakeet-tdt-0.6b-v2'
FRAME_S = 0.08          # TDT timestamp grid: 10 ms window stride x 8 subsampling


def tag_for(model_id: str) -> str:
    """bench/README.md tag rule: lowercase HF name without the org."""
    return model_id.split('/')[-1].lower()


def resolve_model(name: str) -> str:
    return MODELS.get(name.strip().lower(), name)


def extra_args(ap) -> None:
    ap.add_argument('--model', default=DEFAULT_MODEL,
                    help="'v2', 'v3' or a full HF id (default %(default)s)")
    ap.add_argument('--chunk-s', type=float, default=0.0,
                    help='0 = whole clip in one transcribe() call (default). Otherwise cut the '
                         'array before every N seconds at the quietest point and offset the '
                         'timestamps; long-audio escape hatch only')
    ap.add_argument('--search-s', type=float, default=5.0,
                    help='with --chunk-s: look back this many seconds for the quietest cut')
    ap.add_argument('--local-attn', action='store_true',
                    help='rel_pos_local_attn [256,256] (v3 card long-form recipe); changes '
                         'timestamps, do not use for the bench')
    ap.add_argument('--wav-input', action='store_true',
                    help='hand transcribe() a temp 16 kHz float32 wav instead of the numpy array')
    ap.add_argument('--sentence-segments', action='store_true',
                    help='ignore model segments; build them with common.words_to_segments')


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_model(model_id: str, device: str, local_attn: bool):
    # Loading and transcribe(..., timestamps=True) verified on the model cards:
    # https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2  and  .../parakeet-tdt-0.6b-v3
    import nemo.collections.asr as nemo_asr
    logging.getLogger('nemo_logger').setLevel(logging.WARNING)   # best effort; NeMo is chatty
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    if local_attn:
        # https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 ("long-form audio": up to 3 h)
        model.change_attention_model(self_attention_model='rel_pos_local_attn',
                                     att_context_size=[256, 256])
    model = model.to(device).eval()
    print(f'loaded {model_id} on {device} in {time.time() - t0:.1f}s', flush=True)
    return model


def transcribe_array(model, audio: np.ndarray, sr: int, wav_input: bool):
    """One transcribe() call on one 16 kHz float32 mono array. Returns the
    Hypothesis: .text plus .timestamp = {'word': [...], 'segment': [...], 'char': [...]}
    with entries {'word'|'segment': str, 'start': s, 'end': s, 'start_offset', 'end_offset'}.
    Signature verified in nemo/collections/asr/parts/mixins/transcription.py:
    transcribe(audio: Union[str, List[str], np.ndarray, DataLoader], batch_size=4,
    return_hypotheses=False, ..., verbose=True, timestamps=None, ...); numpy arrays are
    wrapped in a list and converted with torch.as_tensor, sample rate taken from
    model.cfg.sample_rate (16 kHz), which common.load_audio already produces."""
    if wav_input:
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / 'in.wav'
            sf.write(str(wav), audio, sr, subtype='PCM_16')      # 16-bit PCM: universally readable by NeMo's audio backends
            hyps = model.transcribe([str(wav)], batch_size=1, timestamps=True, verbose=False)
    else:
        hyps = model.transcribe([audio], batch_size=1, timestamps=True, verbose=False)
    if isinstance(hyps, tuple):          # pre-2.x RNNT/TDT returned (best, all_hyps)
        hyps = hyps[0]
    return hyps[0]


# --------------------------------------------------------------------------- #
# Hypothesis -> transcript schema
# --------------------------------------------------------------------------- #

def hyp_words(hyp, offset: float = 0.0) -> List[dict]:
    ts = getattr(hyp, 'timestamp', None) or {}
    if 'word' not in ts:
        # rnnt_decoding.compute_rnnt_timestamps builds the dict as {'timestep': ...} and
        # only adds 'word' when word_offsets is not None: an EMPTY decode has no 'word'
        # key at all. That is not a broken model, just no words.
        if not (getattr(hyp, 'text', '') or '').strip():
            return []
        raise RuntimeError('hypothesis has text but no word timestamps; transcribe('
                           'timestamps=True) was expected to fill hyp.timestamp["word"]')
    words = []
    for e in ts['word']:
        w = str(e['word'])
        if not w:
            continue
        words.append({'w': ' ' + w, 'start': float(e['start']) + offset,
                      'end': float(e['end']) + offset})
    return words


def hyp_segments(hyp, words: List[dict], offset: float = 0.0,
                 use_model_segments: bool = True) -> List[dict]:
    """Model segments (sentence chunks at . ? !) with their words attached by
    end time; leftover words become extra sentence segments."""
    ts = getattr(hyp, 'timestamp', None) or {}
    msegs = ts.get('segment') if use_model_segments else None
    if not msegs:
        return common.words_to_segments(words)
    segs, i, mismatch = [], 0, 0
    for s in msegs:
        s_start, s_end = float(s['start']) + offset, float(s['end']) + offset
        ws: List[dict] = []
        while i < len(words) and words[i]['end'] <= s_end + 1e-6:
            ws.append(words[i]); i += 1
        if not ws:
            continue
        text = ''.join(w['w'] for w in ws)
        if text.split() != str(s.get('segment', '')).split():
            mismatch += 1
        segs.append({'start': s_start, 'end': s_end, 'text': text, 'words': ws})
    if i < len(words):
        segs.extend(common.words_to_segments(words[i:]))
    if mismatch:
        print(f'  warning: {mismatch}/{len(msegs)} model segments disagree with their words '
              f'(kept the words, check the segment rule)', flush=True)
    return segs


# --------------------------------------------------------------------------- #
# Audio helpers (long-audio path and the #15757 retry)
# --------------------------------------------------------------------------- #

def quietest_sample(x: np.ndarray, sr: int, win_s: float = 0.02) -> int:
    """Index of the centre of the lowest-energy 20 ms window in x."""
    win = max(1, int(win_s * sr))
    m = len(x) // win
    if m == 0:
        return len(x)
    energy = (x[:m * win].reshape(m, win).astype(np.float64) ** 2).mean(axis=1)
    return int(energy.argmin()) * win + win // 2


def split_audio(audio: np.ndarray, sr: int, chunk_s: float,
                search_s: float = 5.0) -> List[Tuple[int, np.ndarray]]:
    """[(start_sample, piece), ...]; pieces are at most chunk_s long and are cut
    at the quietest point inside the last search_s seconds of each window."""
    n, max_len = len(audio), int(chunk_s * sr)
    if chunk_s <= 0 or n <= max_len:
        return [(0, audio)]
    search = min(max_len // 2, max(1, int(search_s * sr)))
    out, start = [], 0
    while n - start > max_len:
        lo, hi = start + max_len - search, start + max_len
        cut = lo + quietest_sample(audio[lo:hi], sr)
        cut = max(cut, start + 1)
        out.append((start, audio[start:cut]))
        start = cut
    out.append((start, audio[start:]))
    return out


def trim_trailing_silence(audio: np.ndarray, sr: int, thresh: float = 1e-3,
                          keep_s: float = 0.1) -> np.ndarray:
    nz = np.flatnonzero(np.abs(audio) > thresh)
    if len(nz) == 0:
        return audio
    return audio[:min(len(audio), int(nz[-1]) + 1 + int(keep_s * sr))]


def release_cuda() -> None:
    """After a CUDA error on one clip, hand the cached blocks back so the next
    clip gets a clean allocator instead of inheriting the OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  best effort, never mask the original error
        pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = common.standard_args(DEFAULT_TAG, extra_args)
    model_id = resolve_model(args.model)
    if args.tag == DEFAULT_TAG and tag_for(model_id) != DEFAULT_TAG:
        args.tag = tag_for(model_id)          # --model v3 without --tag -> parakeet-tdt-0.6b-v3
    files = common.audio_files(args)
    todo = []
    for path in files:
        if common.out_path(args, path).exists() and not args.force:
            print(f'skip {path.name}', flush=True)
        else:
            todo.append(path)
    if not todo:
        print('nothing to do', flush=True)
        return 0

    model = load_model(model_id, args.device, args.local_attn)
    total_audio = total_dt = 0.0
    failed: List[str] = []
    for path in todo:
        out = common.out_path(args, path)
        try:
            audio, sr = common.load_audio(path)
            duration = len(audio) / sr
            pieces = split_audio(audio, sr, args.chunk_s, args.search_s)
            words: List[dict] = []
            segments: List[dict] = []
            dt = 0.0
            for start, piece in pieces:
                off = start / sr
                t0 = time.time()
                hyp = transcribe_array(model, piece, sr, args.wav_input)
                if not (getattr(hyp, 'text', '') or '').strip():
                    # NeMo/Speech #15757: empty decode with a silent tail -> retry trimmed
                    trimmed = trim_trailing_silence(piece, sr)
                    if len(trimmed) < len(piece):
                        print(f'  warning: {path.name} @ {off:.1f}s decoded empty; retrying with '
                              f'{(len(piece) - len(trimmed)) / sr:.2f}s of trailing silence trimmed '
                              f'(NeMo/Speech #15757)', flush=True)
                        hyp = transcribe_array(model, trimmed, sr, args.wav_input)
                dt += time.time() - t0
                pw = hyp_words(hyp, off)
                if not pw:
                    print(f'  warning: {path.name} @ {off:.1f}s: no words in this piece', flush=True)
                words.extend(pw)
                segments.extend(hyp_segments(hyp, pw, off, not args.sentence_segments))
            if not words:
                # do not cache an empty transcript: it would be skipped as done next run
                raise RuntimeError('decoded empty for the whole clip (after the #15757 retry)')
            extra = {'hf_model': model_id, 'frame_s': FRAME_S,
                     'text': ''.join(w['w'] for w in words).strip()}
            if len(pieces) > 1:
                extra['chunk_s'] = args.chunk_s
            common.write_transcript(out, path, args.tag, duration, dt, segments,
                                    word_timestamps=True, extra=extra)
        except Exception as e:  # noqa: BLE001  one bad file must not abort the tag run
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            if 'cuda' in str(e).lower():
                release_cuda()
            continue
        total_audio += duration; total_dt += dt
        print(f'{path.name}: {duration:6.1f}s audio, {dt:5.1f}s, {len(words)} words, '
              f'{len(segments)} segs', flush=True)
    n_done = len(todo) - len(failed)
    print(f'DONE {n_done}/{len(todo)} files, tag {args.tag}, '
          f'RTF {total_dt / max(total_audio, 1e-9):.3f}, '
          f'mean {total_dt / max(1, n_done):.1f}s per conversation'
          + (f', FAILED {len(failed)}: {failed}' if failed else ''), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
