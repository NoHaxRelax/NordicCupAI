"""faster-whisper (CTranslate2) ASR runner for the bench, any Whisper size.

Transcribes every MP3 under data/audio with word timestamps and writes one
transcript per conversation in the bench/README.md schema,
``<out-dir>/<audio stem>.<tag>.json``. This generalises transcribe_cache.py
(same output, same decoding settings) to the bench/asr/common.py conventions:
shared 16 kHz mono decoding, tag naming, --limit/--force, and a timer that
covers only ``transcribe()`` plus consuming its generator (model load excluded).

Install (Linux venv; the H100 nodes have CUDA 12 drivers):

    pip install "faster-whisper>=1.2" soundfile librosa numpy
    # GPU: current ctranslate2 needs cuBLAS for CUDA 12 and cuDNN 9
    # (https://github.com/SYSTRAN/faster-whisper#gpu, "Install with pip (Linux only)")
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
    export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`

LD_LIBRARY_PATH must be set before Python starts. Weights come from the HF hub
on first use (Systran/faster-whisper-* and mobiuslabsgmbh/faster-whisper-large-v3-turbo);
set HF_HOME to blackhole first.

Usage:

    python bench/asr/run_faster_whisper.py                               # large-v3, beam 5, float16 -> tag large-v3
    python bench/asr/run_faster_whisper.py --model large-v3-turbo        # tag large-v3-turbo
    python bench/asr/run_faster_whisper.py --model distil-large-v3 --vad # tag distil-large-v3+vad
    python bench/asr/run_faster_whisper.py --model large-v3 --beam 1     # tag large-v3+beam1
    KMP_DUPLICATE_LIB_OK=TRUE python bench/asr/run_faster_whisper.py --device cpu --model tiny.en \
        --limit 1 --tag smoke-tiny --out-dir bench/results/smoke          # CPU smoke test

Tag rule (bench/README.md): the default tag is the model name, with ``+vad``
appended when --vad is on and ``+beam<N>`` when --beam is not 5, so variants
never overwrite each other. --tag overrides.

Caveats:
- Whisper word times come from cross-attention DTW (100-400 ms jitter) and
  the model hallucinates on silence; large-v3-turbo has documented broken
  word timestamps on short clips. See research/01-asr-timestamped.md.
- --vad runs faster-whisper's bundled Silero VAD, decodes only the speech
  chunks and maps every word/segment time back to the original audio
  (transcribe.py: restore_speech_timestamps), so start/end stay in seconds of
  the original file. The transcript then also carries ``duration_after_vad``.
- The audio array from common.load_audio is passed straight to transcribe();
  faster-whisper assumes 16 kHz for ndarray input (feature_extractor.sampling_rate).
- --device cuda:N is split into device='cuda', device_index=N for CTranslate2.
- The Windows CUDA DLL block below (from transcribe_cache.py) is a no-op on Linux.
- One failing file (decode error, CUDA OOM, CTranslate2 runtime error, a
  hallucination loop) is logged as ``FAIL <name>`` on stderr with a traceback
  and skipped; the run continues and exits 1 at the end if anything failed, so
  the other conversations still get transcripts and the log names what to redo.
- The skip check is provenance-aware: an existing ``<stem>.<tag>.json`` is only
  reused when its ``config.backend`` is ``faster-whisper``, i.e. this runner
  wrote it. transcribe_cache.py writes the same path for the same model name
  but hands faster-whisper the MP3 path (its own PyAV decode, not
  common.load_audio), on a laptop GPU, without a config block; such files are
  redone here instead of silently accepted, so they never stand in for a bench row.
"""
from __future__ import annotations

import os
import site
import sys
import time
import traceback
from pathlib import Path

# Windows: the CUDA runtime DLLs live in pip's nvidia/* packages, not on PATH.
# Kept from transcribe_cache.py. os.add_dll_directory only exists on Windows and
# the nvidia/*/bin folders only exist there, so on Linux nothing happens.
if hasattr(os, 'add_dll_directory'):
    _nv = Path(site.getsitepackages()[0]) / 'Lib' / 'site-packages' / 'nvidia'
    for _d in ('cublas', 'cudnn', 'cuda_nvrtc'):
        _p = _nv / _d / 'bin'
        if _p.is_dir():
            os.add_dll_directory(str(_p))
            os.environ['PATH'] = str(_p) + os.pathsep + os.environ['PATH']
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (SR, Timer, audio_files, load_audio, out_path,  # noqa: E402
                    read_transcript, standard_args, write_transcript)

# Names resolved by faster_whisper.utils._MODELS -> HF repo ids
# (https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py):
#   large-v3 -> Systran/faster-whisper-large-v3
#   large-v3-turbo -> mobiuslabsgmbh/faster-whisper-large-v3-turbo
#   distil-large-v3 -> Systran/faster-distil-whisper-large-v3
#   distil-large-v3.5, medium.en, small.en, tiny.en -> Systran/faster-*-whisper-*
MODELS = ('large-v3', 'large-v3-turbo', 'distil-large-v3', 'distil-large-v3.5',
          'medium.en', 'small.en', 'tiny.en')
# CTranslate2 compute types faster-whisper accepts for WhisperModel(compute_type=...)
# (https://github.com/SYSTRAN/faster-whisper#usage: float16 on GPU, int8 on CPU,
#  int8_float16 for 8-bit on GPU).
COMPUTE = ('float16', 'int8', 'int8_float16', 'float32', 'default')
DEFAULT_BEAM = 5


def extra_args(ap) -> None:
    ap.add_argument('--model', default='large-v3', choices=MODELS)
    ap.add_argument('--beam', type=int, default=DEFAULT_BEAM, help='beam size (default 5)')
    ap.add_argument('--temperature', type=float, default=None,
                    help='single decode temperature (e.g. 0) instead of the fallback ladder')
    ap.add_argument('--no-condition', action='store_true',
                    help='condition_on_previous_text=False (no cross-window prompting)')
    ap.add_argument('--vad', action='store_true',
                    help='faster-whisper Silero VAD filter before decoding (default off)')
    ap.add_argument('--compute', default=None, choices=COMPUTE,
                    help='CTranslate2 compute type; default float16 on cuda, int8 on cpu')


def default_tag(args) -> str:
    tag = args.model
    if args.temperature is not None and args.no_condition:
        tag += '+clean'            # research/06 idea 6: no fallback ladder, no conditioning
    elif args.temperature is not None:
        tag += f'+t{args.temperature:g}'
    elif args.no_condition:
        tag += '+nocond'
    if args.vad:
        tag += '+vad'
    if args.beam != DEFAULT_BEAM:
        tag += f'+beam{args.beam}'
    return tag


def to_segment(s) -> dict:
    """faster_whisper.transcribe.Segment -> README segment dict.

    Segment fields: start, end, text, words (Optional[List[Word]]); Word fields:
    start, end, word, probability (word keeps the tokenizer's leading space).
    https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
    """
    words = [{'w': w.word, 'start': float(w.start), 'end': float(w.end),
              'p': float(w.probability)} for w in (s.words or [])]
    text = ''.join(w['w'] for w in words) if words else s.text
    return {'start': float(s.start), 'end': float(s.end), 'text': text, 'words': words}


def is_ours(out: Path) -> bool:
    """True when `out` was written by this runner (config.backend == 'faster-whisper').

    transcribe_cache.py writes the identical path for the same model name but
    without a config block and from a different decode path, so only a bench
    transcript counts as done. Missing, truncated or non-JSON files count as not done.
    """
    try:
        return read_transcript(out).get('config', {}).get('backend') == 'faster-whisper'
    except Exception:  # noqa: BLE001
        return False


def needs_run(args, path: Path) -> bool:
    if args.force:
        return True
    out = out_path(args, path)
    return not (out.exists() and is_ours(out))


def main() -> int:
    args = standard_args(None, extra_args)   # --audio-dir --out-dir --tag --limit --force --device
    if not args.tag:
        args.tag = default_tag(args)
    device, _, index = args.device.partition(':')
    device_index = int(index) if index else 0
    if args.compute is None:
        args.compute = {'cuda': 'float16', 'cpu': 'int8'}.get(device, 'default')

    files = audio_files(args)
    todo = [f for f in files if needs_run(args, f)]
    # Existing outputs another writer produced (transcribe_cache.py): redone, not reused.
    foreign = [f for f in todo if not args.force and out_path(args, f).exists()]
    print(f'{len(files)} files, {len(todo)} to transcribe'
          + (f' ({len(foreign)} existing but not written by this runner, redoing)' if foreign else '')
          + f', tag {args.tag}, model {args.model}, beam {args.beam}, vad {args.vad}, '
            f'{device}:{device_index} {args.compute}', flush=True)
    if not todo:
        return 0

    import ctranslate2                                   # noqa: E402
    import faster_whisper                                # noqa: E402
    from faster_whisper import WhisperModel              # noqa: E402

    t0 = time.time()
    # WhisperModel(model_size_or_path, device='auto', device_index=0, compute_type='default', ...)
    # https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
    model = WhisperModel(args.model, device=device, device_index=device_index,
                         compute_type=args.compute)
    print(f'loaded {args.model} in {time.time() - t0:.1f}s '
          f'(faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__})',
          flush=True)
    config = {'backend': 'faster-whisper', 'faster_whisper': faster_whisper.__version__,
              'ctranslate2': ctranslate2.__version__, 'model': args.model, 'beam': args.beam,
              'vad': args.vad, 'compute': args.compute, 'device': args.device, 'language': 'en'}

    total_audio = total_dt = 0.0
    todo_set = set(todo)
    failed = []
    for path in files:
        if path not in todo_set:
            print(f'skip {path.name}', flush=True)
            continue
        out = out_path(args, path)
        try:
            audio, sr = load_audio(path)
            assert sr == SR, f'load_audio returned {sr} Hz, faster-whisper assumes {SR}'
            with Timer() as t:
                # transcribe(audio: Union[str, BinaryIO, np.ndarray], language=None, beam_size=5,
                #            word_timestamps=False, vad_filter=False, ...) -> (Iterable[Segment],
                #            TranscriptionInfo); the segments are a lazy generator.
                # https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
                kw = dict(language='en', beam_size=args.beam, word_timestamps=True, vad_filter=args.vad)
                if args.temperature is not None:
                    kw['temperature'] = args.temperature            # a single value disables the fallback ladder
                if args.no_condition:
                    kw['condition_on_previous_text'] = False
                segs, info = model.transcribe(audio, **kw)
                segs = list(segs)
            segments = [to_segment(s) for s in segs]
            # TranscriptionInfo.duration = audio.shape[0] / sampling_rate (before VAD);
            # duration_after_vad = kept speech length when vad_filter=True.
            duration = float(info.duration)
            extra = {'config': config}
            if args.vad:
                extra['duration_after_vad'] = float(info.duration_after_vad)
            write_transcript(out, path, args.tag, duration, t.seconds, segments, extra=extra)
        except Exception as e:  # noqa: BLE001  one bad file must not abort the tag run
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            continue
        total_audio += duration
        total_dt += t.seconds
        n_words = sum(len(s['words']) for s in segments)
        print(f'{path.name}: {duration:6.1f}s audio, {t.seconds:5.1f}s, '
              f'{len(segments)} segs, {n_words} words', flush=True)
    n_done = len(todo) - len(failed)
    if n_done and total_audio:
        print(f'DONE {n_done} files, RTF {total_dt / total_audio:.3f}, '
              f'mean {total_dt / n_done:.1f}s per conversation', flush=True)
    if failed:
        print(f'FAILED {len(failed)}: {" ".join(failed)}', file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
