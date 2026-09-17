"""Canary-Qwen-2.5B runner (NVIDIA NeMo SALM): TEXT ONLY, the silver reference.

nvidia/canary-qwen-2.5b is a FastConformer encoder feeding a Qwen3-1.7B
decoder (SALM, speech-augmented language model, nemo.collections.speechlm2).
It has the best English WER of the open models that run in NeMo (5.63 on the
Open ASR leaderboard, research/01-asr-timestamped.md) and NO timestamps of any
kind. In this bench it is the reference text that compare.py computes WER
against and a text source for external aligners (align_mms.py). It writes
transcripts/<stem>.canary-qwen-2.5b.json with "word_timestamps": false, every
segment with words: [], and one segment per decoded chunk whose start/end are
the chunk boundaries in seconds (with --chunk-s 0: exactly one segment 0 ..
duration). A top-level "text" key holds the joined transcript.

Tag produced: canary-qwen-2.5b.

Install (same NeMo venv as run_parakeet.py; NeMo needs its own venv, it
conflicts with the whisperx + pyannote family). Card:
https://huggingface.co/nvidia/canary-qwen-2.5b

    pip install -U "nemo_toolkit[asr,speechlm2]"
    pip install soundfile librosa           # common.load_audio (mp3 decode)
    export HF_HOME=/dtu/blackhole/1e/205502/hf TMPDIR=/dtu/blackhole/1e/205502/tmp \
           NEMO_CACHE_DIR=/dtu/blackhole/1e/205502/nemo_cache
    # pre-warm all three caches from the login node (network) so the GPU job runs offline-safe:
    python -c "from nemo.collections.speechlm2.models import SALM; SALM.from_pretrained('nvidia/canary-qwen-2.5b')"

The speechlm2 extra is not optional: pip resolves "nemo_toolkit[asr]" to
nemo-toolkit 3.0.0 (PyPI, 2026-08-07, requires_python >= 3.10), whose asr extra
is asr-only + common (lhotse, transformers, soundfile, ...) and does NOT carry
peft, while nemo/collections/speechlm2/models/salm.py at tag v3.0.0 has an
unguarded top-level `from peft import PeftModel`. With the asr extra alone,
`from nemo.collections.speechlm2.models import SALM` raises ImportError before
any weights load. peft<=0.18.0 (with nemo_automodel and flashoptim) ships in
the speechlm2 extra (PyPI requires_dist for 3.0.0). SALM is exported by
nemo.collections.speechlm2.models.__init__ in the release, so the card's
git+https install line is not needed. The runner checks for peft up front and
exits 2 with the pip line above when it is missing.

Run (from medical-appointment/):

    python bench/asr/run_canary_qwen.py                  # tag canary-qwen-2.5b, 30 s chunks
    python bench/asr/run_canary_qwen.py --chunk-s 0      # whole clip in one generate() call
    python bench/asr/run_canary_qwen.py --limit 2 --force

Caveats:

- The card states "maximum audio duration in training was 40 s" and an NVIDIA
  maintainer confirms on the HF discussion "Chunked inference?" that long audio
  is handled by chunking (their Spaces demo does it; "will eventually land in
  the speechlm2 collection"). The recommended chunk length is not stated
  (UNVERIFIED), so the default is 30 s windows cut at the quietest 20 ms inside
  the last --search-s seconds of each window. The cut points are the runner's,
  not the model's: a word can still be split, so do not read chunk edges as
  evidence. --chunk-s 0 sends the whole 1-3.5 min clip in one call; it runs
  (the encoder is length-agnostic) but sits well outside the training
  distribution, expect dropped stretches.
- Audio is handed over as a temp 16 kHz float32 wav path per chunk, the input
  form documented on the card ({"audio": ["speech.wav"]}); SALM loads it with
  lhotse and resamples to 16 kHz, a no-op here, so the samples are exactly
  common.load_audio's. Passing tensors via generate(audios=, audio_lens=) is
  possible (salm_generate.py) but the prompt tensor format is not documented.
- Decoding: greedy by default through model.generate; the answer is cut at the
  first model.text_eos_id (what examples/speechlm2/salm_generate.py does in
  parse_hyp) before model.tokenizer.ids_to_text. max_new_tokens is sized from
  the chunk length (8 tokens per second + 64, min 128) unless --max-new-tokens
  is set; a too-small value silently truncates the tail of a chunk.
- bf16 on the H100 (the card's precision). --dtype float32 for a CPU smoke test
  is slow but works; float16 is untested.
- Weights: SALM.from_pretrained pulls THREE HF repos, not one. HFHubMixin
  .from_pretrained (nemo/collections/speechlm2/parts/hf_hub.py) sets
  cfg.pretrained_weights=False and instantiates SALM; load_pretrained_hf still
  calls AutoConfig.from_pretrained('Qwen/Qwen3-1.7B') (network + HF cache) and
  setup_speech_encoder calls load_pretrained_nemo_config(ASRModel,
  'nvidia/canary-1b-flash'), which goes through ASRModel.from_pretrained(...,
  return_config=True) and hf_hub_download of the whole canary-1b-flash .nemo
  (several GB) into HF_HOME. Budget ~10 GB (canary-qwen-2.5b safetensors ~5 GB
  + the canary-1b-flash .nemo + the Qwen3 config), not ~5 GB. The .nemo is
  unpacked into tempfile.TemporaryDirectory() (TMPDIR); map_location defaults
  to cuda when available. NeMo also has a cache of its own: nemo/constants.py
  defines NEMO_ENV_CACHE_DIR = "NEMO_CACHE_DIR" and resolve_cache_dir() defaults
  to ~/.cache/torch/NeMo/... (snapshot-style downloads, hf_hub_cache/<model>).
  So HF_HOME, TMPDIR and NEMO_CACHE_DIR must all point at blackhole and be
  exported INSIDE the LSF job; $HOME is quota-limited and a miss there fails
  silently. The first run needs outbound network from the gpuh100 node unless
  the caches were pre-warmed from the login node (Install above).
- Errors: one bad file (mp3 decode error, CUDA OOM on one chunk, lhotse failing
  on the temp wav, a tokenizer error) is logged as FAIL and skipped, nothing is
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

DEFAULT_MODEL = 'nvidia/canary-qwen-2.5b'     # https://huggingface.co/nvidia/canary-qwen-2.5b
DEFAULT_TAG = 'canary-qwen-2.5b'
DEFAULT_PROMPT = 'Transcribe the following:'  # the card's ASR prompt


def extra_args(ap) -> None:
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--chunk-s', type=float, default=30.0,
                    help='max seconds of audio per generate() call; 0 = whole clip '
                         '(training max was 40 s)')
    ap.add_argument('--search-s', type=float, default=5.0,
                    help='look back this many seconds for the quietest cut point')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--prompt', default=DEFAULT_PROMPT)
    ap.add_argument('--max-new-tokens', type=int, default=0,
                    help='0 = 8 tokens per audio second + 64 (min 128)')


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

PIP_LINE = 'pip install -U "nemo_toolkit[asr,speechlm2]"'


def check_salm_deps() -> bool:
    """Fast-fail before any download: nemo_toolkit 3.0.0's asr extra lacks peft,
    and speechlm2/models/salm.py imports it unguarded at module level."""
    try:
        import peft  # noqa: F401  nemo_toolkit[speechlm2] extra
    except ImportError:
        print('peft is not installed: nemo.collections.speechlm2 (SALM) imports it at module '
              'level and the nemo_toolkit[asr] extra does not include it. Install the '
              f'speechlm2 extra into this venv:\n    {PIP_LINE}', file=sys.stderr, flush=True)
        return False
    return True


def release_cuda() -> None:
    """After a CUDA error on one chunk, hand the cached blocks back so the next
    file gets a clean allocator instead of inheriting the OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  best effort, never mask the original error
        pass


def load_model(model_id: str, device: str, dtype: str):
    import torch
    # Loading verified on https://huggingface.co/nvidia/canary-qwen-2.5b (Usage):
    #   from nemo.collections.speechlm2.models import SALM
    #   model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    from nemo.collections.speechlm2.models import SALM
    logging.getLogger('nemo_logger').setLevel(logging.WARNING)   # best effort
    t0 = time.time()
    model = SALM.from_pretrained(model_id)
    # dtype / device / eval order as in examples/speechlm2/salm_generate.py
    # (https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/speechlm2/salm_generate.py):
    #   model = model.to(getattr(torch, cfg.dtype)).to(cfg.device); model = model.eval()
    model = model.to(getattr(torch, dtype)).to(device).eval()
    print(f'loaded {model_id} on {device} ({dtype}) in {time.time() - t0:.1f}s', flush=True)
    return model


def transcribe_chunk(model, audio: np.ndarray, sr: int, prompt: str, max_new_tokens: int) -> str:
    """One generate() call on one 16 kHz float32 mono array; returns the text."""
    import soundfile as sf
    import torch
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / 'chunk.wav'
        sf.write(str(wav), audio, sr, subtype='PCM_16')      # 16-bit PCM: universally readable by lhotse/libsndfile
        # Prompt structure and kwargs verified on the model card (Usage, ASR mode):
        #   model.generate(prompts=[[{"role": "user",
        #        "content": f"Transcribe the following: {model.audio_locator_tag}",
        #        "audio": ["speech.wav"]}]], max_new_tokens=128)
        with torch.inference_mode():
            answer_ids = model.generate(
                prompts=[[{'role': 'user',
                           'content': f'{prompt} {model.audio_locator_tag}',
                           'audio': [str(wav)]}]],
                max_new_tokens=max_new_tokens,
            )
    ans = answer_ids[0].cpu()
    # examples/speechlm2/salm_generate.py parse_hyp(): truncate at the first EOS
    # (eos_tokens = [model.text_eos_id]) before decoding.
    eos = getattr(model, 'text_eos_id', None)
    if eos is not None:
        hit = (ans == int(eos)).nonzero(as_tuple=True)[0]
        if hit.numel():
            ans = ans[:int(hit[0])]
    # card: print(model.tokenizer.ids_to_text(answer_ids[0].cpu()))
    return str(model.tokenizer.ids_to_text(ans)).strip()


# --------------------------------------------------------------------------- #
# Audio helpers
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


def auto_max_new_tokens(seconds: float, override: int) -> int:
    return override if override > 0 else max(128, int(seconds * 8) + 64)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = common.standard_args(DEFAULT_TAG, extra_args)
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

    if not check_salm_deps():
        return 2
    model = load_model(args.model, args.device, args.dtype)
    total_audio = total_dt = 0.0
    failed: List[str] = []
    for path in todo:
        out = common.out_path(args, path)
        try:
            audio, sr = common.load_audio(path)
            duration = len(audio) / sr
            pieces = split_audio(audio, sr, args.chunk_s, args.search_s)
            segments: List[dict] = []
            dt = 0.0
            for start, piece in pieces:
                p_start, p_end = start / sr, (start + len(piece)) / sr
                t0 = time.time()
                text = transcribe_chunk(model, piece, sr, args.prompt,
                                        auto_max_new_tokens(len(piece) / sr, args.max_new_tokens))
                dt += time.time() - t0
                if not text:
                    print(f'  warning: {path.name} @ {p_start:.1f}-{p_end:.1f}s decoded empty',
                          flush=True)
                segments.append({'start': p_start, 'end': p_end, 'text': text, 'words': []})
            if len(pieces) == 1:                   # README: text-only model, one segment 0..duration
                segments[0]['start'], segments[0]['end'] = 0.0, duration
            full_text = ' '.join(s['text'] for s in segments if s['text']).strip()
            extra = {'hf_model': args.model, 'text': full_text, 'chunk_s': args.chunk_s}
            common.write_transcript(out, path, args.tag, duration, dt, segments,
                                    word_timestamps=False, extra=extra)
        except Exception as e:  # noqa: BLE001  one bad file must not abort the tag run
            failed.append(path.name)
            print(f'FAIL {path.name}: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
            traceback.print_exc()
            if 'cuda' in str(e).lower():
                release_cuda()
            continue
        total_audio += duration; total_dt += dt
        print(f'{path.name}: {duration:6.1f}s audio, {dt:5.1f}s, {len(pieces)} chunks, '
              f'{len(full_text.split())} words', flush=True)
    n_done = len(todo) - len(failed)
    print(f'DONE {n_done}/{len(todo)} files, tag {args.tag}, '
          f'RTF {total_dt / max(total_audio, 1e-9):.3f}, '
          f'mean {total_dt / max(1, n_done):.1f}s per conversation'
          + (f', FAILED {len(failed)}: {failed}' if failed else ''), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
