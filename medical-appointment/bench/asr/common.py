"""Shared plumbing for the ASR runners. See bench/README.md for the contract.

Every runner:

    args = standard_args('my-tag')          # --audio-dir --out-dir --tag --limit --force
    for path in audio_files(args):
        out = out_path(args, path)
        if out.exists() and not args.force: continue
        audio, sr = load_audio(path)         # float32 mono 16 kHz
        t0 = time.time(); ...run model...; dt = time.time() - t0
        write_transcript(out, path, args.tag, duration, dt, segments)

Keep model loading outside the per-file timer. Never write anything under data/.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent          # bench/asr
CASE = HERE.parent.parent                       # medical-appointment
DEFAULT_AUDIO = CASE / 'data' / 'audio'
DEFAULT_OUT = CASE / 'transcripts'
SR = 16000

_TERMINAL = re.compile(r'[.!?]["\')\]]*$')


def standard_args(default_tag: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-dir', default=str(DEFAULT_AUDIO))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--tag', default=default_tag, help='transcript tag (bench/README.md)')
    ap.add_argument('--limit', type=int, default=0, help='only the first N files')
    ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
    ap.add_argument('--device', default='cuda')
    if extra:
        extra(ap)
    return ap.parse_args()


def audio_files(args) -> List[Path]:
    files = sorted(Path(args.audio_dir).glob('*.mp3'),
                   key=lambda p: int(re.sub(r'\D', '', p.stem) or 0))
    return files[:args.limit] if args.limit else files


def out_path(args, audio_path: Path) -> Path:
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return Path(args.out_dir) / f'{audio_path.stem}.{args.tag}.json'


def load_audio(path: Path, sr: int = SR) -> tuple[np.ndarray, int]:
    """Decode to float32 mono at `sr`. Tries soundfile (libsndfile >= 1.1 reads
    MP3), then librosa/audioread, then an ffmpeg subprocess."""
    try:
        import soundfile as sf
        data, native = sf.read(str(path), dtype='float32', always_2d=True)
        data = data.mean(axis=1)
        if native != sr:
            import librosa
            data = librosa.resample(data, orig_sr=native, target_sr=sr)
        return data.astype(np.float32), sr
    except Exception:
        pass
    try:
        import librosa
        data, _ = librosa.load(str(path), sr=sr, mono=True)
        return data.astype(np.float32), sr
    except Exception:
        pass
    import subprocess
    raw = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy(), sr


def words_to_segments(words: List[dict]) -> List[dict]:
    """Group a flat word list ({w,start,end[,p]}) into sentence segments at
    terminal punctuation. Used by models that return words but no segments."""
    segments, cur = [], []
    for w in words:
        cur.append(w)
        if _TERMINAL.search(w['w'].strip()):
            segments.append(_segment(cur)); cur = []
    if cur:
        segments.append(_segment(cur))
    return segments


def _segment(ws: List[dict]) -> dict:
    return {'start': ws[0]['start'], 'end': ws[-1]['end'],
            'text': ''.join(w['w'] for w in ws), 'words': list(ws)}


def write_transcript(out: Path, audio_path: Path, tag: str, duration: float,
                     seconds: float, segments: List[dict],
                     word_timestamps: bool = True, extra: Optional[dict] = None) -> None:
    data = {
        'file': audio_path.name, 'model': tag, 'duration': float(duration),
        'seconds': float(seconds), 'word_timestamps': word_timestamps,
        'segments': segments,
    }
    if extra:
        data.update(extra)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


def read_transcript(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def flat_words(data: dict) -> List[dict]:
    return [w for s in data['segments'] for w in s.get('words', [])]


class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
