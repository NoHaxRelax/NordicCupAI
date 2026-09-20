"""Transcribe every supplied conversation once with word timestamps and cache
the result under transcripts/ (gitignored), so span analysis and the answering
half can be iterated without re-running ASR.

    python transcribe_cache.py                 # faster-whisper large-v3, fp16
    python transcribe_cache.py --model large-v3-turbo
"""
import argparse, json, os, site, sys, time
from pathlib import Path

# Windows: the CUDA runtime DLLs live in pip's nvidia/* packages, not on PATH.
_nv = Path(site.getsitepackages()[0]) / 'Lib' / 'site-packages' / 'nvidia'
for d in ('cublas', 'cudnn', 'cuda_nvrtc'):
    p = _nv / d / 'bin'
    if p.is_dir():
        os.add_dll_directory(str(p))
        os.environ['PATH'] = str(p) + os.pathsep + os.environ['PATH']
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from faster_whisper import WhisperModel  # noqa: E402

AUDIO = Path(__file__).parent / 'data' / 'audio'
OUT = Path(__file__).parent / 'transcripts'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='large-v3')
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    model = WhisperModel(a.model, device='cuda', compute_type='float16')
    print(f'loaded {a.model} in {time.time()-t0:.1f}s', flush=True)
    files = sorted(AUDIO.glob('*.mp3'), key=lambda p: int(p.stem.split('_')[-1]))
    total_audio = total_dt = 0.0
    for f in files:
        out = OUT / f'{f.stem}.{a.model}.json'
        if out.exists() and not a.force:
            print(f'skip {f.name}', flush=True); continue
        t0 = time.time()
        segs, info = model.transcribe(str(f), language='en', word_timestamps=True,
                                      beam_size=a.beam, vad_filter=False)
        segs = list(segs)
        dt = time.time() - t0
        total_audio += info.duration; total_dt += dt
        data = {
            'file': f.name, 'model': a.model, 'duration': info.duration, 'seconds': dt,
            'segments': [
                {'start': s.start, 'end': s.end, 'text': s.text,
                 'words': [{'w': w.word, 'start': w.start, 'end': w.end, 'p': w.probability}
                           for w in (s.words or [])]}
                for s in segs],
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'{f.name}: {info.duration:6.1f}s audio, {dt:5.1f}s, {len(segs)} segs', flush=True)
    if total_audio:
        print(f'DONE {len(files)} files, RTF {total_dt/total_audio:.3f}, '
              f'mean {total_dt/len(files):.1f}s per conversation', flush=True)


if __name__ == '__main__':
    sys.exit(main())
