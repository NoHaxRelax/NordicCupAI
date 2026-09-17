"""Cut every gold evidence interval out of the audio and write it as a small MP3
data URI, so the timeline page can play the annotated stretch anywhere (the
published page cannot reach the local audio server).

Training gold: data/question_train.csv. Validation gold: bench/mine/span_state.json
(the recovered spans). Output: bench/ref/clips.json, {"<stem>:<q>": "data:audio/mpeg;base64,..."}.
Decoding and encoding go through PyAV (bundled FFmpeg), no ffmpeg binary needed.

    python bench/clips.py                # 16 kHz mono MP3 at 32 kbit/s, about 4 KB per second
"""
from __future__ import annotations

import base64
import csv
import io
import json
import re
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
OUT = HERE / 'ref' / 'clips.json'
RATE = 16000
BITRATE = 32000


def decode(path: Path) -> np.ndarray:
    c = av.open(str(path))
    rs = av.AudioResampler(format='s16', layout='mono', rate=RATE)
    chunks = []
    for fr in c.decode(c.streams.audio[0]):
        for o in rs.resample(fr):
            chunks.append(o.to_ndarray())
    for o in rs.resample(None):
        chunks.append(o.to_ndarray())
    c.close()
    return np.concatenate(chunks, axis=1)[0]


def encode(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    out = av.open(buf, 'w', format='mp3')
    st = out.add_stream('libmp3lame', rate=RATE)
    st.bit_rate = BITRATE
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1).astype(np.int16), format='s16p', layout='mono')
    frame.sample_rate = RATE
    frame.pts = 0
    for pkt in st.encode(frame):
        out.mux(pkt)
    for pkt in st.encode(None):
        out.mux(pkt)
    out.close()
    return buf.getvalue()


def intervals() -> dict[str, list[tuple[str, float, float]]]:
    """audio path -> [(key, start, end)]"""
    work: dict[str, list[tuple[str, float, float]]] = {}
    order: dict[str, list[str]] = {}
    for r in csv.DictReader(open(CASE / 'data' / 'question_train.csv', encoding='utf-8')):
        order.setdefault(r['transcript_id'], []).append(r['question_id'])
        if r['answer'] == 'yes' and r['evidence_start']:
            stem = f"conversation_{r['transcript_id']}"
            key = f"{stem}:{order[r['transcript_id']].index(r['question_id']) + 1}"
            work.setdefault(str(CASE / 'data' / 'audio' / f'{stem}.mp3'), []).append((key, float(r['evidence_start']), float(r['evidence_end'])))
    sp = HERE / 'mine' / 'span_state.json'
    if sp.exists():
        for key, q in json.loads(sp.read_text(encoding='utf-8')).items():
            if q.get('stage') == 'done':
                stem = key.split(':')[0]
                work.setdefault(str(CASE / 'request_dump' / f'{stem}.mp3'), []).append((key, q['g'], q['h']))
    return work


def main() -> int:
    clips = {}
    total = 0
    for path, items in sorted(intervals().items(), key=lambda kv: int(re.sub(r'\D', '', Path(kv[0]).stem))):
        if not Path(path).exists():
            print('missing audio', path); continue
        pcm = decode(Path(path))
        for key, s, e in items:
            a, b = int(s * RATE), min(len(pcm), int(e * RATE))
            if b - a < RATE // 20:
                continue
            data = encode(pcm[a:b])
            total += len(data)
            clips[key] = 'data:audio/mpeg;base64,' + base64.b64encode(data).decode('ascii')
        print(f'{Path(path).stem}: {len(items)} clips', flush=True)
    OUT.write_text(json.dumps(clips), encoding='utf-8')
    print(f'{len(clips)} clips, {total // 1024} KB of MP3, {OUT.stat().st_size // 1024} KB as JSON -> {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
