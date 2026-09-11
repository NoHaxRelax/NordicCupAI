"""ASR baseline over PriMock57.

Two conditions, and the gap between them is the interesting part:

    mixed      doctor and patient summed to one channel. This is what a real
               consultation recording looks like, overlapping speech included,
               and it is what the competition will almost certainly hand us.
    per-channel  each speaker transcribed from their own clean track. Perfect
               diarization, no cross-talk. Not a realistic condition; it is the
               ceiling that tells us how much the mixing itself costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .dataset import Consultation
from .metrics import corpus_wer, error_profile

_MODEL_CACHE: dict[tuple, object] = {}


def _register_cuda_dlls() -> None:
    """On Windows, CTranslate2 needs cuBLAS and cuDNN on the DLL search path.

    The pip wheels install them under site-packages/nvidia/*/bin, which is not
    on PATH, so faster-whisper fails at the first encode with "cublas64_12.dll
    is not found". Registering the directories here means nobody has to edit
    their PATH before the competition.
    """
    import os
    import site

    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    found = []
    for sp in roots:
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cublas/lib", "nvidia/cudnn/lib"):
            p = os.path.join(sp, *sub.split("/"))
            if os.path.isdir(p):
                found.append(p)
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(p)
                    except (OSError, FileNotFoundError):
                        pass
    # add_dll_directory alone is not enough for CTranslate2's loader; it also
    # resolves through PATH, so both are set.
    if found:
        os.environ["PATH"] = os.pathsep.join(found) + os.pathsep + os.environ.get("PATH", "")


def get_model(size: str = "large-v3", device: str = "auto", compute_type: str | None = None):
    """Load (and cache) a faster-whisper model.

    int8_float16 on GPU keeps large-v3 near 3 GB, which matters on an 8 GB
    laptop card; on CPU int8 is the only sane choice.
    """
    from faster_whisper import WhisperModel
    import ctranslate2

    _register_cuda_dlls()
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"
    key = (size, device, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def mix_channels(c: Consultation, out_dir: Path) -> Path | None:
    """Sum the two speaker tracks into one mono file, as a real recording would be."""
    if len(c.audio) < 2:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{c.cid}_mixed.wav"
    if out.is_file() and out.stat().st_size > 4096:
        return out

    tracks, sr = [], None
    for p in (c.audio.get("Doctor"), c.audio.get("Patient")):
        x, s = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        sr = s if sr is None else sr
        if s != sr:
            raise ValueError(f"sample-rate mismatch in {c.cid}")
        tracks.append(x)

    n = max(len(t) for t in tracks)
    mix = np.zeros(n, dtype=np.float32)
    for t in tracks:
        mix[: len(t)] += t
    peak = float(np.abs(mix).max())
    if peak > 1.0:                      # summing two full-scale tracks clips
        mix /= peak
    sf.write(out, mix, sr)
    return out


@dataclass
class Result:
    cid: str
    hypothesis: str
    audio_s: float
    decode_s: float
    segments: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: seconds of audio per second of compute. Higher is faster."""
        return self.audio_s / max(self.decode_s, 1e-9)


def transcribe(path: Path, model, *, language: str = "en", beam_size: int = 5,
               vad: bool = True, hotwords: str | None = None,
               initial_prompt: str | None = None) -> Result:
    t0 = time.perf_counter()
    segs, info = model.transcribe(
        str(path), language=language, beam_size=beam_size,
        vad_filter=vad, condition_on_previous_text=False,
        hotwords=hotwords, initial_prompt=initial_prompt,
    )
    segs = list(segs)                    # faster-whisper is lazy; this forces decode
    dt = time.perf_counter() - t0
    return Result(
        cid=path.stem, hypothesis=" ".join(s.text.strip() for s in segs),
        audio_s=float(info.duration), decode_s=dt, segments=len(segs),
        extra={"segments": [(float(s.start), float(s.end), s.text.strip()) for s in segs]},
    )


def evaluate(consultations: list[Consultation], model, work_dir: Path,
             *, condition: str = "mixed", **kw) -> dict:
    """Transcribe and score a set of consultations. Returns the corpus WER, which
    is total edits over total reference words, not a mean of per-file WERs."""
    pairs, results = [], []
    for c in consultations:
        if condition == "mixed":
            p = mix_channels(c, work_dir)
            if p is None:
                continue
            r = transcribe(p, model, **kw)
            ref = c.reference()
        else:
            # Transcribe each speaker's clean track, then rebuild ONE time-ordered
            # transcript by interleaving segments on their timestamps. Simply
            # concatenating "all doctor, then all patient" scores ~83% WER against
            # a time-ordered reference purely because the word order is wrong,
            # which measures the concatenation, not the model.
            segs, audio_s, dec_s = [], 0.0, 0.0
            for spk, p in sorted(c.audio.items()):
                rr = transcribe(p, model, **kw)
                segs.extend((s, e, spk, txt) for s, e, txt in rr.extra["segments"])
                audio_s += rr.audio_s
                dec_s += rr.decode_s
            segs.sort(key=lambda x: x[0])
            r = Result(c.cid, " ".join(t for _, _, _, t in segs), audio_s, dec_s,
                       segments=len(segs), extra={"diarized": segs})
            ref = c.reference()
        r.cid = c.cid
        results.append(r)
        pairs.append((ref, r.hypothesis))

    score, counts = corpus_wer(pairs)
    # The same systems scored under the other convention: human transcribers
    # record every "yeah"/"ok"; Whisper drops many as non-speech. Reporting both
    # keeps us honest about which number a given rule would produce.
    score_nb, counts_nb = corpus_wer(pairs, drop_backchannels=True)
    audio_s = sum(r.audio_s for r in results)
    dec_s = sum(r.decode_s for r in results)
    return {
        "condition": condition,
        "consultations": len(results),
        "wer": round(score, 4),
        "wer_no_backchannel": round(score_nb, 4),
        "sub": counts.subs, "del": counts.dels, "ins": counts.ins,
        "ref_words": counts.ref_len,
        "audio_h": round(audio_s / 3600, 3),
        "decode_s": round(dec_s, 1),
        "rtf": round(audio_s / max(dec_s, 1e-9), 1),
        "pairs": pairs,
        "results": results,
        "worst_words": error_profile(pairs, top=20),
    }
