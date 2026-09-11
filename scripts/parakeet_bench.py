"""Benchmark NVIDIA Parakeet against the Whisper baseline, same audio, same scorer.

Run it exactly like scripts/baseline.py so the WER column is comparable:

    python scripts/parakeet_bench.py --n 6

Two things about this script are load-bearing.

*The weights ship as bfloat16* and the audio features arrive as float32, which
makes the first convolution raise "Input type (float) and bias type (struct
c10::BFloat16) should be the same". Loading in float32 is the fix; on CPU it is
also the faster dtype, since CPU bf16 conv falls back to an emulated path.

*Parakeet has no long-form mode.* Whisper is trained on 30 s windows and
faster-whisper stitches them for you; Parakeet is a plain encoder + CTC head
with 5000 positions, so a ten-minute consultation must be cut up by hand. Cuts
land on the quietest point near each boundary rather than on a fixed grid,
because slicing mid-word costs an error on both sides of the seam.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + mkl both ship libiomp5md

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.asr import mix_channels  # noqa: E402
from medvoice.dataset import load, split  # noqa: E402
from medvoice.metrics import corpus_wer, error_profile  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "parakeet-ctc-0.6b"


def split_points(x: np.ndarray, sr: int, target_s: float = 25.0,
                 search_s: float = 3.0, win_s: float = 0.4) -> list[tuple[int, int]]:
    """Cut a long recording into chunks, each break at the quietest moment near
    the target length. Returns (start, end) sample indices."""
    target, search, win = int(target_s * sr), int(search_s * sr), int(win_s * sr)
    if len(x) <= target + search:
        return [(0, len(x))]

    # RMS on a win_s grid, so the quiet search is a cheap lookup rather than a scan
    hop = win // 4
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    spans, start = [], 0
    while len(x) - start > target + search:
        lo = max(0, (start + target - search) // hop)
        hi = min(len(rms) - 1, (start + target + search) // hop)
        quietest = lo + int(np.argmin(rms[lo:hi + 1])) if hi > lo else lo
        cut = min(len(x), quietest * hop + win // 2)
        if cut <= start:                       # pathological; fall back to the grid
            cut = start + target
        spans.append((start, cut))
        start = cut
    spans.append((start, len(x)))
    return spans


def load_model(path: Path, device: str = "cpu"):
    import torch
    from transformers import (ParakeetFeatureExtractor, ParakeetForCTC,
                              ParakeetProcessor, ParakeetTokenizer)
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()      # 974 per-parameter bar updates, one line each

    # The repo has no processor_config.json, so ParakeetProcessor.from_pretrained
    # cannot find its class; assembling it from the two halves works regardless.
    proc = ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor.from_pretrained(path),
        tokenizer=ParakeetTokenizer.from_pretrained(path),
    )
    model = ParakeetForCTC.from_pretrained(path, dtype=torch.float32).to(device).eval()
    return model, proc


def ctc_decode(ids: np.ndarray, tokenizer, blank: int) -> str:
    """Greedy CTC: collapse runs, drop blanks, then detokenize."""
    keep = ids[np.insert(ids[1:] != ids[:-1], 0, True)]
    keep = keep[keep != blank]
    if keep.size == 0:
        return ""
    return tokenizer.decode(keep.tolist(), skip_special_tokens=True).strip()


def transcribe(path: Path, model, proc, *, target_s: float = 25.0,
               device: str = "cpu") -> tuple[str, float, float, int]:
    import torch

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path.name} is {sr} Hz; Parakeet's feature extractor wants 16 kHz")

    blank = model.config.pad_token_id
    spans = split_points(x, sr, target_s=target_s)
    parts, t0 = [], time.perf_counter()
    for a, b in spans:
        chunk = x[a:b]
        if len(chunk) < sr // 10:              # sub-100 ms tail, nothing to decode
            continue
        inputs = proc(chunk, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        ids = logits[0].argmax(dim=-1).cpu().numpy()
        text = ctc_decode(ids, proc.tokenizer, blank)
        if text:
            parts.append(text)
    dt = time.perf_counter() - t0
    return " ".join(parts), len(x) / sr, dt, len(spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="consultations to score (6 matches the Whisper table)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=float, default=25.0)
    ap.add_argument("--out", default="runs/parakeet_bench.json")
    args = ap.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cons = load()
    _, dev = split(cons)
    dev = [c for c in dev if len(c.audio) == 2][: args.n]
    print(f"dev: {len(dev)} consultations, {sum(c.duration for c in dev)/3600:.2f} h, "
          f"{sum(len(c.reference().split()) for c in dev):,} reference words")
    print(f"model: {Path(args.model).name}  device: {device}  chunk: {args.chunk:.0f}s\n")

    t0 = time.perf_counter()
    model, proc = load_model(Path(args.model), device)
    print(f"loaded in {time.perf_counter()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)\n")

    pairs, audio_s, dec_s, per_file = [], 0.0, 0.0, []
    for c in dev:
        mixed = mix_channels(c, Path("work"))
        hyp, a, d, n_chunks = transcribe(mixed, model, proc, target_s=args.chunk, device=device)
        ref = c.reference()
        pairs.append((ref, hyp))
        audio_s += a
        dec_s += d
        w, _ = corpus_wer([(ref, hyp)])
        per_file.append({"cid": c.cid, "wer": round(w, 4), "chunks": n_chunks,
                         "audio_s": round(a, 1), "decode_s": round(d, 1)})
        print(f"  {c.cid:<26} wer {w:6.3f}  {n_chunks:>2} chunks  "
              f"{a/60:5.1f} min audio in {d/60:5.1f} min  ({a/max(d,1e-9):.2f}x)")

    score, counts = corpus_wer(pairs)
    score_nb, _ = corpus_wer(pairs, drop_backchannels=True)
    row = {
        "model": Path(args.model).name, "condition": "mixed", "device": device,
        "chunk_s": args.chunk, "consultations": len(pairs),
        "wer": round(score, 4), "wer_no_backchannel": round(score_nb, 4),
        "sub": counts.subs, "del": counts.dels, "ins": counts.ins,
        "ref_words": counts.ref_len, "audio_h": round(audio_s / 3600, 3),
        "decode_s": round(dec_s, 1), "rtf": round(audio_s / max(dec_s, 1e-9), 1),
        "worst_words": error_profile(pairs, top=20), "per_file": per_file,
    }
    print(f"\n{'CORPUS':<26} wer {score:6.3f}   no-backchannel {score_nb:6.3f}")
    print(f"{'':<26} sub {counts.subs}  del {counts.dels}  ins {counts.ins}  "
          f"of {counts.ref_len} words")
    print(f"{'':<26} rtf {row['rtf']}x on {device}")
    print("most-missed: " + ", ".join(f"{w}({n})" for w, n in row["worst_words"][:12]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
