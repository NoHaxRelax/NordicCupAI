"""Baseline sweep: which Whisper size, at what accuracy, at what speed.

Produces the numbers that decide the stack. Two conditions per model so we can
separate "the model is weak" from "mixing two speakers into one channel is
hard".

    python scripts/baseline.py --n 6 --models tiny.en,base.en,small.en,distil-large-v3,large-v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.asr import evaluate, get_model  # noqa: E402
from medvoice.dataset import load, split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="consultations to score")
    ap.add_argument("--models", default="tiny.en,base.en,small.en,distil-large-v3,large-v3")
    ap.add_argument("--conditions", default="mixed,per-channel")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/asr_baseline.json")
    args = ap.parse_args()

    cons = load()
    _, dev = split(cons)
    dev = [c for c in dev if len(c.audio) == 2][: args.n]
    audio_h = sum(c.duration for c in dev) / 3600
    print(f"dev: {len(dev)} consultations, {audio_h:.2f} h audio, "
          f"{sum(len(c.reference().split()) for c in dev):,} reference words\n")

    rows = []
    print(f"{'model':<18} {'condition':<12} {'WER':>7} {'WER-nbc':>8} {'sub':>6} {'del':>6} {'ins':>6} {'RTF':>7}")
    print("-" * 84)
    for size in args.models.split(","):
        try:
            t0 = time.perf_counter()
            model = get_model(size, device=args.device)
            load_s = time.perf_counter() - t0
        except Exception as exc:
            print(f"{size:<18} load FAILED: {str(exc)[:50]}")
            continue
        for cond in args.conditions.split(","):
            try:
                r = evaluate(dev, model, Path("work"), condition=cond)
            except Exception as exc:
                print(f"{size:<18} {cond:<12} FAILED: {str(exc)[:44]}")
                continue
            print(f"{size:<18} {cond:<12} {r['wer']:>7.3f} {r['wer_no_backchannel']:>8.3f} "
                  f"{r['sub']:>6} {r['del']:>6} {r['ins']:>6} {r['rtf']:>6.1f}x")
            rows.append({k: v for k, v in r.items() if k not in ("pairs", "results")}
                        | {"model": size, "load_s": round(load_s, 1)})
            if cond == "mixed":
                worst = ", ".join(f"{w}({n})" for w, n in r["worst_words"][:10])
                print(f"{'':<18} {'':<12} most-missed: {worst}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
