"""Does the meaning survive 13.9% WER?

This is the measurement that decides how the week is spent, and it is the one
question WER cannot answer. WER weights every token identically, so dropping two
hundred "yeah"s scores the same as losing every drug name. On our large-v3 run,
twenty function-word and backchannel types carry 49.6% of the combined
substitution-and-deletion mass, which means the headline 13.9% might be almost
entirely made of noise nobody is grading.

So: transcribe, then ask a different question. Of the words that carry clinical
meaning, how many make it from the reference into the hypothesis?

If content recall is high, the ASR workstream is close to finished and every
remaining hour belongs to the extraction half. If it is low, the opposite. The
two answers imply completely different four days, which is why this runs first.

It also saves the hypotheses, which the original baseline sweep threw away.

    python scripts/term_recall.py --n 6 --model large-v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.asr import evaluate, get_model  # noqa: E402
from medvoice.dataset import load, split  # noqa: E402
from medvoice.metrics import corpus_wer, normalise, term_recall  # noqa: E402
from medvoice.structured import canon, load_labels  # noqa: E402

TERMS = Path(__file__).resolve().parents[1] / "labels" / "content_terms.json"


def load_terms() -> dict[str, set[str]]:
    d = json.loads(TERMS.read_text(encoding="utf-8"))
    return {k: {w.lower() for w in v} for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hotwords", action="store_true",
                    help="bias decoding with the same lexicon we are measuring")
    ap.add_argument("--terms", type=int, default=0,
                    help="cap the hotword list at N terms. 0 means all of them, which is "
                         "measured to be catastrophic: 94 terms is 928 characters, Whisper's "
                         "prompt window is 224 tokens, and the overflow costs 39 WER points "
                         "and a third of the transcript")
    ap.add_argument("--out", default="runs/term_recall.json")
    args = ap.parse_args()

    terms = load_terms()
    every = set().union(*terms.values())
    labels = load_labels()

    cons = load()
    _, dev = split(cons)
    dev = [c for c in dev if len(c.audio) == 2][: args.n]
    print(f"{len(dev)} consultations, model {args.model}"
          f"{', hotword-biased' if args.hotwords else ''}\n")

    model = get_model(args.model, device=args.device)
    kw = {}
    if args.hotwords:
        pool = sorted(terms["diagnoses"] | terms["drugs"])
        if args.terms:
            # Longest terms first: the rare polysyllabic words are the ones a
            # general model is most likely to mangle, and the ones worth the
            # scarce prompt budget. "chest" needs no help; "nitrofurantoin" might.
            pool = sorted(pool, key=lambda w: -len(w))[: args.terms]
        kw["hotwords"] = " ".join(sorted(pool))
        print(f"  biasing with {len(pool)} terms, {len(kw['hotwords'])} chars\n")
    r = evaluate(dev, model, Path("work"), condition="mixed", **kw)
    pairs = r["pairs"]

    print(f"corpus WER {r['wer']:.4f}   no-backchannel {r['wer_no_backchannel']:.4f}")
    print(f"  sub {r['sub']}  del {r['del']}  ins {r['ins']}  of {r['ref_words']} words\n")

    print("CONTENT-TERM RECALL  (of the terms present in the reference, how many survive)")
    print(f"  {'category':<16}{'found':>8}{'present':>9}{'recall':>9}")
    print("  " + "-" * 42)
    out = {"model": args.model, "hotwords": args.hotwords, "wer": r["wer"],
           "wer_no_backchannel": r["wer_no_backchannel"], "categories": {}}
    for name in ("diagnoses", "drugs", "investigations", "red_flags", "temporal"):
        f = t = 0
        for ref, hyp in pairs:
            a, b = term_recall(ref, hyp, terms[name])
            f += a
            t += b
        rec = f / t if t else float("nan")
        out["categories"][name] = {"found": f, "present": t, "recall": round(rec, 4)}
        print(f"  {name:<16}{f:>8}{t:>9}{rec:>9.3f}")
    f = t = 0
    for ref, hyp in pairs:
        a, b = term_recall(ref, hyp, every)
        f += a
        t += b
    out["overall"] = {"found": f, "present": t, "recall": round(f / max(t, 1), 4)}
    print("  " + "-" * 42)
    print(f"  {'ALL CONTENT':<16}{f:>8}{t:>9}{f/max(t,1):>9.3f}")

    # The sharper question: the diagnosis is one word in a 1500-word transcript.
    # Corpus recall over all content terms can look fine while the single word
    # that decides the answer is gone.
    print("\nTHE DECIDING WORD  (did the gold diagnosis survive, per consultation)")
    kept = miss = 0
    rows = []
    for c, (ref, hyp) in zip(dev, pairs):
        lab = labels.get(c.cid)
        if not lab:
            continue
        head = canon(lab.diagnoses[0]).split()[-1]
        h = set(normalise(hyp))
        rr = set(normalise(ref))
        in_ref = head in rr
        in_hyp = head in h
        rows.append({"cid": c.cid, "dx": lab.diagnoses[0], "head": head,
                     "in_reference": in_ref, "in_hypothesis": in_hyp})
        mark = "kept " if in_hyp else ("LOST " if in_ref else "n/a  ")
        if in_ref:
            kept += in_hyp
            miss += not in_hyp
        print(f"  {mark} {c.cid:<24} {lab.diagnoses[0]:<26} head={head!r}")
    out["deciding_word"] = {"kept": kept, "lost": miss, "rows": rows}
    print(f"\n  of the consultations where the diagnosis word is actually spoken: "
          f"{kept} kept, {miss} lost")
    print("  ('n/a' means the clinician never said the diagnosis out loud, which is its")
    print("   own finding: no ASR quality recovers a word that was never uttered.)")

    out["hypotheses"] = [{"cid": c.cid, "hypothesis": h} for c, (_, h) in zip(dev, pairs)]
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {o}  (hypotheses included, the baseline sweep discarded them)")


if __name__ == "__main__":
    main()
