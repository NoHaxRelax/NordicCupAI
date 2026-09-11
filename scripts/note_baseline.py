"""Trivial note baselines, and the control that tells us what ROUGE is measuring.

Before writing a note generator it is worth knowing what score you get for
doing nothing. Five baselines, none of which involves a model:

    complaint     the patient's one-line presenting complaint
    lead-N        the first N words of the transcript
    doctor-lead   the first N words the doctor says
    keywords      content words from the transcript, ranked by how note-like
                  they are, in transcript order
    WRONG NOTE    a real clinical note about a different patient entirely

The last one is the control. It is fluent, correctly formatted, uses all the
right shorthand, and is about someone else's illness. Whatever it scores is
the part of the metric that comes from writing in the house style rather than
from listening to this consultation. Anything a real system earns above that
line is the only part that reflects understanding.

    python scripts/note_baseline.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.dataset import load  # noqa: E402
from medvoice.notes import load_notes, score, toks  # noqa: E402

STOP = set("""a an the and or but if of to in on at for with as is are was were be been being
i you he she it we they me him her us them my your his its our their this that these those
so do does did done have has had not no yes ok okay yeah right just like well very really
about from up down out over under then than there here what which who when where how why
can could will would shall should may might must am s t re ve ll d""".split())


def lead(words: list[str], n: int) -> str:
    return " ".join(words[:n])


def keyword_note(text: str, n: int) -> str:
    """Content words in transcript order, one pass, capped at n.

    Not a summary in any meaningful sense. It exists to show how much of a
    ROUGE score is available from term overlap alone, with no structure,
    ordering or clinical reasoning."""
    seen: set[str] = set()
    out = []
    for w in toks(text):
        if w in STOP or len(w) < 3 or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=131, help="median human note length")
    ap.add_argument("--out", default="runs/note_baselines.json")
    args = ap.parse_args()

    notes = load_notes()
    cons = {c.cid: c for c in load()}
    cids = [c for c in sorted(notes) if c in cons and notes[c].refs]
    print(f"{len(cids)} consultations with both a transcript and human notes\n")

    systems: dict[str, dict[str, str]] = {
        "empty": {},
        "complaint": {},
        "lead-N": {},
        "doctor-lead": {},
        "keywords": {},
        "WRONG NOTE": {},
    }
    for i, cid in enumerate(cids):
        c, ns = cons[cid], notes[cid]
        full = c.reference()
        doc = c.by_speaker().get("Doctor", "")
        systems["empty"][cid] = ""
        systems["complaint"][cid] = ns.presenting_complaint
        systems["lead-N"][cid] = lead(full.split(), args.words)
        systems["doctor-lead"][cid] = lead(doc.split(), args.words)
        systems["keywords"][cid] = keyword_note(full, args.words)
        # deterministic: the next consultation round-robin, so every note is a
        # real note and no consultation is ever paired with itself
        systems["WRONG NOTE"][cid] = notes[cids[(i + 1) % len(cids)]].doctor

    print(f"{'baseline':<14}{'vs doctor':>12}{'best of 5':>12}{'mean of 6':>12}{'words':>8}")
    print("-" * 58)
    out = {}
    for name, hyps in systems.items():
        d, b, m, w = [], [], [], []
        for cid in cids:
            h, ns = hyps[cid], notes[cid]
            if ns.doctor:
                d.append(score([ns.doctor], h)["rougeL_max"])
            if ns.evaluators:
                b.append(score(list(ns.evaluators.values()), h)["rougeL_max"])
            m.append(score(ns.refs, h)["rougeL_mean"])
            w.append(len(h.split()))
        row = {"vs_doctor": round(statistics.mean(d), 4),
               "best_of_5": round(statistics.mean(b), 4),
               "mean_of_6": round(statistics.mean(m), 4),
               "words": round(statistics.mean(w))}
        out[name] = row
        print(f"{name:<14}{row['vs_doctor']:>12.3f}{row['best_of_5']:>12.3f}"
              f"{row['mean_of_6']:>12.3f}{row['words']:>8}")

    print(f"{'HUMAN':<14}{0.178:>12.3f}{0.338:>12.3f}{0.256:>12.3f}{131:>8}"
          "   <- leave-one-out, scripts/note_ceiling.py")

    # The controlled version of the wrong-patient test.
    #
    # Reading it off the table above would compare a doctor-written note against
    # an evaluator-written one, and the two groups write differently: the doctor
    # was in the room, the evaluators all worked from the same recording. Any
    # gap would then be partly authorship. So hold the writer fixed -- the same
    # doctor's note either way -- and swap only which patient it is about.
    right, wrong = [], []
    for i, cid in enumerate(cids):
        ev = list(notes[cid].evaluators.values())
        if not ev or not notes[cid].doctor:
            continue
        right.append(score(ev, notes[cid].doctor)["rougeL_max"])
        wrong.append(score(ev, notes[cids[(i + 1) % len(cids)]].doctor)["rougeL_max"])
    r, w = statistics.mean(right), statistics.mean(wrong)
    out["wrong_patient_control"] = {"right_patient": round(r, 4), "wrong_patient": round(w, 4)}
    print(f"\nWRONG-PATIENT CONTROL (same writer, same references, different patient)")
    print(f"  the doctor's note for THIS consultation   {r:.3f}")
    print(f"  the doctor's note for ANOTHER one         {w:.3f}")
    print(f"\n{w/r:.0%} of the score survives swapping the patient out entirely: house style,"
          f"\nsection headings and stock clinical phrasing. Everything that reflects having"
          f"\nunderstood this consultation is the remaining {r-w:.3f} of ROUGE-L.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
