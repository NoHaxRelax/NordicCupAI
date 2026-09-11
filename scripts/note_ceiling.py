"""What score does a real clinician get on this task?

Six people wrote up each of the 57 consultations: the doctor who ran it, plus
five evaluators working from the recording. Scoring each of them against the
other five gives the human ceiling, leave-one-out, on exactly the metric a
leaderboard would use.

This is the number that stops a team burning a day chasing a ROUGE score that
no human reaches. It also sets the bar honestly: a system at the human number
is not "only 0.2 ROUGE-L", it is writing notes as close to a clinician's as
clinicians are to each other.

    python scripts/note_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.notes import load_notes, score  # noqa: E402

METRICS = ("rouge1", "rouge2", "rouge3", "rougeL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/note_ceiling.json")
    args = ap.parse_args()

    notes = load_notes()
    print(f"{len(notes)} consultations, "
          f"{sum(1 + len(n.evaluators) for n in notes.values())} human notes\n")

    # leave-one-out: every human note scored against the other five
    rows: list[dict] = []
    for cid, ns in sorted(notes.items()):
        writers = [("doctor", ns.doctor)] + sorted(ns.evaluators.items())
        writers = [(w, t) for w, t in writers if t.strip()]
        if len(writers) < 2:
            continue
        for w, text in writers:
            others = [t for ww, t in writers if ww != w]
            rows.append({"cid": cid, "writer": w, "words": len(text.split())} | score(others, text))

    def agg(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))

    print("HUMAN CEILING, leave-one-out (each note vs the other five)")
    print(f"{'':<12}" + "".join(f"{m:>22}" for m in METRICS))
    print(f"{'':<12}" + "".join(f"{'max':>10}{'mean':>12}" for _ in METRICS))
    print("-" * (12 + 22 * len(METRICS)))
    out = {"leave_one_out": {}, "doctor_vs_evaluators": {}, "by_writer": {}}
    for label, sel in (("all humans", lambda r: True),
                       ("the doctor", lambda r: r["writer"] == "doctor"),
                       ("evaluators", lambda r: r["writer"] != "doctor")):
        line = f"{label:<12}"
        rec = {}
        for m in METRICS:
            mx, _, _ = agg(sel, f"{m}_max")
            mn, _, _ = agg(sel, f"{m}_mean")
            line += f"{mx:>10.3f}{mn:>12.3f}"
            rec[m] = {"max": round(mx, 4), "mean": round(mn, 4)}
        out["leave_one_out"][label] = rec
        print(line)

    sd = statistics.stdev([r["rougeL_max"] for r in rows])
    m = statistics.mean([r["rougeL_max"] for r in rows])
    print(f"\nROUGE-L (max over 5 references): {m:.3f} +/- {sd:.3f} over {len(rows)} notes")
    print(f"  range {min(r['rougeL_max'] for r in rows):.3f} to "
          f"{max(r['rougeL_max'] for r in rows):.3f}")
    out["rougeL_max"] = {"mean": round(m, 4), "sd": round(sd, 4), "n": len(rows)}

    # Single-reference is the regime a leaderboard most likely uses, and it is
    # much harsher than best-of-five. Worth knowing before the number lands.
    single = []
    for cid, ns in sorted(notes.items()):
        ev = sorted(ns.evaluators.items())
        if ns.doctor and ev:
            single.append(score([ns.doctor], ev[0][1])["rougeL_max"])
    if single:
        print(f"\nagainst the DOCTOR'S note alone (one reference, as a grader would):"
              f" ROUGE-L {statistics.mean(single):.3f}")
        out["single_reference_rougeL"] = round(statistics.mean(single), 4)

    lens = [r["words"] for r in rows]
    print(f"\nnote length: median {statistics.median(lens):.0f} words, "
          f"{min(lens)} to {max(lens)}")
    out["note_words"] = {"median": statistics.median(lens), "min": min(lens), "max": max(lens)}

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out | {"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
