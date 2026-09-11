"""Which automatic metric actually tracks a doctor's judgement of a note?

PriMock57 ships the data behind "Human Evaluation and Correlation with Automatic
Metrics in Consultation Note Generation": 1425 judged notes over 57
consultations, 5 evaluators and 11 systems (10 models plus the doctor's own
note), each row carrying both the machine scores (24 metrics, 4 reference
regimes) and what a clinician did with the note -- how long they spent fixing
it, and how many statements were wrong or missing, split by whether the error
was critical.

We use it for two decisions:

    1. If the organisers grade note generation with ROUGE, our dev loop should
       optimise ROUGE even where ROUGE is a poor proxy for quality. If they
       grade with anything human-shaped, the ranking of metrics below tells us
       which cheap metric to iterate against.
    2. The reference regime matters as much as the metric. "human_note" scores
       against the one doctor who ran the consultation; "max" scores against
       the best of several independently written notes. The gap between them is
       inter-annotator variance, and it is large enough to swamp real system
       differences if you report the wrong one.

    python scripts/note_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "primock57" / "human_eval_data" / "metric-scores.json"

# Metrics whose sign is already "higher is better" are starred in the source
# file. The unstarred ones are distances, and the file pre-negates them, so
# every column here reads the same direction.
HUMAN = {
    "time_sec": "post-edit time",
    "incorrect_critical": "critical incorrect",
    "incorrect_noncritical": "non-critical incorrect",
    "omissions_critical": "critical omissions",
    "omissions_noncritical": "non-critical omissions",
}


def rank(xs: list[float]) -> list[float]:
    """Average ranks, ties shared, so Spearman is Pearson on these."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(rank(a), rank(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT))
    ap.add_argument("--out", default="runs/note_metric_correlations.json")
    ap.add_argument("--regime", default="avg", help="reference regime for the main table")
    args = ap.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    regimes = sorted({k.split(" | ")[0] for k in rows[0] if " | " in k})
    metrics = sorted({k.split(" | ")[1] for k in rows[0] if " | " in k})
    models = sorted({r["model_id"] for r in rows})
    print(f"{len(rows)} judged notes, {len(models)} systems, "
          f"{len({r['consultation_id'] for r in rows})} consultations")
    print(f"regimes: {', '.join(regimes)}")
    print(f"metrics: {len(metrics)}\n")

    # A doctor who has to fix more wrong statements and restore more omissions
    # spends longer doing it; the composite is what we actually want to minimise.
    for r in rows:
        r["critical_errors"] = r["incorrect_critical"] + r["omissions_critical"]
        r["all_errors"] = r["critical_errors"] + r["incorrect_noncritical"] + r["omissions_noncritical"]
    HUMAN["critical_errors"] = "critical errors (both kinds)"
    HUMAN["all_errors"] = "all errors"

    # ---- note level: does the metric rank individual notes the way a doctor would?
    out = {"note_level": {}, "system_level": {}, "regime_gap": {}}
    print(f"NOTE LEVEL  (Spearman, n={len(rows)}, reference regime = {args.regime!r})")
    print("Sign is flipped so positive always means 'metric agrees with the doctor'.\n")
    hdr = f"{'metric':<20}" + "".join(f"{h[:13]:>15}" for h in
                                      ("post-edit time", "critical err", "all errors"))
    print(hdr)
    print("-" * len(hdr))
    table = []
    for m in metrics:
        col = [r[f"{args.regime} | {m}"] for r in rows]
        # every human column is a cost, so a good metric anti-correlates with it
        vals = {h: -spearman(col, [float(r[h]) for r in rows])
                for h in ("time_sec", "critical_errors", "all_errors")}
        table.append((m, vals))
        out["note_level"][m] = {k: round(v, 4) for k, v in vals.items()}
    table.sort(key=lambda kv: -kv[1]["critical_errors"])
    for m, v in table:
        print(f"{m:<20}" + "".join(f"{v[k]:>15.3f}" for k in
                                   ("time_sec", "critical_errors", "all_errors")))

    # ---- system level: would the metric pick the same winner out of the field?
    #
    # "doctor" has to come out. That system's note IS the human_note reference,
    # so it scores a perfect 1.000 against itself while also drawing the fewest
    # human-judged errors. Leaving it in hands every metric one free correctly
    # ranked pair and makes them all look better at ranking systems than they
    # are. The ten real systems sit in a narrow band, which is the hard case and
    # the only one that resembles a leaderboard.
    ranked = [m for m in models if m != "doctor"]
    print(f"\nSYSTEM LEVEL  (Spearman over {len(ranked)} systems, self-referential")
    print("  'doctor' excluded; 'would this metric pick the right model')")
    sysrows = []
    for m in metrics:
        mm, hh = [], []
        for mid in ranked:
            sub = [r for r in rows if r["model_id"] == mid]
            mm.append(sum(r[f"{args.regime} | {m}"] for r in sub) / len(sub))
            hh.append(sum(r["critical_errors"] for r in sub) / len(sub))
        rho = -spearman(mm, hh)
        sysrows.append((m, rho))
        out["system_level"][m] = round(rho, 4)
    sysrows.sort(key=lambda kv: -kv[1])
    for m, rho in sysrows:
        print(f"  {m:<20} {rho:>7.3f}")

    # ---- how much does the choice of reference change the number?
    print("\nREFERENCE REGIME  (mean score of the same notes, same metric)")
    print(f"{'metric':<20}" + "".join(f"{g:>14}" for g in regimes))
    print("-" * (20 + 14 * len(regimes)))
    for m in ("ROUGE-L-F1*", "BertScore*", "Stanza+Snomed*", "METEOR*"):
        if m not in metrics:
            continue
        means = {g: sum(r[f"{g} | {m}"] for r in rows) / len(rows) for g in regimes}
        out["regime_gap"][m] = {g: round(v, 4) for g, v in means.items()}
        print(f"{m:<20}" + "".join(f"{means[g]:>14.3f}" for g in regimes))

    # ---- what the errors actually are, since that is what we have to design against
    n = len(rows)
    print("\nWHAT GOES WRONG  (mean per note, across all systems)")
    for k, label in (("omissions_critical", "critical omissions"),
                     ("incorrect_critical", "critical incorrect statements"),
                     ("omissions_noncritical", "non-critical omissions"),
                     ("incorrect_noncritical", "non-critical incorrect")):
        print(f"  {label:<32} {sum(float(r[k]) for r in rows)/n:5.2f}")
    print(f"  {'post-edit time (s)':<32} {sum(float(r['time_sec']) for r in rows)/n:5.1f}")
    share = sum(float(r["omissions_critical"]) for r in rows) / max(
        1e-9, sum(float(r["critical_errors"]) for r in rows))
    print(f"\n  omissions are {share:.0%} of all critical errors")

    # The human ceiling. A doctor's own note, read by a different clinician,
    # still draws critical corrections, so zero is not the target and anything
    # near this line is at parity with a person.
    doc = [r for r in rows if r["model_id"] == "doctor"]
    gen = [r for r in rows if r["model_id"] != "doctor"]
    if doc and gen:
        out["human_ceiling"] = {
            "critical_errors": round(sum(r["critical_errors"] for r in doc) / len(doc), 2),
            "time_sec": round(sum(float(r["time_sec"]) for r in doc) / len(doc), 1),
        }
        print(f"\n  a doctor's OWN note, marked up by another clinician: "
              f"{out['human_ceiling']['critical_errors']:.2f} critical errors, "
              f"{out['human_ceiling']['time_sec']:.0f}s to edit")
        print(f"  the 10 generated systems average:                    "
              f"{sum(r['critical_errors'] for r in gen)/len(gen):.2f} critical errors, "
              f"{sum(float(r['time_sec']) for r in gen)/len(gen):.0f}s to edit")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
