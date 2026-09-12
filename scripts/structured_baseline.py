"""What you get on the structured task for free, before any model runs.

Four baselines, in rising order of effort and none of them a language model:

    majority     always answer the most common value for every field
    prior        answer the most common diagnosis, always
    lexical      regex over the transcript for urgency and follow-up interval
    oracle-span  the best any retrieval layer could possibly do: was the answer
                 even present in the transcript in a findable form

The last one is the point of this script. If a regex already recovers most of
the follow-up interval, then a semantic retrieval layer is solving a problem
that a twenty-line function solves, and the LLM's remaining job is small and
well-defined. If the regex fails, the oracle-span number says whether that is
the retriever's fault or whether the information simply is not in the words.

    python scripts/structured_baseline.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.dataset import load  # noqa: E402
from medvoice.structured import URGENCY, load_labels, score_all  # noqa: E402

# Spoken forms of an interval. Clinicians say "three to five days", "a week",
# "forty eight hours", "a couple of weeks", almost never a date.
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "fourteen": 14, "twenty": 20, "twenty four": 24,
        "forty eight": 48, "seventy two": 72, "couple": 2, "few": 3}
_UNIT = {"hour": 1 / 24, "hours": 1 / 24, "day": 1, "days": 1,
         "week": 7, "weeks": 7, "month": 30, "months": 30, "fortnight": 14}

# Anchoring this on a preposition ("in three days", "within a week") looked
# reasonable and was wrong: it recovered the interval in only 14 of 41 cases and
# made it look as though the information was absent from the audio. It is not.
# Clinicians drop the preposition constantly -- "give it three to five days",
# "see how you go a week", "come back Monday if" -- and dropping the requirement
# takes recall to 40 of 41. Worth remembering as a general point: an extractor
# that is too strict is indistinguishable from data that is not there.
_INTERVAL = re.compile(
    r"(?P<lo>\d+|" + "|".join(_NUM) + r")"
    r"(?:\s*(?:to|-|or|and)\s*(?P<hi>\d+|" + "|".join(_NUM) + r"))?"
    r"\s+(?:of\s+)?(?P<unit>hours?|days?|weeks?|months?|fortnight)\b", re.I)

_EMERGENCY = re.compile(r"\b(?:999|ambulance|blue light|paramedic|emergency services)\b", re.I)
_AANDE = re.compile(r"\b(?:a\s*&\s*e|a and e|accident and emergency|casualty|emergency department)\b", re.I)
_TODAY = re.compile(r"\b(?:today|this afternoon|this morning|straight away|right away|"
                    r"as soon as possible|asap|urgent(?:ly)?|next few hours)\b", re.I)

# Where a follow-up statement actually lives. Clinicians signal it before they
# say it, which is the cheap prefilter a semantic retriever would have to beat.
_FOLLOWUP_CUE = re.compile(
    r"\b(?:come back|call us|contact us|ring (?:us|back)|get in touch|see (?:you|me) again|"
    r"book (?:an|another)?\s*appointment|review|follow[- ]up|check (?:back|in)|"
    r"if (?:it|things|symptoms|you)[^.?!]{0,40}(?:worse|not better|no better|worsen)|"
    r"speak to (?:us|a doctor)|come in|pop back)\b", re.I)


def to_days(word: str, unit: str) -> float | None:
    w = word.lower().strip()
    n = int(w) if w.isdigit() else _NUM.get(w)
    if n is None:
        return None
    return n * _UNIT[unit.lower()]


def lexical_urgency(text: str) -> str:
    if _EMERGENCY.search(text):
        return "emergency"
    if _AANDE.search(text) or _TODAY.search(text):
        return "urgent_today"
    for m in _INTERVAL.finditer(text):
        d = to_days(m.group("lo"), m.group("unit"))
        if d is None:
            continue
        if d <= 10:
            return "days"
        return "weeks"
    return "conditional_only"


def lexical_interval(text: str) -> list | None:
    """Last interval mentioned wins: the plan comes at the end of a consultation."""
    best = None
    for m in _INTERVAL.finditer(text):
        lo = to_days(m.group("lo"), m.group("unit"))
        if lo is None or lo > 60:
            continue
        hi = to_days(m.group("hi"), m.group("unit")) if m.group("hi") else lo
        best = [round(lo, 2), round(hi or lo, 2)]
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/structured_baselines.json")
    args = ap.parse_args()

    labels = load_labels()
    cons = {c.cid: c for c in load()}
    cids = [c for c in sorted(labels) if c in cons]
    print(f"{len(labels)} labelled consultations, {len(cids)} with a transcript\n")

    # ---- what the labels look like, which is what any baseline exploits
    print("LABEL DISTRIBUTION")
    urg = Counter(l.urgency for l in labels.values())
    print("  urgency:      " + ", ".join(f"{k} {v}" for k, v in urg.most_common()))
    mod = Counter(l.modality for l in labels.values())
    print("  modality:     " + ", ".join(f"{k} {v}" for k, v in mod.most_common()))
    dx = Counter(d for l in labels.values() for d in l.diagnoses)
    print("  diagnoses:    %d distinct over %d consultations" % (len(dx), len(labels)))
    print("                " + ", ".join(f"{k} x{v}" for k, v in dx.most_common(8)))
    n_dx = Counter(len(l.diagnoses) for l in labels.values())
    print("  list length:  " + ", ".join(f"{k}:{v}" for k, v in sorted(n_dx.items())))
    print("  conditional follow-up: %d of %d (%.0f%%)"
          % (sum(l.conditional for l in labels.values()), len(labels),
             100 * sum(l.conditional for l in labels.values()) / len(labels)))
    print("  interval stated:       %d of %d"
          % (sum(l.interval_days is not None for l in labels.values()), len(labels)))
    print("  no explicit Imp line:  %d of %d"
          % (sum(not l.explicit for l in labels.values()), len(labels)))

    top_urg = urg.most_common(1)[0][0]
    top_mod = mod.most_common(1)[0][0]
    top_dx = dx.most_common(1)[0][0]

    systems = {
        "majority": {c: {"diagnoses": [], "followup": {
            "urgency": top_urg, "interval_days": None, "modality": top_mod, "conditional": True}}
            for c in cids},
        "prior": {c: {"diagnoses": [top_dx], "followup": {
            "urgency": top_urg, "interval_days": None, "modality": top_mod, "conditional": True}}
            for c in cids},
        "lexical": {},
    }
    for c in cids:
        t = cons[c].reference()
        systems["lexical"][c] = {"diagnoses": [], "followup": {
            "urgency": lexical_urgency(t), "interval_days": lexical_interval(t),
            "modality": top_mod, "conditional": True}}

    print(f"\n{'baseline':<12}{'dx F1':>8}{'dx top1':>9}{'urg exact':>11}"
          f"{'urg cost':>10}{'undercall':>11}{'interval':>10}")
    print("-" * 71)
    out = {}
    for name, preds in systems.items():
        s = score_all(labels, preds)
        out[name] = {k: v for k, v in s.items() if k != "rows"}
        print(f"{name:<12}{s['dx_f1']:>8.3f}{s['dx_top1']:>9.3f}{s['urgency_exact']:>11.3f}"
              f"{s['urgency_cost']:>10.3f}{s['urgency_undercall']:>11.3f}{s['interval']:>10.3f}")

    # ---- the oracle: is the answer even in the words?
    print("\nORACLE SPAN CHECK (is the information findable in the transcript at all)")
    cue_hit = interval_present = emergency_present = 0
    for c in cids:
        t = cons[c].reference()
        lab = labels[c]
        if _FOLLOWUP_CUE.search(t):
            cue_hit += 1
        if lab.interval_days is not None:
            g0, g1 = min(lab.interval_days), max(lab.interval_days)
            found = False
            for m in _INTERVAL.finditer(t):
                for grp in ("lo", "hi"):            # "three to five days" -> check both ends
                    if m.group(grp):
                        d = to_days(m.group(grp), m.group("unit"))
                        if d is not None and g0 - 2 <= d <= g1 + 2:
                            found = True
            interval_present += bool(found)
        if lab.urgency == "emergency":
            emergency_present += bool(_EMERGENCY.search(t))
    n_int = sum(l.interval_days is not None for c, l in labels.items() if c in cids)
    n_emg = sum(l.urgency == "emergency" for c, l in labels.items() if c in cids)
    print(f"  a follow-up cue phrase appears:            {cue_hit}/{len(cids)}")
    print(f"  the gold interval is literally spoken:     {interval_present}/{n_int}")
    print(f"  an emergency word appears when it should:  {emergency_present}/{n_emg}")
    out["oracle"] = {"followup_cue": [cue_hit, len(cids)],
                     "interval_spoken": [interval_present, n_int],
                     "emergency_word": [emergency_present, n_emg]}

    print(f"\nRead this as: a retrieval layer can only ever find what the top row says is\n"
          f"there, and the lexical row shows what a regex already recovers without one.")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
