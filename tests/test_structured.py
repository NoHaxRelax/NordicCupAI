"""Checks on the structured labels and their scorer.

The label file is hand-written, so most of this validates it against itself:
every enum value legal, every exclusion actually listed as a diagnosis, every
emergency marked same-day. Hand-built labels are where silent inconsistencies
live, and a wrong label costs more than a wrong model.

    python tests/test_structured.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.structured import (MODALITIES, URGENCY, canon, diagnosis_score,  # noqa: E402
                                 interval_score, load_labels, score_one,
                                 urgency_cost)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


labels = load_labels()

print("label file")
check("57 consultations", len(labels) == 57, str(len(labels)))
bad = [l.cid for l in labels.values() if l.urgency not in URGENCY]
check("every urgency is a legal value", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values() if l.modality not in MODALITIES]
check("every modality is a legal value", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values() if not l.diagnoses]
check("every consultation has a diagnosis", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values()
       if not set(map(canon, l.to_exclude)) <= set(map(canon, l.diagnoses))]
check("every excluded condition is also listed as a diagnosis", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values() if l.urgency == "emergency" and l.interval_days != [0, 0]]
check("emergencies are same-day", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values()
       if l.interval_days is not None and (len(l.interval_days) != 2
                                           or l.interval_days[0] > l.interval_days[1])]
check("intervals are well-formed [min, max]", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values() if l.urgency == "conditional_only" and not l.conditional]
check("conditional_only implies conditional", not bad, ", ".join(bad))
bad = [l.cid for l in labels.values() if l.modality == "ambulance" and l.urgency != "emergency"]
check("ambulance implies emergency", not bad, ", ".join(bad))
check("all flags are known defects",
      all(f in ("presenting_complaint_mismatch", "duplicated_note_body", "leaked_note")
          for l in labels.values() for f in l.flags))

print("\ncanonicalisation")
check("spelling variants collapse", canon("Urine infection") == canon("UTI"))
check("hedges are stripped", canon("? likely UTI") == canon("UTI"), canon("? likely UTI"))
check("plurals collapse", canon("migraines") == canon("migraine"))
check("unknown terms pass through", canon("Sarcoidosis") == "sarcoidosis")
check("distinct conditions stay distinct", canon("UTI") != canon("URTI"))

print("\nurgency cost")
check("correct costs nothing", urgency_cost("days", "days") == 0.0)
check("missing an emergency costs everything", urgency_cost("emergency", "days") == 1.0)
check("missing an emergency by one step still costs everything",
      urgency_cost("emergency", "urgent_today") == 1.0)
check("under-calling costs more than over-calling",
      urgency_cost("urgent_today", "weeks") > urgency_cost("weeks", "urgent_today"),
      f"{urgency_cost('urgent_today', 'weeks'):.2f} vs {urgency_cost('weeks', 'urgent_today'):.2f}")
check("cost stays in [0, 1]",
      all(0.0 <= urgency_cost(a, b) <= 1.0 for a in URGENCY for b in URGENCY))
check("unknown label costs everything", urgency_cost("days", "nonsense") == 1.0)

print("\ninterval")
check("both absent agrees", interval_score(None, None) == 1.0)
check("one absent disagrees", interval_score([7, 7], None) == 0.0)
check("a point inside a range matches", interval_score([3, 5], [4, 4]) == 1.0)
check("overlapping ranges match", interval_score([3, 5], [5, 10]) == 1.0)
check("tolerance is applied", interval_score([7, 7], [9, 9]) == 1.0)
check("beyond tolerance fails", interval_score([7, 7], [14, 14]) == 0.0)

print("\ndiagnosis scoring")
d = diagnosis_score(["UTI"], ["urine infection"])
check("an alias counts as a hit", d["f1"] == 1.0 and d["top1"] == 1.0)
d = diagnosis_score(["migraine", "tension headache"], ["tension headache", "migraine"])
check("order does not affect F1", d["f1"] == 1.0)
check("order does affect top-1", d["top1"] == 0.0)
d = diagnosis_score(["UTI"], ["UTI", "cystitis", "pyelonephritis", "stones"])
check("padding the list hurts precision", d["precision"] == 0.25 and d["recall"] == 1.0)
check("but the primary is still recalled", d["primary_recalled"] == 1.0)
check("an empty prediction scores zero", diagnosis_score(["UTI"], [])["f1"] == 0.0)

print("\nend to end")
g = labels["day2_consultation07"]           # the cardiac emergency
perfect = score_one(g, {"diagnoses": ["acute coronary syndrome"], "followup": {
    "urgency": "emergency", "interval_days": [0, 0], "modality": "ambulance", "conditional": False}})
check("a perfect prediction scores perfectly",
      perfect["dx_top1"] == 1.0 and perfect["urgency_cost"] == 0.0
      and perfect["interval"] == 1.0 and perfect["modality_exact"] == 1.0)
missed = score_one(g, {"diagnoses": ["indigestion"], "followup": {
    "urgency": "days", "interval_days": [7, 7], "modality": "remote_gp", "conditional": True}})
check("sending a heart attack home is maximally penalised",
      missed["urgency_cost"] == 1.0 and missed["urgency_undercall"] == 1.0)

print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
sys.exit(1 if FAILED else 0)
