"""The structured-output task: differential diagnosis plus follow-up plan.

Our working hypothesis is that Challenge 3 asks for a small JSON object rather
than a transcript, because free text is painful to rank and a JSON object is
not. labels/primock57_structured.json is that object, built by hand off the
clinician's own note for all 57 consultations, so a pipeline can be measured
end to end before the event on the shape we expect.

The scoring here is deliberately opinionated in one place. Urgency is not a
flat categorical: calling an ambulance when the truth was "review in a week" is
a wasted appointment, while saying "review in a week" when the truth was a blue
light is the failure that ends a company. `urgency_cost` encodes that asymmetry
explicitly, and plain accuracy is reported alongside it so nobody has to take
our weighting on trust.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LABELS = Path(__file__).resolve().parents[1] / "labels" / "primock57_structured.json"

# Ordered by how fast the patient needs to be seen. The index is the scale the
# cost function works on, so the order is load-bearing.
URGENCY = ["none", "conditional_only", "weeks", "days", "urgent_today", "emergency"]

MODALITIES = ["ambulance", "a_and_e", "f2f_gp", "remote_gp", "blood_test",
              "specialist", "physio", "self_referral", "pharmacy", "none"]

# Surface forms that mean the same condition. A system that writes "urine
# infection" and a label that says "UTI" agree clinically and should not be
# scored as a miss; this is the note-generation "ok"/"okay" problem again, and
# it is worth the same twenty lines here.
_ALIASES = {
    "uti": "uti", "urinary tract infection": "uti", "urine infection": "uti",
    "cystitis": "cystitis", "lower urinary tract infection": "uti",
    "urti": "urti", "upper respiratory tract infection": "urti",
    "viral urti": "urti", "viral upper respiratory tract infection": "urti",
    "common cold": "urti", "cold": "urti",
    "lrti": "lrti", "lower respiratory tract infection": "lrti",
    "chest infection": "lrti", "pneumonia": "lrti",
    "gastroenteritis": "gastroenteritis", "stomach bug": "gastroenteritis",
    "food poisoning": "food poisoning", "viral gastroenteritis": "gastroenteritis",
    "migraine": "migraine", "migraines": "migraine",
    "tension headache": "tension headache", "tension type headache": "tension headache",
    "eczema": "eczema", "eczema flare": "eczema", "atopic dermatitis": "eczema",
    "dermatitis": "dermatitis", "contact dermatitis": "contact dermatitis",
    "asthma exacerbation": "asthma exacerbation", "acute exacerbation of asthma": "asthma exacerbation",
    "exacerbation of asthma": "asthma exacerbation", "asthma attack": "asthma exacerbation",
    "mi": "myocardial infarction", "heart attack": "myocardial infarction",
    "acute coronary syndrome": "myocardial infarction", "acute cardiac event": "myocardial infarction",
    "stroke": "stroke", "cva": "stroke", "cerebrovascular accident": "stroke",
    "tia": "tia", "transient ischaemic attack": "tia", "mini stroke": "tia",
    "pe": "pulmonary embolism", "pulmonary embolism": "pulmonary embolism",
    "anaphylaxis": "anaphylaxis", "anaphylactic reaction": "anaphylaxis",
    "urticaria": "urticaria", "hives": "urticaria",
    "hypothyroidism": "hypothyroidism", "underactive thyroid": "hypothyroidism",
    "depression": "depression", "low mood": "depression",
    "anxiety disorder": "anxiety", "anxiety": "anxiety",
    "heart failure exacerbation": "heart failure exacerbation",
    "exacerbation of heart failure": "heart failure exacerbation",
    "pyrexia of unknown origin": "puo", "puo": "puo", "fever of unknown origin": "puo",
    "gord": "gord", "gerd": "gord", "acid reflux": "gord", "heartburn": "gord",
    "ms": "multiple sclerosis", "multiple sclerosis": "multiple sclerosis",
    "labyrinthitis": "labyrinthitis", "vestibular neuritis": "labyrinthitis",
    "bell's palsy": "bells palsy", "bells palsy": "bells palsy",
}

_STRIP = re.compile(r"\b(likely|possible|probable|suspected|acute|chronic|query|\?+)\b")


def canon(term: str) -> str:
    """Normalise a diagnosis string so two spellings of one condition compare equal."""
    t = term.lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)
    t = _STRIP.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _ALIASES.get(t, t)


@dataclass
class Label:
    cid: str
    diagnoses: list[str] = field(default_factory=list)
    to_exclude: list[str] = field(default_factory=list)
    explicit: bool = True
    urgency: str = "none"
    interval_days: list | None = None
    modality: str = "none"
    booked: bool = False
    conditional: bool = False
    trigger: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return canon(self.diagnoses[0]) if self.diagnoses else ""


def load_labels(path: str | Path = LABELS) -> dict[str, Label]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for cid, r in d["consultations"].items():
        f = r["followup"]
        out[cid] = Label(
            cid=cid, diagnoses=r["diagnoses"], to_exclude=r.get("diagnoses_to_exclude", []),
            explicit=r.get("diagnosis_explicit", True), urgency=f["urgency"],
            interval_days=f.get("interval_days"), modality=f.get("modality", "none"),
            booked=bool(f.get("booked")), conditional=bool(f.get("conditional")),
            trigger=f.get("trigger", ""), flags=r.get("flags", []),
        )
    return out


# --------------------------------------------------------------------------- scoring

def diagnosis_score(gold: list[str], pred: list[str]) -> dict:
    """Set F1 over canonicalised diagnoses, plus whether the top guess was right.

    Both halves matter and they pull in opposite directions. A system can win F1
    by listing every plausible condition, which is exactly the behaviour a
    clinician would call useless, so top-1 is reported next to it.
    """
    g = [canon(x) for x in gold]
    p = [canon(x) for x in pred]
    gs, ps = set(g), set(p)
    hit = len(gs & ps)
    prec = hit / len(ps) if ps else 0.0
    rec = hit / len(gs) if gs else 0.0
    return {
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec, "recall": rec,
        "top1": 1.0 if p and g and p[0] == g[0] else 0.0,
        "primary_recalled": 1.0 if g and g[0] in ps else 0.0,
        "n_pred": len(ps), "n_gold": len(gs),
    }


def urgency_cost(gold: str, pred: str) -> float:
    """Asymmetric cost in [0, 1]. 0 is correct.

    Under-calling is punished at twice the rate of over-calling, and any
    under-call of a true emergency is a full 1.0 regardless of distance. Sending
    a well patient to hospital wastes an appointment; leaving a stroke at home
    for a week does not.
    """
    if gold == pred:
        return 0.0
    try:
        gi, pi = URGENCY.index(gold), URGENCY.index(pred)
    except ValueError:
        return 1.0
    if gold == "emergency" and pred != "emergency":
        return 1.0
    steps = abs(gi - pi) / (len(URGENCY) - 1)
    return min(1.0, steps * (2.0 if pi < gi else 1.0))


def interval_score(gold: list | None, pred: list | None, tol_days: float = 2.0) -> float:
    """1.0 if the predicted window overlaps the gold window within a tolerance.

    Ranges, not points: a clinician says "3 to 5 days", and a system answering
    "4 days" is right. Both sides null also counts as agreement, since "no
    interval was agreed" is a real and common answer.
    """
    if gold is None and pred is None:
        return 1.0
    if gold is None or pred is None:
        return 0.0
    g0, g1 = min(gold), max(gold)
    p0, p1 = min(pred), max(pred)
    return 1.0 if (p0 <= g1 + tol_days and p1 >= g0 - tol_days) else 0.0


def score_one(gold: Label, pred: dict) -> dict:
    """Score one predicted JSON object against its label."""
    f = pred.get("followup", pred)
    d = diagnosis_score(gold.diagnoses, pred.get("diagnoses", []) or [])
    ug, up = gold.urgency, f.get("urgency", "none")
    return {
        "cid": gold.cid,
        "dx_f1": round(d["f1"], 4), "dx_top1": d["top1"],
        "dx_primary_recalled": d["primary_recalled"],
        "dx_precision": round(d["precision"], 4), "dx_recall": round(d["recall"], 4),
        "urgency_exact": 1.0 if ug == up else 0.0,
        "urgency_cost": round(urgency_cost(ug, up), 4),
        "urgency_undercall": 1.0 if (ug in URGENCY and up in URGENCY
                                     and URGENCY.index(up) < URGENCY.index(ug)) else 0.0,
        "interval": interval_score(gold.interval_days, f.get("interval_days")),
        "modality_exact": 1.0 if gold.modality == f.get("modality") else 0.0,
        "conditional_exact": 1.0 if gold.conditional == bool(f.get("conditional")) else 0.0,
    }


def score_all(labels: dict[str, Label], preds: dict[str, dict]) -> dict:
    """Mean over consultations, plus the two numbers that would actually matter
    to a clinician: how often the real diagnosis was anywhere in the list, and
    how often the system under-called urgency."""
    rows = [score_one(labels[c], preds[c]) for c in sorted(preds) if c in labels]
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "cid"]
    out = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    out["n"] = len(rows)
    out["rows"] = rows
    return out
