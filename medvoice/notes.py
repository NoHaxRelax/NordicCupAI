"""Consultation note generation: scoring, references, and house style.

The second half of the challenge. Transcription gets you words; the note is
what a clinician would actually keep, and it is graded differently.

Three things the PriMock57 human-evaluation data settles for us, all measured
in scripts/note_metrics.py rather than assumed:

*Omissions dominate.* Across 1425 clinician-marked notes, 64% of critical
errors are things the note left out, not things it made up. The instinct with a
language model in a clinical setting is to clamp down on hallucination; the
measured failure is under-coverage. Design for recall.

*n-gram overlap beats embeddings.* Ranking metrics by how well they agree with
a clinician's error count, ROUGE-3/4 and WIL come top (rho about 0.68) and the
sentence-embedding metrics come last (Embedding Average, rho 0.26). BertScore
is mid-table and costs a model to run. So the dev-loop metric is ROUGE, which
is twenty lines and no dependency.

*The reference you score against moves the number more than the system does.*
The same notes score ROUGE-L 0.380 against the one doctor who ran the
consultation and 0.835 against the best of several independently written notes.
Two clinicians writing up the same conversation simply do not write the same
note. Always say which regime a number came from.

House style matters for the same reason ROUGE does. These notes are dense
clinical shorthand ("3/7 hx of diarrhea", "LLQ pain", "PMH: Asthma", "Imp:
gastroenteritis"), around 136 words. A fluent, correct, plain-English paragraph
scores badly against that, which is an argument about format, not about
medicine, and it is cheap to win with a few-shot prompt.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "primock57"

_RE_CID = re.compile(r"day(\d+)[_ ]*consultation[_ ]*(\d+)", re.I)


def norm_cid(s: str) -> str:
    """Canonical consultation id: two-digit number, matching the audio and
    transcript filenames.

    The corpus is not internally consistent about this. human_eval_data's
    results.csv zero-pads to three ("day1_consultation010") while every filename
    and notes/*.json use two ("day1_consultation10"). Joining the two sources on
    the raw string loses every consultation numbered 10 and up: 11 of 57, and it
    fails silently as an empty reference list rather than as an error.
    """
    m = _RE_CID.search(s or "")
    return f"day{int(m.group(1))}_consultation{int(m.group(2)):02d}" if m else (s or "")

# The shorthand the PriMock57 clinicians actually use, with counts over the 57
# notes. A generated note that spells these out loses n-gram overlap on every
# occurrence, so they go in the prompt.
SHORTHAND = {
    "Imp": ("impression / working diagnosis", 42),
    "SH": ("social history", 36),
    "Nil": ("none", 36),
    "hx": ("history", 33),
    "DH": ("drug history", 33),
    "PMH": ("past medical history", 30),
    "sx": ("symptoms", 27),
    "FH": ("family history", 27),
    "SOB": ("shortness of breath", 19),
    "PC": ("presenting complaint", 18),
    "NKDA": ("no known drug allergies", 17),
    "HPC": ("history of presenting complaint", 15),
    "LLQ": ("left lower quadrant", 0),
    "RUQ": ("right upper quadrant", 0),
    "ADLs": ("activities of daily living", 0),
    "OTC": ("over the counter", 0),
    "etOH": ("alcohol", 0),
    "D/w": ("discussed with", 0),
    "N/V": ("nausea and vomiting", 0),
    "3/7": ("three days (n/7 = days, n/52 = weeks, n/12 = months)", 0),
    "x6/day": ("six times a day", 0),
}

# The headings the corpus actually uses, in the order they appear. A note laid
# out this way lines up with the reference section by section.
SECTIONS = ["PC", "HPC", "PMH", "DH", "FH", "SH", "Imp", "Plan"]


# --------------------------------------------------------------------------- scoring

def toks(text: str) -> list[str]:
    """Lowercased word tokens. Shorthand like "3/7" and "x6/day" carries meaning,
    so digits and slashes survive; everything else is a separator."""
    return re.findall(r"[a-z0-9]+(?:[/][a-z0-9]+)*", text.lower())


def _f1(match: int, n_ref: int, n_hyp: int) -> tuple[float, float, float]:
    p = match / n_hyp if n_hyp else 0.0
    r = match / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rouge_n(ref: str, hyp: str, n: int = 1) -> tuple[float, float, float]:
    """Clipped n-gram overlap, precision / recall / F1."""
    a, b = toks(ref), toks(hyp)
    if len(a) < n or len(b) < n:
        return 0.0, 0.0, 0.0
    ga = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    gb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    match = sum(min(c, gb[g]) for g, c in ga.items())
    return _f1(match, sum(ga.values()), sum(gb.values()))


def rouge_l(ref: str, hyp: str) -> tuple[float, float, float]:
    """Longest common subsequence, precision / recall / F1.

    Two rolling rows rather than the full table: notes run to a few hundred
    tokens, and against five references per consultation the quadratic table is
    the slow part of a dev loop that should be instant.
    """
    a, b = toks(ref), toks(hyp)
    if not a or not b:
        return 0.0, 0.0, 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return _f1(prev[-1], len(a), len(b))


def score(refs: list[str], hyp: str) -> dict:
    """Score one note against every available reference.

    Reports both regimes, because they answer different questions: `max` is
    "did the system write a defensible note", `mean` is "does it agree with the
    typical clinician". The metrics are the ones that correlate with clinician
    judgement; recall is broken out separately since omissions are the dominant
    critical error.
    """
    if not refs:
        raise ValueError("no references")
    per = []
    for r in refs:
        e = {"rouge1": rouge_n(r, hyp, 1), "rouge2": rouge_n(r, hyp, 2),
             "rouge3": rouge_n(r, hyp, 3), "rougeL": rouge_l(r, hyp)}
        per.append(e)
    out: dict = {"n_refs": len(refs), "hyp_words": len(toks(hyp))}
    for m in ("rouge1", "rouge2", "rouge3", "rougeL"):
        f1s = [p[m][2] for p in per]
        out[f"{m}_max"] = round(max(f1s), 4)
        out[f"{m}_mean"] = round(sum(f1s) / len(f1s), 4)
    out["rougeL_recall_max"] = round(max(p["rougeL"][1] for p in per), 4)
    out["rougeL_prec_max"] = round(max(p["rougeL"][0] for p in per), 4)
    return out


def corpus_score(pairs: list[tuple[list[str], str]]) -> dict:
    """Mean of per-note scores. Unlike WER there is no token-pooled definition
    that means anything here, so this really is an average over notes."""
    if not pairs:
        return {}
    rows = [score(refs, hyp) for refs, hyp in pairs]
    keys = [k for k in rows[0] if k != "n_refs"]
    out = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
    out["notes"] = len(rows)
    return out


# --------------------------------------------------------------------------- references

@dataclass
class NoteSet:
    """Every note written about one consultation."""
    cid: str
    doctor: str = ""                                  # the clinician who ran it
    evaluators: dict[str, str] = field(default_factory=dict)   # 5 independent write-ups
    presenting_complaint: str = ""
    highlights: list[str] = field(default_factory=list)

    @property
    def refs(self) -> list[str]:
        return [self.doctor] + list(self.evaluators.values()) if self.doctor else list(self.evaluators.values())


def load_notes(root: str | Path = DEFAULT_ROOT) -> dict[str, NoteSet]:
    """The doctor's note plus the five evaluator notes per consultation.

    The evaluator notes are the reason to bother: they come from
    human_eval_data/results.csv, which most people never open, and they turn a
    single-reference metric into a five-reference one. Given how far apart two
    clinicians' notes are, that is the difference between a number that tracks
    quality and one that tracks writing habits.
    """
    root = Path(root)
    out: dict[str, NoteSet] = {}
    for f in sorted((root / "notes").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        cid = norm_cid(f"day{d['day']}_consultation{d['consultation']}")
        out[cid] = NoteSet(cid, doctor=d.get("note", "").strip(),
                           presenting_complaint=d.get("presenting_complaint", "").strip(),
                           highlights=d.get("highlights", []))

    res = root / "human_eval_data" / "results.csv"
    if res.is_file():
        # the note fields run to thousands of characters with embedded newlines
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with res.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cid, ev = norm_cid(row.get("Consultation", "")), row.get("Evaluator", "")
                note = (row.get("Evaluator Note") or "").strip()
                if cid in out and ev and note:
                    out[cid].evaluators.setdefault(ev, note)
    return out


# --------------------------------------------------------------------------- prompting

def style_card() -> str:
    """The format half of the prompt. Kept separate from the examples so it can
    be swapped for whatever house style the organisers turn out to want."""
    gloss = "\n".join(f"  {k:<7} {v}" for k, (v, _) in SHORTHAND.items())
    return (
        "Write the note the way a UK GP writes it in the record: clipped "
        "fragments, no full sentences, no preamble, around 130 words.\n\n"
        "Sections, in this order, omitting any the conversation does not cover:\n"
        f"  {' / '.join(SECTIONS)}\n\n"
        "Standard abbreviations, used the same way:\n" + gloss + "\n\n"
        "Durations are written n/7 for days, n/52 for weeks, n/12 for months, "
        "so three days is 3/7.\n"
        "Record explicit negatives the patient gave you ('No blood in stool', "
        "'Nil smoking'); they are clinically load-bearing and a clinician marks "
        "their absence as an omission."
    )


def few_shot(notes: dict[str, NoteSet], transcripts: dict[str, str],
             k: int = 3, exclude: str | None = None) -> list[tuple[str, str]]:
    """Pick k (transcript, note) examples.

    Chosen by note length closest to the corpus median, so the model copies a
    typical note rather than the longest or the most telegraphic one. Anything
    in `exclude` is dropped, which is what keeps a dev-set consultation from
    teaching the model its own answer.
    """
    cands = [(cid, n) for cid, n in notes.items()
             if n.doctor and cid in transcripts and cid != exclude]
    if not cands:
        return []
    lens = sorted(len(n.doctor.split()) for _, n in cands)
    median = lens[len(lens) // 2]
    cands.sort(key=lambda cn: abs(len(cn[1].doctor.split()) - median))
    return [(transcripts[cid], n.doctor) for cid, n in cands[:k]]


def build_prompt(transcript: str, examples: list[tuple[str, str]]) -> str:
    parts = [
        "You are writing the clinical record entry for a primary-care "
        "consultation, from its transcript.\n",
        style_card(),
        "\nCover everything the patient said that bears on the diagnosis or the "
        "plan. Leaving something out is the most common and most serious fault "
        "in these notes; when in doubt, include it.\n",
    ]
    for i, (t, n) in enumerate(examples, 1):
        parts.append(f"\n--- example {i}: transcript ---\n{t}\n--- example {i}: note ---\n{n}")
    parts.append(f"\n--- transcript ---\n{transcript}\n--- note ---\n")
    return "\n".join(parts)
