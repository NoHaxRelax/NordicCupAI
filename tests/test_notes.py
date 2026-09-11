"""Checks on the note scorer, run with `python tests/test_notes.py`.

The interesting ones are the last two. `norm_cid` guards a join that fails
silently -- eleven consultations losing every reference note, with no
exception -- and the corpus check pins the human ceiling, so if a refactor
moves the number we find out here rather than halfway through the event.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medvoice.notes import (load_notes, norm_cid, rouge_l, rouge_n, score,  # noqa: E402
                            toks)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


print("tokenisation")
check("shorthand survives", toks("3/7 hx of LLQ pain") == ["3/7", "hx", "of", "llq", "pain"],
      str(toks("3/7 hx of LLQ pain")))
check("punctuation is a separator", toks("Imp: gastroenteritis.") == ["imp", "gastroenteritis"])
check("x6/day stays one token", toks("Opening bowels x6/day") == ["opening", "bowels", "x6/day"])

print("\nrouge")
check("identical text scores 1", close(rouge_l("a b c", "a b c")[2], 1.0))
check("disjoint text scores 0", close(rouge_l("a b c", "x y z")[2], 0.0))
check("empty hypothesis scores 0", close(rouge_l("a b c", "")[2], 0.0))
check("empty reference scores 0", close(rouge_l("", "a b c")[2], 0.0))
# LCS of "a b c d" and "a c d" is "a c d" -> p 3/3, r 3/4, f1 6/7
check("rouge-l uses LCS, not bag of words", close(rouge_l("a b c d", "a c d")[2], 6 / 7))
# order matters for LCS but not for unigrams
check("rouge-1 ignores order", close(rouge_n("a b c", "c b a", 1)[2], 1.0))
check("rouge-l respects order", rouge_l("a b c", "c b a")[2] < 1.0)
# clipping: two "a" in the hypothesis cannot both match one "a" in the reference
check("n-gram counts are clipped", close(rouge_n("a b", "a a b", 1)[2], 2 * (2 / 3) * 1 / (2 / 3 + 1)))

print("\nconsultation ids")
check("three-digit form normalises", norm_cid("day1_consultation010") == "day1_consultation10")
check("two-digit form is unchanged", norm_cid("day1_consultation10") == "day1_consultation10")
check("single digit pads", norm_cid("day1_consultation1") == "day1_consultation01")
check("already padded stays", norm_cid("day1_consultation01") == "day1_consultation01")
check("filename suffix tolerated", norm_cid("day3_consultation07_doctor.wav") == "day3_consultation07")
check("unparseable passes through", norm_cid("not-an-id") == "not-an-id")

print("\ncorpus")
try:
    notes = load_notes()
except FileNotFoundError as exc:
    print(f"  skip  corpus checks: {exc}")
    notes = {}

if notes:
    check("57 consultations", len(notes) == 57, str(len(notes)))
    per = {cid: len(n.evaluators) for cid, n in notes.items()}
    thin = [c for c, k in per.items() if k != 5]
    check("every consultation has 5 evaluator notes", not thin,
          f"{len(thin)} with a different count" if thin else "")
    check("every consultation has 6 references",
          all(len(n.refs) == 6 for n in notes.values()))
    check("every reference is non-empty",
          all(r.strip() for n in notes.values() for r in n.refs))

    # a note scored against a reference set containing itself must be perfect
    c = notes["day1_consultation01"]
    check("self-match is 1.0", close(score(c.refs, c.doctor)["rougeL_max"], 1.0))

    # the human ceiling, leave-one-out, as reported in the dossier
    import statistics
    loo = []
    for n in notes.values():
        writers = [n.doctor] + list(n.evaluators.values())
        for i, w in enumerate(writers):
            loo.append(score(writers[:i] + writers[i + 1:], w)["rougeL_max"])
    m = statistics.mean(loo)
    check("human ceiling is 0.32 ROUGE-L", 0.31 <= m <= 0.33, f"{m:.3f} over {len(loo)} notes")

print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
sys.exit(1 if FAILED else 0)
