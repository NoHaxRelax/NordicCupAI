"""Scoring. WER and the variants this task could plausibly be graded on.

Normalisation is the part that decides the number. Two systems that transcribe
identically can differ by several WER points purely on casing, punctuation,
contractions and how numbers are written, so the normaliser here follows the
conventions Whisper's English normaliser uses. Whatever the organisers score,
we want our dev number computed the same way theirs will be.

No hard dependency on jiwer: the edit distance is 20 lines and one less thing
to install on the day.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Spoken-form contractions and fillers. Whisper's normaliser expands these, and
# a reference that says "do not" while the hypothesis says "don't" is not an
# error anyone means to count.
_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'t": " not", "'ve": " have",
    "'m": " am",
}
_FILLERS = {"uh", "um", "erm", "hmm", "mhm", "mm", "er", "ah", "eh", "hm"}

# Spelling conventions that differ between human transcribers and Whisper but
# are the SAME SPOKEN WORD. Measured on PriMock57: the reference writes "ok"
# 2058 times and "okay" 34, while Whisper always writes "Okay". Left unmapped,
# that single convention mismatch accounts for several WER points of pure
# measurement artifact.
_VARIANTS = {
    "okay": "ok", "o.k.": "ok", "kay": "ok",
    "yep": "yeah", "yup": "yeah", "yah": "yeah", "ya": "yeah", "yes": "yes",
    "alright": "all right", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
    "cause": "because", "cos": "because", "coz": "because",
    "doctor": "dr", "mister": "mr", "missus": "mrs",
}

# Short affirmations and acknowledgements. Human transcribers faithfully record
# every one; Whisper's VAD and decoder drop many as non-speech. Whether that
# counts as an error depends entirely on the scoring convention, so it is
# measured separately rather than silently chosen.
_BACKCHANNELS = {"yeah", "ok", "right", "sure", "mmm", "uhhuh", "yes", "no", "oh"}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}


def normalise(text: str, *, drop_fillers: bool = True, numbers: bool = True,
              variants: bool = True, drop_backchannels: bool = False) -> list[str]:
    """Text -> comparable token list."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"<[^>]*>", " ", t)                 # transcription tags
    t = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", t)     # bracketed asides
    for a, b in _CONTRACTIONS.items():
        t = t.replace(a, b)
    t = re.sub(r"[^\w\s%]", " ", t)                # keep % , it is a real token in dosages
    toks = t.split()
    if drop_fillers:
        toks = [w for w in toks if w not in _FILLERS]
    if variants:
        expanded: list[str] = []
        for w in toks:
            expanded.extend(_VARIANTS.get(w, w).split())
        toks = expanded
    if numbers:
        toks = [_NUMBER_WORDS.get(w, w) for w in toks]
    if drop_backchannels:
        toks = [w for w in toks if w not in _BACKCHANNELS]
    return toks


@dataclass
class EditCounts:
    hits: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0

    @property
    def ref_len(self) -> int:
        return self.hits + self.subs + self.dels

    @property
    def wer(self) -> float:
        return (self.subs + self.dels + self.ins) / max(1, self.ref_len)

    def __add__(self, o: "EditCounts") -> "EditCounts":
        return EditCounts(self.hits + o.hits, self.subs + o.subs, self.dels + o.dels, self.ins + o.ins)


def align(ref: list[str], hyp: list[str]) -> tuple[EditCounts, list[tuple[str, str, str]]]:
    """Levenshtein alignment. Returns counts and the (op, ref_tok, hyp_tok) trail,
    because the trail is what tells you WHICH words a system gets wrong, which is
    far more actionable than the scalar."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    i, j, c, trail = n, m, EditCounts(), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                c.hits += 1
                trail.append(("=", ref[i - 1], hyp[j - 1]))
            else:
                c.subs += 1
                trail.append(("S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            c.dels += 1
            trail.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            c.ins += 1
            trail.append(("I", "", hyp[j - 1]))
            j -= 1
    trail.reverse()
    return c, trail


def wer(ref: str, hyp: str, **kw) -> float:
    return align(normalise(ref, **kw), normalise(hyp, **kw))[0].wer


def corpus_wer(pairs: list[tuple[str, str]], **kw) -> tuple[float, EditCounts]:
    """WER over a corpus is total-edits / total-ref-words, NOT the mean of
    per-utterance WERs, which over-weights short utterances."""
    total = EditCounts()
    for r, h in pairs:
        total = total + align(normalise(r, **kw), normalise(h, **kw))[0]
    return total.wer, total


def term_recall(ref: str, hyp: str, terms: set[str], **kw) -> tuple[int, int]:
    """How many of a target vocabulary present in the reference survive into the
    hypothesis. Plain WER treats "metformin" and "um" as equally valuable; for a
    clinical transcript they are not, so this is tracked separately."""
    r, h = normalise(ref, **kw), normalise(hyp, **kw)
    hyp_counts: dict[str, int] = {}
    for w in h:
        hyp_counts[w] = hyp_counts.get(w, 0) + 1
    found = total = 0
    for w in r:
        if w in terms:
            total += 1
            if hyp_counts.get(w, 0) > 0:
                hyp_counts[w] -= 1
                found += 1
    return found, total


def error_profile(pairs: list[tuple[str, str]], top: int = 25, **kw) -> list[tuple[str, int]]:
    """The reference words a system most often gets wrong. This is the list that
    tells you what to put in a biasing lexicon."""
    bad: dict[str, int] = {}
    for r, h in pairs:
        for op, rt, _ in align(normalise(r, **kw), normalise(h, **kw))[1]:
            if op in ("S", "D") and rt:
                bad[rt] = bad.get(rt, 0) + 1
    return sorted(bad.items(), key=lambda kv: -kv[1])[:top]
