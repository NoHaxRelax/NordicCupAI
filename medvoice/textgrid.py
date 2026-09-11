"""Praat TextGrid reader, scoped to what PriMock57 actually uses.

PriMock57 ships one IntervalTier per file, named "Doctor" or "Patient", whose
intervals are utterances with timings. A general TextGrid library would be
overkill and one more thing to install on competition day, so this reads the
subset we need and refuses anything it does not understand rather than guessing.

Transcription tags in the text:
    <UNIN/>              an unintelligible stretch of audio
    <UNSURE>...</UNSURE> the transcriber was not confident of the words
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NUM = r"([-\d.eE+]+)"
_RE_XMIN = re.compile(rf"^\s*xmin\s*=\s*{_NUM}", re.M)
_RE_XMAX = re.compile(rf"^\s*xmax\s*=\s*{_NUM}", re.M)
_RE_NAME = re.compile(r'^\s*name\s*=\s*"(.*)"', re.M)
_RE_INTERVAL = re.compile(
    rf'intervals\s*\[\d+\]:\s*'
    rf'xmin\s*=\s*{_NUM}\s*'
    rf'xmax\s*=\s*{_NUM}\s*'
    rf'text\s*=\s*"((?:[^"]|"")*)"',
    re.S,
)

UNIN = "<UNIN/>"
_RE_UNSURE = re.compile(r"<UNSURE>(.*?)</UNSURE>", re.S)
_RE_ANYTAG = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class Utterance:
    speaker: str
    start: float
    end: float
    text: str          # raw, tags intact

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def unintelligible(self) -> bool:
        return UNIN in self.text

    @property
    def unsure(self) -> bool:
        return "<UNSURE>" in self.text

    def clean(self, keep_unsure: bool = True) -> str:
        """Strip tags. `<UNSURE>` wording is kept by default because it is a real
        transcription of real speech; only the marker is removed."""
        t = _RE_UNSURE.sub(r"\1" if keep_unsure else " ", self.text)
        return " ".join(_RE_ANYTAG.sub(" ", t).split())


def read_textgrid(path: str | Path, speaker: str | None = None) -> list[Utterance]:
    """Parse one TextGrid into its non-empty utterances, in time order.

    The tier name is NOT a reliable speaker label: most PriMock57 files name
    their tier "Speaker" and only a handful say "Doctor" or "Patient". The
    filename suffix is authoritative, so callers pass `speaker` explicitly and
    the tier name is only a fallback.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    if 'Object class = "TextGrid"' not in raw:
        raise ValueError(f"not a TextGrid: {path}")

    if speaker is None:
        names = [n.strip() for n in _RE_NAME.findall(raw) if n.strip()]
        tier = names[0] if names else ""
        speaker = tier if tier.lower() in ("doctor", "patient") else Path(path).stem.rsplit("_", 1)[-1].title()

    out: list[Utterance] = []
    for xmin, xmax, text in _RE_INTERVAL.findall(raw):
        text = text.replace('""', '"').strip()
        if not text:
            continue          # silence between utterances
        out.append(Utterance(speaker, float(xmin), float(xmax), text))
    out.sort(key=lambda u: u.start)
    return out


def merge_speakers(*tiers: list[Utterance]) -> list[Utterance]:
    """Interleave several speakers' utterances into one time-ordered stream."""
    merged = [u for tier in tiers for u in tier]
    merged.sort(key=lambda u: (u.start, u.speaker))
    return merged
