"""PriMock57 as a dev set.

57 mock primary-care consultations, CC BY 4.0, with three things that matter:
the audio is split into a doctor channel and a patient channel (so speaker
labels are ground truth, not an annotation to be trusted), transcripts are
utterance-level with timings, and each consultation has the clinician's own
written note.

That covers every shape the challenge brief could take:
    transcription        -> reference text per consultation
    diarization          -> per-channel speaker truth
    note generation      -> notes/*.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .textgrid import Utterance, merge_speakers, read_textgrid

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "primock57"
_RE_ID = re.compile(r"(day\d+_consultation\d+)_(doctor|patient)", re.I)


@dataclass
class Consultation:
    cid: str                                   # e.g. "day1_consultation01"
    utterances: list[Utterance] = field(default_factory=list)
    note: dict | None = None
    audio: dict[str, Path] = field(default_factory=dict)   # speaker -> wav

    @property
    def duration(self) -> float:
        return max((u.end for u in self.utterances), default=0.0)

    def reference(self, *, speaker_tags: bool = False, keep_unsure: bool = True,
                  drop_unintelligible: bool = True) -> str:
        """The gold transcript.

        `drop_unintelligible` removes utterances the transcriber could not make
        out at all. Scoring a system against audio no human could decode
        measures noise, not the system, so it is off by default.
        """
        parts = []
        for u in self.utterances:
            if drop_unintelligible and u.unintelligible and not u.clean(keep_unsure).strip():
                continue
            txt = u.clean(keep_unsure)
            if not txt:
                continue
            parts.append(f"[{u.speaker}] {txt}" if speaker_tags else txt)
        return " ".join(parts)

    def by_speaker(self) -> dict[str, str]:
        out: dict[str, list[str]] = {}
        for u in self.utterances:
            t = u.clean()
            if t:
                out.setdefault(u.speaker, []).append(t)
        return {k: " ".join(v) for k, v in out.items()}

    @property
    def stats(self) -> dict:
        spk = {}
        for u in self.utterances:
            s = spk.setdefault(u.speaker, {"utts": 0, "secs": 0.0, "words": 0})
            s["utts"] += 1
            s["secs"] += u.duration
            s["words"] += len(u.clean().split())
        return {
            "cid": self.cid, "duration_s": round(self.duration, 1),
            "utterances": len(self.utterances),
            "words": sum(len(u.clean().split()) for u in self.utterances),
            "unintelligible": sum(u.unintelligible for u in self.utterances),
            "unsure": sum(u.unsure for u in self.utterances),
            "speakers": {k: {"utts": v["utts"], "secs": round(v["secs"], 1), "words": v["words"]}
                         for k, v in spk.items()},
            "has_note": self.note is not None,
            "has_audio": sorted(self.audio),
        }


def load(root: str | Path = DEFAULT_ROOT) -> list[Consultation]:
    root = Path(root)
    tdir, ndir, adir = root / "transcripts", root / "notes", root / "audio"
    if not tdir.is_dir():
        raise FileNotFoundError(
            f"no transcripts at {tdir}. Clone it first:\n"
            f"  git clone https://github.com/babylonhealth/primock57.git {root}\n"
            f"  cd {root} && git lfs pull"
        )

    cons: dict[str, Consultation] = {}
    for tg in sorted(tdir.glob("*.TextGrid")):
        m = _RE_ID.search(tg.stem)
        if not m:
            continue
        cid, spk = m.group(1).lower(), m.group(2).title()
        c = cons.setdefault(cid, Consultation(cid))
        c.utterances = merge_speakers(c.utterances, read_textgrid(tg, speaker=spk))

    for c in cons.values():
        nf = ndir / f"{c.cid}.json"
        if nf.is_file():
            try:
                c.note = json.loads(nf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        for spk in ("doctor", "patient"):
            wav = adir / f"{c.cid}_{spk}.wav"
            # an LFS pointer is a few hundred bytes; real audio is megabytes
            if wav.is_file() and wav.stat().st_size > 4096:
                c.audio[spk.title()] = wav

    return [cons[k] for k in sorted(cons)]


def split(consultations: list[Consultation], dev_frac: float = 0.3) -> tuple[list, list]:
    """Deterministic split by consultation id. Never split within a consultation:
    the same two voices appearing in both halves would flatter any adaptation."""
    n_dev = max(1, int(len(consultations) * dev_frac))
    ordered = sorted(consultations, key=lambda c: c.cid)
    return ordered[n_dev:], ordered[:n_dev]        # (train, dev)
