"""Capture and replay of every exchange with the real server.

This is the highest-value module in the package and the only one whose value
does not depend on a single guess being right.

From the first request of the competition, every (state, belief, action,
wire payload, score) tuple is appended to a JSONL file. That log is the raw
material for everything afterwards: measuring whether a change helped,
fitting a forward model, diagnosing a schema surprise, or reconstructing what
the policy was thinking when it walked into a predator.

Data you did not capture in hour one cannot be recovered in hour three.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from schema import GameState, schema_drift


@dataclass
class Frame:
    """One request/response exchange."""

    seq: int
    wall_time: float
    state: dict[str, Any]
    actions: dict[str, Any] = field(default_factory=dict)
    wire: Any = None
    belief: list[dict[str, Any]] = field(default_factory=list)
    drift: dict[str, list[str]] = field(default_factory=dict)
    policy: str = ""
    codec: str = ""
    note: str = ""


class Capture:
    """Append-only JSONL writer. Flushes every frame -- a crashed process
    that loses its buffer loses exactly the data we most wanted."""

    def __init__(self, path: str | Path, policy: str = "", codec: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._seq = 0
        self.policy = policy
        self.codec = codec
        self.drift_seen: dict[str, list[str]] = {}

    def record(
        self,
        payload: dict[str, Any],
        state: GameState | None = None,
        actions: dict | None = None,
        wire: Any = None,
        belief: list[dict] | None = None,
        note: str = "",
    ) -> Frame:
        self._seq += 1
        drift = schema_drift(payload)
        if drift:
            # Report each novel drift once; a per-frame warning is noise.
            novel = {k: v for k, v in drift.items() if self.drift_seen.get(k) != v}
            if novel:
                self.drift_seen.update(novel)
                print(f"[capture] SCHEMA DRIFT at frame {self._seq}: {novel}", flush=True)
        frame = Frame(
            seq=self._seq,
            wall_time=time.time(),
            state=payload,
            actions={str(k): asdict(v) for k, v in (actions or {}).items()},
            wire=wire,
            belief=belief or [],
            drift=drift,
            policy=self.policy,
            codec=self.codec,
            note=note,
        )
        self._fh.write(json.dumps(asdict(frame), default=str) + "\n")
        self._fh.flush()
        return frame

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Capture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_frames(path: str | Path) -> Iterator[Frame]:
    """Stream frames back off disk, skipping corrupt lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield Frame(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue


def replay(path: str | Path, policy) -> dict[str, Any]:
    """Re-run a policy over captured states.

    This is offline evaluation against *real* data, and it is the only
    measurement in this package that is not contaminated by our guesses.
    It cannot tell you the score a new policy would have earned -- the game
    would have gone somewhere else -- but it does tell you where the new
    policy diverges from the old one, which is what you want when deciding
    whether a change is safe to ship mid-competition.
    """
    policy.reset()
    divergences = 0
    total = 0
    examples = []
    for frame in read_frames(path):
        state = GameState.parse(frame.state)
        if not state.agents:
            continue
        new = policy.act(state)
        for agent_id, action in new.items():
            old = frame.actions.get(str(agent_id))
            if old is None:
                continue
            total += 1
            turn_delta = abs(action.turn - float(old.get("turn", 0.0)))
            throttle_delta = abs(action.throttle - float(old.get("throttle", 0.0)))
            if turn_delta > 0.3 or throttle_delta > 0.25:
                divergences += 1
                if len(examples) < 5:
                    examples.append(
                        {"seq": frame.seq, "agent_id": agent_id, "old": old, "new": asdict(action)}
                    )
    return {
        "frames": total,
        "divergences": divergences,
        "divergence_rate": round(divergences / total, 3) if total else 0.0,
        "examples": examples,
    }


def transitions(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield (state, action, next_state) triples.

    This is the training set for a learned forward model. It exists only once
    real data is flowing -- which is precisely why the model is not built in
    advance. See README.md, "What is deliberately not built".
    """
    previous: Frame | None = None
    for frame in read_frames(path):
        if previous is not None:
            yield {
                "state": previous.state,
                "actions": previous.actions,
                "next_state": frame.state,
                "dt": round(frame.wall_time - previous.wall_time, 4),
            }
        previous = frame
