"""Canonical action representation and the wire codec seam.

The response format is unknown (ASSUMPTIONS.md, A2): the endpoint accepts any
body and no-ops it, so it cannot be probed. Policies therefore emit an
`Action` of *intent*, and a `Codec` translates intent to whatever the wire
turns out to want. On competition day the only thing that should need to
change is which codec is selected, not any policy code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


def wrap_angle(theta: float) -> float:
    """Normalise to (-pi, pi]."""
    return -((-theta + math.pi) % (2 * math.pi) - math.pi)


@dataclass(frozen=True)
class Action:
    """One timestep of intent for one agent.

    Deliberately expressed in quantities the observation schema already
    implies (bearings in radians, a speed we know has a sprint variant)
    rather than in a guessed verb vocabulary.
    """

    turn: float = 0.0        # radians, agent-relative, sign matches `angle`
    throttle: float = 0.0    # 0..1 fraction of `speed`
    sprint: bool = False     # request `sprint_speed` instead of `speed`
    verb: str | None = None  # optional discrete intent, vocabulary unknown

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn", wrap_angle(self.turn))
        object.__setattr__(self, "throttle", max(0.0, min(1.0, self.throttle)))

    @property
    def is_idle(self) -> bool:
        return self.throttle == 0.0 and self.turn == 0.0 and self.verb is None


class Codec(Protocol):
    """Translates intent to a wire payload."""

    name: str

    def encode(self, actions: dict[int, Action]) -> Any: ...


@dataclass
class ContinuousCodec:
    """Per-agent continuous control. Matches the shape of `agent_status`:
    a list of records keyed by `agent_id`."""

    name: str = "continuous"
    key: str = "actions"

    def encode(self, actions: dict[int, Action]) -> dict[str, Any]:
        return {
            self.key: [
                {
                    "agent_id": agent_id,
                    "turn": action.turn,
                    "throttle": action.throttle,
                    "sprint": action.sprint,
                }
                for agent_id, action in sorted(actions.items())
            ]
        }


@dataclass
class DiscreteVerbCodec:
    """Race-car style: a flat list of verb strings.

    The 2025 race-car task used `{"actions": ["ACCELERATE", ...]}`, so the
    organisers have shipped this shape before. Continuous intent is quantised
    onto the nearest verbs.
    """

    name: str = "discrete"
    key: str = "actions"
    turn_deadzone: float = math.pi / 12
    move_threshold: float = 0.1
    vocabulary: tuple[str, ...] = (
        "MOVE", "SPRINT", "TURN_LEFT", "TURN_RIGHT", "NOTHING",
    )

    def encode(self, actions: dict[int, Action]) -> dict[str, Any]:
        verbs: list[str] = []
        for _, action in sorted(actions.items()):
            if action.turn > self.turn_deadzone:
                verbs.append("TURN_LEFT")
            elif action.turn < -self.turn_deadzone:
                verbs.append("TURN_RIGHT")
            if action.throttle >= self.move_threshold:
                verbs.append("SPRINT" if action.sprint else "MOVE")
            if action.verb:
                verbs.append(action.verb.upper())
        return {self.key: verbs or ["NOTHING"]}


@dataclass
class HeadingCodec:
    """Absolute-bearing control, for a server that wants a target direction
    rather than a delta. Requires the caller to track heading; here the
    delta is emitted under a different key set."""

    name: str = "heading"
    key: str = "actions"

    def encode(self, actions: dict[int, Action]) -> dict[str, Any]:
        return {
            self.key: [
                {
                    "agent_id": agent_id,
                    "direction": action.turn,
                    "speed": action.throttle,
                    "sprinting": action.sprint,
                }
                for agent_id, action in sorted(actions.items())
            ]
        }


@dataclass
class NoOpCodec:
    """Emits nothing actionable. Used to measure the do-nothing score floor,
    which is the control every other policy must beat."""

    name: str = "noop"

    def encode(self, actions: dict[int, Action]) -> dict[str, Any]:
        return {"actions": []}


CODECS: dict[str, Codec] = {
    c.name: c for c in (ContinuousCodec(), DiscreteVerbCodec(), HeadingCodec(), NoOpCodec())
}


def get_codec(name: str) -> Codec:
    if name not in CODECS:
        raise KeyError(f"unknown codec {name!r}; have {sorted(CODECS)}")
    return CODECS[name]
