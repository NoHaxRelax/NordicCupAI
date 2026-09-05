"""Wire schema for the survival-simulator payload.

Transcribed from a single screenshot of a stubbed endpoint that returns a
frozen example state (see ASSUMPTIONS.md, A1). Field *shapes* are treated as
likely; field *values* are meaningless placeholders and nothing here should
be tuned to them.

Parsing is deliberately tolerant. Unknown observation types, missing fields
and unknown keys are preserved rather than rejected, so a schema change on
competition day degrades the policy instead of crashing the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Keys we have actually seen. Used only for drift detection, never to reject.
KNOWN_STATE_KEYS = {"game_status", "score", "agent_status"}
KNOWN_AGENT_KEYS = {
    "agent_id", "observations", "energy", "max_energy", "biome", "age",
    "speed", "sprint_speed", "hearing_radius", "vision_angle", "vision_range",
}
KNOWN_OBS_KEYS = {"type", "distance", "angle", "rel_dir", "coords"}
KNOWN_OBS_TYPES = {"tree", "predator", "edge"}


def _f(value: Any) -> float | None:
    """Coerce to float, tolerating nulls and strings from a sloppy encoder."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Observation:
    """One perceived thing.

    Kept as a single flat record rather than a class hierarchy: we have seen
    three `type` values and have no reason to believe that is the full
    vocabulary. `raw` retains everything so an unmodelled field is still
    reachable without a code change.
    """

    type: str
    distance: float | None = None
    angle: float | None = None
    rel_dir: float | None = None
    coords: tuple[tuple[float, float], ...] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_positional(self) -> bool:
        """True when this observation carries a usable polar bearing."""
        return self.distance is not None and self.angle is not None

    @property
    def xy(self) -> tuple[float, float] | None:
        """Egocentric cartesian position: +x along the agent's facing."""
        if not self.is_positional:
            return None
        import math

        return (
            self.distance * math.cos(self.angle),
            self.distance * math.sin(self.angle),
        )

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "Observation":
        coords = payload.get("coords")
        parsed_coords = None
        if isinstance(coords, (list, tuple)):
            points = []
            for point in coords:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = _f(point[0]), _f(point[1])
                    if x is not None and y is not None:
                        points.append((x, y))
            parsed_coords = tuple(points) or None
        return cls(
            type=str(payload.get("type", "unknown")),
            distance=_f(payload.get("distance")),
            angle=_f(payload.get("angle")),
            rel_dir=_f(payload.get("rel_dir")),
            coords=parsed_coords,
            raw=dict(payload),
        )


@dataclass(frozen=True)
class AgentStatus:
    agent_id: int
    observations: tuple[Observation, ...] = ()
    energy: float | None = None
    max_energy: float | None = None
    biome: str | None = None
    age: float | None = None
    speed: float | None = None
    sprint_speed: float | None = None
    hearing_radius: float | None = None
    vision_angle: float | None = None
    vision_range: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def energy_fraction(self) -> float:
        """Energy in [0, 1]. Defaults to 1.0 when unreported, so an absent
        field never makes the policy behave as if it is starving."""
        if self.energy is None or not self.max_energy:
            return 1.0
        return max(0.0, min(1.0, self.energy / self.max_energy))

    def observations_of(self, *types: str) -> tuple[Observation, ...]:
        wanted = set(types)
        return tuple(o for o in self.observations if o.type in wanted)

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "AgentStatus":
        raw_obs = payload.get("observations") or ()
        observations = tuple(
            Observation.parse(o) for o in raw_obs if isinstance(o, dict)
        )
        agent_id = payload.get("agent_id")
        try:
            agent_id = int(agent_id)
        except (TypeError, ValueError):
            agent_id = 0
        biome = payload.get("biome")
        return cls(
            agent_id=agent_id,
            observations=observations,
            energy=_f(payload.get("energy")),
            max_energy=_f(payload.get("max_energy")),
            biome=str(biome) if biome is not None else None,
            age=_f(payload.get("age")),
            speed=_f(payload.get("speed")),
            sprint_speed=_f(payload.get("sprint_speed")),
            hearing_radius=_f(payload.get("hearing_radius")),
            vision_angle=_f(payload.get("vision_angle")),
            vision_range=_f(payload.get("vision_range")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class GameState:
    game_status: str = "unknown"
    score: float | None = None
    agents: tuple[AgentStatus, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_running(self) -> bool:
        return self.game_status == "running"

    def agent(self, agent_id: int) -> AgentStatus | None:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "GameState":
        raw_agents = payload.get("agent_status") or ()
        # A single agent may plausibly arrive unwrapped rather than in a list.
        if isinstance(raw_agents, dict):
            raw_agents = [raw_agents]
        agents = tuple(
            AgentStatus.parse(a) for a in raw_agents if isinstance(a, dict)
        )
        return cls(
            game_status=str(payload.get("game_status", "unknown")),
            score=_f(payload.get("score")),
            agents=agents,
            raw=dict(payload),
        )


def schema_drift(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Report payload content we have no model for.

    Run this on the first real states of the competition. Every entry is
    either a field worth exploiting or a sign that the transcribed schema was
    wrong, and both are worth knowing in minute one rather than hour three.
    """
    drift: dict[str, list[str]] = {}

    unknown_state = sorted(set(payload) - KNOWN_STATE_KEYS)
    if unknown_state:
        drift["state_keys"] = unknown_state
    missing_state = sorted(KNOWN_STATE_KEYS - set(payload))
    if missing_state:
        drift["missing_state_keys"] = missing_state

    agent_keys: set[str] = set()
    obs_keys: set[str] = set()
    obs_types: set[str] = set()
    raw_agents = payload.get("agent_status") or ()
    if isinstance(raw_agents, dict):
        raw_agents = [raw_agents]
    for agent in raw_agents:
        if not isinstance(agent, dict):
            continue
        agent_keys |= set(agent)
        for obs in agent.get("observations") or ():
            if isinstance(obs, dict):
                obs_keys |= set(obs)
                obs_types.add(str(obs.get("type", "unknown")))

    if agent_keys - KNOWN_AGENT_KEYS:
        drift["agent_keys"] = sorted(agent_keys - KNOWN_AGENT_KEYS)
    if obs_keys - KNOWN_OBS_KEYS:
        drift["observation_keys"] = sorted(obs_keys - KNOWN_OBS_KEYS)
    if obs_types - KNOWN_OBS_TYPES:
        drift["observation_types"] = sorted(obs_types - KNOWN_OBS_TYPES)
    return drift
