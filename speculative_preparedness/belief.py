"""Belief state: persistence and motion estimation under partial observation.

The agent sees a cone (`vision_range`, `vision_angle`) plus a presumed
omnidirectional `hearing_radius`. Everything outside that is unobserved, not
absent. A stateless policy therefore forgets a predator the instant it leaves
the cone, which is exactly when it is most dangerous.

This module is the low-assumption half of a world model. It assumes only that
objects persist and move continuously between frames -- not how the game
scores, feeds, or kills. It needs no action format and no simulator, so it is
built from the observation stream alone and is correct even if every guess in
ASSUMPTIONS.md is wrong.

The high-assumption half -- a learned forward model p(next | belief, action)
-- is deliberately NOT built here. It cannot be trained before the action
format exists. See README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from actions import Action, wrap_angle
from schema import AgentStatus, Observation

# Types whose position is a point we can track. `edge` is a line segment and
# is handled separately.
TRACKABLE = ("predator", "tree")


@dataclass
class Track:
    """A single remembered object in the agent's egocentric frame.

    Position is cartesian with +x along the agent's current facing, which
    means every track must be transformed when the agent moves.
    """

    track_id: int
    type: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    last_seen: float = 0.0     # seconds since this track was observed
    hits: int = 1              # times associated with an observation
    rel_dir: float | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)

    @property
    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    @property
    def confidence(self) -> float:
        """Decays with staleness; grows with repeat sightings.

        A track seen once and lost is weak evidence; one seen ten times and
        lost a moment ago is strong. Half-life is deliberately generous
        because losing a predator is far more costly than imagining one.
        """
        freshness = math.exp(-self.last_seen / BeliefConfig.track_half_life)
        support = 1.0 - 0.5 ** self.hits
        return freshness * support

    def closing_speed(self) -> float:
        """Rate of decrease of range. Positive means it is closing on us."""
        d = self.distance
        if d < 1e-6:
            return 0.0
        return -(self.x * self.vx + self.y * self.vy) / d

    def time_to_contact(self) -> float:
        """Seconds until range hits zero at the current closing speed."""
        closing = self.closing_speed()
        if closing <= 1e-6:
            return math.inf
        return self.distance / closing


class BeliefConfig:
    """Tunables. Class-level so `Track.confidence` can reach them without
    threading config through every record."""

    track_half_life = 3.0        # seconds; confidence e-folds at this age
    association_gate = 8.0       # max distance to match obs to existing track
    velocity_smoothing = 0.5     # EMA factor on velocity estimates
    forget_below = 0.02          # drop tracks under this confidence
    max_tracks = 200


@dataclass
class Belief:
    """Per-agent belief. One instance per `agent_id`, updated every frame."""

    agent_id: int
    tracks: list[Track] = field(default_factory=list)
    edges: tuple[Observation, ...] = ()
    last_status: AgentStatus | None = None
    elapsed: float = 0.0
    _next_id: int = 1

    def update(
        self,
        status: AgentStatus,
        dt: float = 1.0,
        commanded: Action | None = None,
    ) -> "Belief":
        """Fold one observation frame into the belief.

        `commanded` is the action we asked for last frame. It is used only as
        a motion prior -- we have no confirmation it was executed, and the
        association step corrects it when it was not (ASSUMPTIONS.md, A5).
        """
        self.elapsed += dt
        self._predict(status, dt, commanded)
        self._associate(status, dt)
        self._prune()
        self.edges = status.observations_of("edge")
        self.last_status = status
        return self

    def _predict(self, status: AgentStatus, dt: float, commanded: Action | None) -> None:
        """Advance tracks by their velocity, then subtract our own motion."""
        turn = 0.0
        forward = 0.0
        if commanded is not None:
            turn = commanded.turn
            speed = status.speed or 0.0
            if commanded.sprint and status.sprint_speed:
                speed = status.sprint_speed
            forward = commanded.throttle * speed * dt

        cos_t, sin_t = math.cos(-turn), math.sin(-turn)
        for track in self.tracks:
            # Object's own motion.
            x = track.x + track.vx * dt
            y = track.y + track.vy * dt
            # Our translation along the (pre-turn) facing.
            x -= forward
            # Our rotation, applied to the whole frame.
            track.x = x * cos_t - y * sin_t
            track.y = x * sin_t + y * cos_t
            vx, vy = track.vx, track.vy
            track.vx = vx * cos_t - vy * sin_t
            track.vy = vx * sin_t + vy * cos_t
            track.last_seen += dt
            if track.rel_dir is not None:
                track.rel_dir = wrap_angle(track.rel_dir - turn)

    def _associate(self, status: AgentStatus, dt: float) -> None:
        """Greedy nearest-neighbour matching of observations to tracks."""
        unmatched = [t for t in self.tracks if t.type in TRACKABLE]
        for obs in status.observations_of(*TRACKABLE):
            point = obs.xy
            if point is None:
                continue
            ox, oy = point
            best, best_d = None, BeliefConfig.association_gate
            for track in unmatched:
                if track.type != obs.type:
                    continue
                d = math.hypot(track.x - ox, track.y - oy)
                if d < best_d:
                    best, best_d = track, d
            if best is None:
                self._spawn(obs, ox, oy)
                continue
            unmatched.remove(best)
            self._refine(best, obs, ox, oy, dt)

    def _spawn(self, obs: Observation, ox: float, oy: float) -> None:
        if len(self.tracks) >= BeliefConfig.max_tracks:
            return
        self.tracks.append(
            Track(
                track_id=self._next_id,
                type=obs.type,
                x=ox,
                y=oy,
                last_seen=0.0,
                hits=1,
                rel_dir=obs.rel_dir,
                raw=obs.raw,
            )
        )
        self._next_id += 1

    def _refine(self, track: Track, obs: Observation, ox: float, oy: float, dt: float) -> None:
        if dt > 1e-6:
            mvx = (ox - track.x) / dt
            mvy = (oy - track.y) / dt
            a = BeliefConfig.velocity_smoothing
            track.vx = (1 - a) * track.vx + a * mvx
            track.vy = (1 - a) * track.vy + a * mvy
        track.x, track.y = ox, oy
        track.last_seen = 0.0
        track.hits += 1
        if obs.rel_dir is not None:
            track.rel_dir = obs.rel_dir
        track.raw = obs.raw

    def _prune(self) -> None:
        self.tracks = [t for t in self.tracks if t.confidence >= BeliefConfig.forget_below]

    # -- queries the policy actually uses -------------------------------

    def of_type(self, obs_type: str) -> list[Track]:
        return [t for t in self.tracks if t.type == obs_type]

    def threats(self) -> list[Track]:
        """Predators, most urgent first: soonest contact, then nearest."""
        return sorted(
            self.of_type("predator"),
            key=lambda t: (t.time_to_contact(), t.distance),
        )

    def nearest_threat(self) -> Track | None:
        threats = self.threats()
        return threats[0] if threats else None

    def threat_vector(self) -> tuple[float, float]:
        """Confidence-weighted sum of directions to known predators.

        Summing rather than taking the nearest matters when two predators
        flank: fleeing directly from one can run into the other.
        """
        vx = vy = 0.0
        for track in self.of_type("predator"):
            d = max(track.distance, 1e-3)
            weight = track.confidence / d
            vx += (track.x / d) * weight
            vy += (track.y / d) * weight
        return vx, vy

    def snapshot(self) -> dict:
        """Flat record for logging and, later, as a feature vector source."""
        threat = self.nearest_threat()
        return {
            "agent_id": self.agent_id,
            "elapsed": round(self.elapsed, 3),
            "n_tracks": len(self.tracks),
            "n_predators": len(self.of_type("predator")),
            "n_trees": len(self.of_type("tree")),
            "threat_distance": round(threat.distance, 3) if threat else None,
            "threat_ttc": (
                round(threat.time_to_contact(), 3)
                if threat and math.isfinite(threat.time_to_contact())
                else None
            ),
            "threat_confidence": round(threat.confidence, 3) if threat else None,
            "energy_fraction": (
                round(self.last_status.energy_fraction, 3) if self.last_status else None
            ),
        }


class BeliefSet:
    """Beliefs for every agent, created on demand as agents appear."""

    def __init__(self) -> None:
        self._beliefs: dict[int, Belief] = {}

    def update(self, agents, dt: float = 1.0, commanded: dict[int, Action] | None = None):
        commanded = commanded or {}
        for status in agents:
            belief = self._beliefs.get(status.agent_id)
            if belief is None:
                belief = Belief(agent_id=status.agent_id)
                self._beliefs[status.agent_id] = belief
            belief.update(status, dt=dt, commanded=commanded.get(status.agent_id))
        return self

    def __getitem__(self, agent_id: int) -> Belief:
        return self._beliefs[agent_id]

    def __iter__(self):
        return iter(self._beliefs.values())

    def __len__(self) -> int:
        return len(self._beliefs)

    def reset(self) -> None:
        self._beliefs.clear()
