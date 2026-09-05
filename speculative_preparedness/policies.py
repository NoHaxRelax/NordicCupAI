"""Policies, ordered by how much they assume.

Every belief about game dynamics lives in `PolicyConfig` as a named, tunable
number -- never inline in a rule. That is the whole defence against the
dangerous assumption: when the real game contradicts us, we change a config
value, not the control flow. A guess buried in an `if` is a guess we cannot
find under time pressure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from actions import Action, wrap_angle
from belief import Belief, BeliefSet
from schema import AgentStatus, GameState


@dataclass
class PolicyConfig:
    """Named guesses. Not one of these is known to be true."""

    # -- threat response
    panic_distance: float = 20.0     # flee below this range
    panic_ttc: float = 4.0           # or below this predicted time-to-contact
    flee_confidence: float = 0.15    # ignore threats we barely believe in
    sprint_ttc: float = 2.5          # sprint only when contact is imminent

    # -- energy economy (A6: we assume sprinting costs more than moving)
    sprint_energy_floor: float = 0.25   # never sprint below this energy
    move_energy_floor: float = 0.05     # near-total stop below this
    rest_energy_ceiling: float = 0.9    # stop resting once topped up

    # -- exploration
    cruise_throttle: float = 0.6
    wander_turn: float = 0.4         # radians of random walk per step
    wander_persistence: float = 0.85  # correlation between successive turns

    # -- obstacles
    tree_clearance: float = 4.0
    edge_clearance: float = 8.0
    avoid_gain: float = 1.5


class Policy:
    """Interface. `act` returns one Action per agent_id present in the state."""

    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, state: GameState, dt: float = 1.0) -> dict[int, Action]:
        raise NotImplementedError


class NoOpPolicy(Policy):
    """The score floor. Every other policy is measured against this."""

    name = "noop"

    def act(self, state: GameState, dt: float = 1.0) -> dict[int, Action]:
        return {a.agent_id: Action() for a in state.agents}


class ReactivePolicy(Policy):
    """Level 0: a pure function of the current frame. No memory, no model.

    This is the policy that survives being wrong about everything, because it
    assumes nothing beyond "predators are bad and walls are solid". It is the
    one to have running in the first ten minutes of the competition.
    """

    name = "reactive"

    def __init__(self, config: PolicyConfig | None = None, seed: int = 0) -> None:
        self.config = config or PolicyConfig()
        self._rng = random.Random(seed)
        self._wander: dict[int, float] = {}

    def reset(self) -> None:
        self._wander.clear()

    def act(self, state: GameState, dt: float = 1.0) -> dict[int, Action]:
        return {a.agent_id: self._act_one(a) for a in state.agents}

    def _act_one(self, status: AgentStatus) -> Action:
        cfg = self.config
        energy = status.energy_fraction

        threat_x = threat_y = 0.0
        nearest = math.inf
        for obs in status.observations_of("predator"):
            point = obs.xy
            if point is None:
                continue
            d = max(obs.distance or 1e-3, 1e-3)
            nearest = min(nearest, d)
            threat_x += point[0] / (d * d)
            threat_y += point[1] / (d * d)

        avoid = self._avoidance(status)

        if nearest <= cfg.panic_distance and (threat_x or threat_y):
            # Flee directly away from the weighted threat, blended with
            # obstacle avoidance so we do not reverse into a tree.
            flee = math.atan2(-threat_y, -threat_x)
            heading = self._blend(flee, avoid, cfg.avoid_gain)
            sprint = nearest <= cfg.panic_distance * 0.5 and energy >= cfg.sprint_energy_floor
            throttle = 1.0 if energy > cfg.move_energy_floor else 0.2
            return Action(turn=heading, throttle=throttle, sprint=sprint)

        if energy <= cfg.move_energy_floor:
            return Action()  # conserve; assumes idling is cheaper (A6)

        return Action(
            turn=self._blend(self._wander_turn(status.agent_id), avoid, cfg.avoid_gain),
            throttle=cfg.cruise_throttle,
            sprint=False,
        )

    def _avoidance(self, status: AgentStatus) -> tuple[float, float]:
        """Repulsion vector from trees and edges, in egocentric cartesian."""
        cfg = self.config
        ax = ay = 0.0
        for obs in status.observations_of("tree"):
            point = obs.xy
            if point is None or obs.distance is None:
                continue
            if obs.distance > cfg.tree_clearance * 3:
                continue
            d = max(obs.distance, 1e-3)
            ax -= point[0] / (d * d)
            ay -= point[1] / (d * d)
        for obs in status.observations_of("edge"):
            for px, py in obs.coords or ():
                d = max(math.hypot(px, py), 1e-3)
                if d > cfg.edge_clearance * 3:
                    continue
                ax -= px / (d * d)
                ay -= py / (d * d)
        return ax, ay

    def _blend(self, desired: float, avoid: tuple[float, float], gain: float) -> float:
        ax, ay = avoid
        if ax == 0.0 and ay == 0.0:
            return wrap_angle(desired)
        x = math.cos(desired) + gain * ax
        y = math.sin(desired) + gain * ay
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return wrap_angle(desired)
        return wrap_angle(math.atan2(y, x))

    def _wander_turn(self, agent_id: int) -> float:
        cfg = self.config
        prev = self._wander.get(agent_id, 0.0)
        step = self._rng.uniform(-cfg.wander_turn, cfg.wander_turn)
        value = cfg.wander_persistence * prev + (1 - cfg.wander_persistence) * step
        self._wander[agent_id] = value
        return value


class BeliefPolicy(ReactivePolicy):
    """Level 1: the same rules, driven by the belief state instead of the
    current frame. Reacts to remembered predators and to closing speed, which
    the reactive policy cannot see.

    It still assumes nothing about game *rules* -- only that what it saw a
    moment ago is probably still there.
    """

    name = "belief"

    def __init__(self, config: PolicyConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.beliefs = BeliefSet()
        self._last: dict[int, Action] = {}

    def reset(self) -> None:
        super().reset()
        self.beliefs.reset()
        self._last.clear()

    def act(self, state: GameState, dt: float = 1.0) -> dict[int, Action]:
        self.beliefs.update(state.agents, dt=dt, commanded=self._last)
        out = {}
        for status in state.agents:
            out[status.agent_id] = self._act_belief(self.beliefs[status.agent_id], status)
        self._last = out
        return out

    def _act_belief(self, belief: Belief, status: AgentStatus) -> Action:
        cfg = self.config
        energy = status.energy_fraction
        threat = belief.nearest_threat()
        avoid = self._avoidance(status)

        if threat is not None and threat.confidence >= cfg.flee_confidence:
            ttc = threat.time_to_contact()
            if threat.distance <= cfg.panic_distance or ttc <= cfg.panic_ttc:
                tx, ty = belief.threat_vector()
                flee = math.atan2(-ty, -tx) if (tx or ty) else wrap_angle(threat.angle + math.pi)
                heading = self._blend(flee, avoid, cfg.avoid_gain)
                # Distance is the fallback trigger: on a first sighting we
                # have no velocity estimate yet, so ttc is infinite and would
                # veto sprinting even from a predator at arm's length.
                urgent = ttc <= cfg.sprint_ttc or threat.distance <= cfg.panic_distance * 0.5
                sprint = urgent and energy >= cfg.sprint_energy_floor
                throttle = 1.0 if energy > cfg.move_energy_floor else 0.2
                return Action(turn=heading, throttle=throttle, sprint=sprint)

        if energy <= cfg.move_energy_floor:
            return Action()

        return Action(
            turn=self._blend(self._wander_turn(status.agent_id), avoid, cfg.avoid_gain),
            throttle=cfg.cruise_throttle,
            sprint=False,
        )


POLICIES = {p.name: p for p in (NoOpPolicy, ReactivePolicy, BeliefPolicy)}


def build(name: str, **kwargs) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    return POLICIES[name](**kwargs) if name != "noop" else NoOpPolicy()
