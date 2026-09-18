"""Local, short-lived directional crowd memory; no map or absolute positions."""

from dataclasses import dataclass, field
import math

from pydantic import BaseModel, ConfigDict, Field

from models.nikolaj.policy_inputs import RememberedNeighbor


class CrowdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(strict=True)
    memory_seconds: float = Field(ge=0)
    neighbor_radius: float = Field(gt=0)
    max_remembered_neighbors: int = Field(ge=1, strict=True)
    direction_bins: int = Field(ge=4, le=360, strict=True)
    angular_spread: float = Field(gt=0, le=math.pi)
    full_strength_neighbors: float = Field(gt=0)
    minimum_contrast: float = Field(ge=0, le=1)
    tie_score_fraction: float = Field(ge=0, le=1)
    exploration_strength: float = Field(ge=0, le=1)
    foraging_strength: float = Field(ge=0, le=1)


def wrap(angle):
    return (angle + math.pi) % math.tau - math.pi


@dataclass
class Sighting:
    angle: float
    distance: float
    seen_at: float


@dataclass
class AgentCrowdMemory:
    last_time: float
    last_age: float
    last_turn: float = 0.0
    sightings: dict[int, Sighting] = field(default_factory=dict)


class CrowdTracker:
    def __init__(self, config: CrowdConfig):
        self.config = config
        self.memories: dict[int, AgentCrowdMemory] = {}

    def reset(self):
        self.memories.clear()

    def prune(self, living_ids):
        self.memories = {key: value for key, value in self.memories.items() if key in living_ids}

    def _neighbors(self, memory, now):
        return tuple(RememberedNeighbor(
            agent_id=agent_id, angle=sighting.angle, distance=sighting.distance,
            weight=max(0.0, 1 - (now - sighting.seen_at) / self.config.memory_seconds)
                   * max(0.0, 1 - sighting.distance / self.config.neighbor_radius),
        ) for agent_id, sighting in sorted(memory.sightings.items()))

    def observe(self, state, now):
        config = self.config
        if not config.enabled or config.memory_seconds == 0:
            self.reset()
            return ()
        agent_id, age = state["agent_id"], state["age"]
        memory = self.memories.get(agent_id)
        if memory and (now < memory.last_time or age < memory.last_age):
            memory = None
        if memory is None:
            memory = AgentCrowdMemory(now, age)
            self.memories[agent_id] = memory
        elif now == memory.last_time:
            # Retrying the same tick must not rotate or refresh old sightings.
            return self._neighbors(memory, now)
        else:
            for sighting in memory.sightings.values():
                sighting.angle = wrap(sighting.angle - memory.last_turn)

        memory.sightings = {
            key: sighting for key, sighting in memory.sightings.items()
            if now < sighting.seen_at + config.memory_seconds
            and not math.isclose(now, sighting.seen_at + config.memory_seconds, rel_tol=0, abs_tol=1e-9)
        }
        sightings = sorted((obs for obs in state["observations"]
                            if obs["type"] == "Agent" and "id" in obs and obs["id"] != agent_id),
                           key=lambda obs: (obs["id"], obs["distance"], obs["angle"]))
        seen = set()
        for observation in sightings:
            neighbor_id = observation["id"]
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            distance = observation["distance"]
            if 0 <= distance < config.neighbor_radius:
                # Refresh an existing ID, rather than counting each frame again.
                memory.sightings[neighbor_id] = Sighting(wrap(observation["angle"]), distance, now)
            else:
                # A fresh distant sighting supersedes its old nearby location.
                memory.sightings.pop(neighbor_id, None)
        memory.last_time, memory.last_age = now, age
        neighbors = self._neighbors(memory, now)
        if len(neighbors) > config.max_remembered_neighbors:
            keep = {neighbor.agent_id for neighbor in sorted(
                neighbors, key=lambda neighbor: (-neighbor.weight, neighbor.agent_id)
            )[:config.max_remembered_neighbors]}
            memory.sightings = {key: value for key, value in memory.sightings.items() if key in keep}
            neighbors = self._neighbors(memory, now)
        return neighbors

    def remember_turn(self, agent_id, turn):
        if agent_id in self.memories:
            self.memories[agent_id].last_turn = turn


def crowd_direction(neighbors, preferred, config, agent_id):
    """Return an open bearing and confidence-scaled crowd strength, or None.

    Smooth each sighting across nearby directions, then choose a low-load
    direction. Ties favor the current goal; agent IDs break left/right symmetry.
    This is a crowd heuristic, not a calibrated encounter probability.
    """
    if not config.enabled or not neighbors:
        return None
    effective_count = sum(neighbor.weight for neighbor in neighbors)
    if effective_count <= 0:
        return None
    candidates = [wrap(math.tau * index / config.direction_bins) for index in range(config.direction_bins)]
    candidates.append(wrap(preferred))
    loads = [sum(neighbor.weight * math.exp(
        -0.5 * (wrap(angle - neighbor.angle) / config.angular_spread) ** 2
    ) for neighbor in neighbors) for angle in candidates]
    low, high = min(loads), max(loads)
    contrast = (high - low) / high if high > 0 else 0
    if contrast <= config.minimum_contrast:
        return None
    tolerance = config.tie_score_fraction * (high - low)
    open_directions = [angle for angle, load in zip(candidates, loads) if load <= low + tolerance]
    side = 1 if agent_id % 2 == 0 else -1
    direction = min(open_directions, key=lambda angle: (
        round(abs(wrap(angle - preferred)), 12),
        -side * wrap(angle - preferred),
    ))
    strength = min(1.0, effective_count / config.full_strength_neighbors) * contrast
    return direction, strength
