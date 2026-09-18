"""Prepare local vectors using only information supplied in observations."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RelativeTarget:
    # x is forward, y is at +pi/2 relative to the agent's current facing.
    vector: tuple[float, float]
    distance: float
    angle: float
    ripeness: float | None = None


@dataclass(frozen=True)
class SectionHint:
    # Direction toward the assigned center in the current facing frame.
    vector: tuple[float, float]
    strength: float


@dataclass(frozen=True)
class ExplorationHint:
    # Planned displacement in current facing coordinates; survival overrides it.
    vector: tuple[float, float]
    role: str
    objective: str
    obstacle_margin: float = 4.0
    # None preserves the ordinary walking cap. Explicit requests still pass
    # through the policy's physical limits and post-action energy accounting.
    max_speed: float | None = None
    minimum_energy_reserve: float = 75.0
    # Facing and travel are independent controls; radians relative to facing.
    look_direction: float | None = None
    # Healthy scouts can keep a route while collecting only nearby fruit.
    # None keeps the full local food priority for residents and hungry agents.
    food_distance_limit: float | None = None


@dataclass(frozen=True)
class HarvestHint:
    vector: tuple[float, float] | None
    track_id: int | None
    estimated_energy: float
    waiting: bool
    survey: bool = False
    look_direction: float | None = None
    # A bounded observation sweep is different from an idle survey heading.
    scan_while_stationary: bool = False
    allow_local_food: bool = True
    # Retired from foraging/breeding; may still scout unvisited terrain.
    retired: bool = False


@dataclass(frozen=True)
class ReproductionHint:
    energy_threshold: float
    allowed: bool = True
    minimum_energy_reserve: float = 0.
    # An explicit colony-renewal decision may spend an aging parent's reserve
    # to keep a younger generation alive. Ordinary breeding keeps its floor.
    preserve_lineage: bool = False


@dataclass(frozen=True)
class RememberedNeighbor:
    agent_id: int
    angle: float
    distance: float
    weight: float


@dataclass(frozen=True)
class PolicyInputs:
    agent_id: int
    stats: dict
    fruits: tuple[RelativeTarget, ...]
    predator: RelativeTarget | None
    # Filled by the policy's memory layer, in the current facing frame.
    remembered_escape_direction: float | None = None
    escape_seconds_remaining: float = 0.0
    section_hint: SectionHint | None = None
    exploration_hint: ExplorationHint | None = None
    obstacle_edges: tuple = ()
    reproduction_hint: ReproductionHint | None = None
    # One entry per recently seen agent, with distance/freshness weighting.
    neighbors: tuple[RememberedNeighbor, ...] = ()
    observations_fresh: bool = True
    population_phase: bool = False
    harvest_hint: HarvestHint | None = None


def _target(observation: dict) -> RelativeTarget:
    distance = float(observation["distance"])
    angle = float(observation["angle"])
    ripeness = observation.get("ripeness")
    return RelativeTarget(
        vector=(distance * math.cos(angle), distance * math.sin(angle)),
        distance=distance,
        angle=angle,
        ripeness=None if ripeness is None else float(ripeness),
    )


def observed_edges(observations):
    """Read edge endpoints in current facing coordinates, ignoring malformed data."""
    edges = set()
    for observation in observations:
        if observation.get("type") != "Edge":
            continue
        try:
            coords = observation["coords"]
            if len(coords) != 2 or any(len(point) != 2 for point in coords):
                continue
            edge = tuple(tuple(float(value) for value in point) for point in coords)
            if all(math.isfinite(value) for point in edge for value in point):
                # Many vision rays hit the same complete segment. Rechecking
                # it in every steering candidate adds work, not information.
                edges.add(edge)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return tuple(sorted(edges))


def prepare_inputs(
    agent_state: dict, nearest_fruits: int, predator_danger_radius: float
) -> PolicyInputs:
    """Select the n nearest observed fruits and the nearest nearby predator.

    This function prepares current observations; the policy adds escape memory
    separately. Missing ripeness stays None rather than being invented from
    simulator internals. All own stats are retained, including unused traits.
    """
    fruits = []
    predators = []
    for observation in agent_state["observations"]:
        if observation["type"] == "Fruit":
            fruits.append(_target(observation))
        elif (
            observation["type"] == "Predator"
            and observation["distance"] <= predator_danger_radius
        ):
            predators.append(_target(observation))

    # Observations originate from sets; angle makes equal-distance ties stable.
    fruits.sort(key=lambda target: (target.distance, target.angle))
    predator = min(
        predators, key=lambda target: (target.distance, target.angle), default=None
    )
    return PolicyInputs(
        agent_id=agent_state["agent_id"],
        stats={
            key: value for key, value in agent_state.items()
            if key not in ("agent_id", "observations")
        },
        fruits=tuple(fruits[:nearest_fruits]),
        predator=predator,
        obstacle_edges=observed_edges(agent_state["observations"]),
    )
