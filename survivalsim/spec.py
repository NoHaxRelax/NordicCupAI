"""What we know, and what we are guessing.

The observation schema is known exactly: it came off the wire, so the shape of
`agent_status`, the entity dicts and every field name are fixed and must not be
randomized. Building against a fixed interface is the whole point, because it is
the part that will still be correct on competition day.

The dynamics are entirely unknown. Nothing in the payload says how fast energy
drains, what a tree is worth, how a predator hunts, or what `score` rewards. So
every one of those is sampled per episode from a deliberately wide range. A
policy that survives the whole family cannot have overfitted to a guessed
constant, which is the only useful thing to train right now.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

# Fixed by the captured payload. These are interface, not guesses.
ENTITY_TYPES = ("tree", "predator", "edge")
GAME_STATUS = ("running", "game_over")

# Observed in the capture, used to centre the randomisation ranges so the real
# setting is near the middle of what we train on rather than at an edge.
SEEN = {
    "max_energy": 500.0,
    "energy": 85.0,
    "speed": 12.5,
    "sprint_speed": 13.5,
    "hearing_radius": 10.0,
    "vision_angle": 1.57,
    "vision_range": 50.0,
    "biome": "forest",
    "age": 5.2,
    "score": 123.4,
    "world_extent": 100.0,  # from the `edge` coords [[50,50],[100,100]]
}

BIOMES = ("forest", "plains", "desert", "swamp", "tundra")


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Log-uniform, for quantities whose plausible range spans orders of magnitude."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


@dataclass
class WorldSpec:
    """One sampled member of the environment family."""

    # --- world
    world_size: float = 100.0
    n_agents: int = 1
    dt: float = 0.1
    max_age: float = 300.0

    # --- agent body (reported in the payload, so plausibly upgradeable in-game)
    max_energy: float = 500.0
    start_energy_frac: float = 0.3
    speed: float = 12.5
    sprint_mult: float = 1.08
    turn_rate: float = 2.5          # rad/s
    vision_angle: float = 1.57      # half-angle of the cone
    vision_range: float = 50.0
    hearing_radius: float = 10.0

    # --- energy economy
    drain_idle: float = 1.0         # per second
    drain_move: float = 2.0
    drain_sprint_mult: float = 4.0

    # --- trees
    n_trees: int = 30
    tree_energy: float = 60.0
    tree_respawn: float = 20.0
    tree_reach: float = 3.0

    # --- predators
    n_predators: int = 3
    predator_speed_mult: float = 0.9
    predator_aggro: float = 25.0
    predator_kill_range: float = 2.5
    predator_wander: float = 0.6
    predator_fov: float = 1.2       # predators only chase what they can see

    # --- what `score` actually rewards (unknown, so weighted and randomised)
    w_age: float = 1.0
    w_energy_gathered: float = 1.0
    w_trees_eaten: float = 0.0
    death_penalty: float = 10.0

    biome: str = "forest"

    # --- structural switches the capture cannot rule out
    predators_hunt: bool = True
    trees_respawn: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def sample_spec(rng: np.random.Generator, *, hardness: float = 1.0) -> WorldSpec:
    """Draw one environment from the family.

    `hardness` in [0, 1] scales the adversarial dimensions only (predator count,
    aggression, drain), so a curriculum can move through the family without also
    changing its geometry.
    """
    h = float(np.clip(hardness, 0.0, 1.0))

    world = _u(rng, 60.0, 220.0)
    speed = _u(rng, 6.0, 22.0)

    return WorldSpec(
        world_size=world,
        n_agents=int(rng.integers(1, 5)),
        dt=0.1,
        max_age=_u(rng, 120.0, 600.0),

        max_energy=_logu(rng, 150.0, 1200.0),
        start_energy_frac=_u(rng, 0.12, 0.6),
        speed=speed,
        # The capture showed sprint only 8% above walk, which is oddly small, so
        # the range spans "barely worth it" to "real escape tool".
        sprint_mult=_u(rng, 1.05, 1.9),
        turn_rate=_u(rng, 1.2, 5.0),
        vision_angle=_u(rng, 0.6, 2.2),
        vision_range=_u(rng, 25.0, 90.0),
        hearing_radius=_u(rng, 4.0, 25.0),

        drain_idle=_logu(rng, 0.2, 4.0),
        drain_move=_logu(rng, 0.5, 9.0),
        drain_sprint_mult=_u(rng, 1.5, 7.0),

        n_trees=int(rng.integers(8, 70)),
        tree_energy=_logu(rng, 15.0, 220.0),
        tree_respawn=_logu(rng, 4.0, 90.0),
        tree_reach=_u(rng, 1.5, 6.0),

        n_predators=int(rng.integers(0, 2 + int(8 * h))),
        predator_speed_mult=_u(rng, 0.55, 0.85 + 0.5 * h),
        predator_aggro=_u(rng, 10.0, 25.0 + 45.0 * h),
        predator_kill_range=_u(rng, 1.5, 5.0),
        predator_wander=_u(rng, 0.2, 1.4),
        predator_fov=_u(rng, 0.7, np.pi),

        w_age=_u(rng, 0.0, 2.0),
        w_energy_gathered=_u(rng, 0.0, 2.0),
        w_trees_eaten=_u(rng, 0.0, 2.0),
        death_penalty=_logu(rng, 1.0, 60.0),

        biome=str(rng.choice(BIOMES)),
        predators_hunt=bool(rng.random() < 0.85),
        trees_respawn=bool(rng.random() < 0.8),
    )


def captured_spec() -> WorldSpec:
    """The single point in the family most consistent with the captured payload.

    Useful as an eval env: it is the best guess at the real thing, and a policy
    trained on the family should already be decent here without ever seeing it.
    """
    return WorldSpec(
        world_size=SEEN["world_extent"],
        n_agents=1,
        max_energy=SEEN["max_energy"],
        start_energy_frac=SEEN["energy"] / SEEN["max_energy"],
        speed=SEEN["speed"],
        sprint_mult=SEEN["sprint_speed"] / SEEN["speed"],
        vision_angle=SEEN["vision_angle"],
        vision_range=SEEN["vision_range"],
        hearing_radius=SEEN["hearing_radius"],
        biome=SEEN["biome"],
    )
