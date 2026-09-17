"""World views. Everything above this layer reads only these structures.

The oracle fills them from the true engine state; the estimator fills them from
observations. Fields the estimator cannot know are ``None`` (predator energy,
resting flag, fruit age) and the trap logic must not depend on them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Rect, wrap

AGENT_RADIUS = 5.0
PREDATOR_RADIUS = 10.0
KILL_DISTANCE = AGENT_RADIUS + PREDATOR_RADIUS      # strict: distance < 15 kills
PRED_HEARING = 60.0
PRED_VISION = 250.0
PRED_CONE = math.pi / 3
PRED_WALK = 11.0
PRED_SPRINT = 15.0
PRED_MAX_ENERGY = 200.0
PRED_CHARGE_RANGE = PRED_HEARING * 1.5               # 90: inside, it charges regardless of facing
MOVE_PENALTY = dict(forest=1.0, grassland=1.0, swamp=0.5, desert=0.8, river=0.3)


@dataclass
class AgentView:
    id: int
    x: float
    y: float
    heading: float
    energy: float
    max_energy: float
    age: float
    speed: float
    sprint_speed: float
    hearing: float
    vision_range: float
    vision_angle: float
    biome: str
    state: dict = field(default_factory=dict, repr=False)   # the raw observation DTO

    @property
    def p(self):
        return (self.x, self.y)

    @property
    def walk(self):
        return min(self.speed, self.sprint_speed)

    @property
    def can_sprint(self):
        return self.energy >= self.max_energy / 5 and self.sprint_speed > self.speed

    @property
    def move_modifier(self):
        return MOVE_PENALTY.get(self.biome, 1.0)


@dataclass
class PredatorView:
    pid: int
    x: float
    y: float
    heading: float
    vx: float = 0.0            # units per tick, from successive positions
    vy: float = 0.0
    last_seen: float = 0.0     # simulation time of the last observation
    still_ticks: int = 0       # consecutive ticks without movement
    resting: bool | None = None
    energy: float | None = None
    fresh: bool = True         # position is current (oracle) or predicted to current

    @property
    def p(self):
        return (self.x, self.y)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


@dataclass
class FoodView:
    x: float
    y: float
    age: float | None = None   # fruit internal age (2 x seconds) when known
    energy: float | None = None

    @property
    def p(self):
        return (self.x, self.y)


@dataclass
class WorldState:
    time: float
    width: float
    height: float
    agents: dict[int, AgentView]
    predators: list[PredatorView]
    rects: list[Rect]                       # every known obstacle incl. arena boundaries
    fruits: list[FoodView] = field(default_factory=list)
    trees: list[FoodView] = field(default_factory=list)
    complete_map: bool = False              # True when rects are the complete obstacle set

    def predator(self, pid):
        for p in self.predators:
            if p.pid == pid:
                return p
        return None

    def biome_at(self, p):
        """Movement modifier lookup; oracle overrides with the true biome map."""
        return 1.0


def rel_dir_from(observer_p, target_p, target_heading):
    """Engine ``rel_dir``: bearing of the observer in the target's body frame."""
    return wrap(math.atan2(observer_p[1] - target_p[1], observer_p[0] - target_p[0]) - target_heading)
