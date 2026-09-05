"""A mock survival sim whose rules are randomised on every episode.

READ THIS BEFORE USING IT AS A TRAINING TARGET -- it is not one.

The purpose of this module is NOT to be right about the game. It cannot be:
every rule in it is a guess, and the real game will differ. Its purpose is to
be a harness that (a) exercises the full loop under load, and (b) makes
overfitting to our guesses *detectable*.

It does that by drawing the unknown rules fresh per episode (`RuleDraw`),
including the ones that are ambiguous in the observed schema: whether angles
are counter-clockwise-positive, whether `vision_angle` is a half-cone or a
full field of view, whether `edge.coords` are world or agent-relative,
whether sprinting drains energy, whether predators pursue at all.

A policy that scores well only under one draw is fitted to a guess. A policy
that scores well across draws is fitted to the *structure* -- and structure is
the part we have real evidence for.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from schema import GameState


@dataclass(frozen=True)
class RuleDraw:
    """One sampled interpretation of the ambiguous rules."""

    ccw_positive: bool = True          # sign convention of `angle`
    vision_is_half_angle: bool = True  # is vision_angle a half-cone?
    edges_are_world: bool = True       # are edge coords absolute?
    sprint_drains: bool = True
    sprint_cost: float = 3.0           # energy per second while sprinting
    move_cost: float = 1.0
    idle_cost: float = 0.2
    predators_pursue: bool = True
    predator_speed: float = 11.0
    predator_aggro_range: float = 45.0
    kill_range: float = 2.0
    score_per_second: float = 1.0
    n_predators: int = 3
    n_trees: int = 25
    world_size: float = 200.0

    @classmethod
    def sample(cls, rng: random.Random) -> "RuleDraw":
        return cls(
            ccw_positive=rng.random() < 0.5,
            vision_is_half_angle=rng.random() < 0.5,
            edges_are_world=rng.random() < 0.5,
            sprint_drains=rng.random() < 0.8,
            sprint_cost=rng.uniform(1.5, 8.0),
            move_cost=rng.uniform(0.3, 2.5),
            idle_cost=rng.uniform(0.0, 0.6),
            predators_pursue=rng.random() < 0.8,
            predator_speed=rng.uniform(8.0, 16.0),
            predator_aggro_range=rng.uniform(20.0, 70.0),
            kill_range=rng.uniform(1.0, 4.0),
            n_predators=rng.randint(1, 6),
            n_trees=rng.randint(5, 40),
            world_size=rng.uniform(120.0, 400.0),
        )


@dataclass
class _Entity:
    x: float
    y: float
    heading: float = 0.0


@dataclass
class MockSim:
    """A minimal 2D survival world emitting the observed payload shape."""

    rules: RuleDraw = field(default_factory=RuleDraw)
    seed: int = 0
    dt: float = 0.2
    n_agents: int = 1
    max_energy: float = 500.0
    speed: float = 12.5
    sprint_speed: float = 18.0
    vision_range: float = 50.0
    vision_angle: float = math.pi / 2
    hearing_radius: float = 10.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> GameState:
        r, w = self._rng, self.rules.world_size
        self.t = 0.0
        self.score = 0.0
        self.alive = {i + 1: True for i in range(self.n_agents)}
        self.energy = {i + 1: self.max_energy * 0.6 for i in range(self.n_agents)}
        self.agents = {
            i + 1: _Entity(r.uniform(0, w), r.uniform(0, w), r.uniform(-math.pi, math.pi))
            for i in range(self.n_agents)
        }
        self.trees = [_Entity(r.uniform(0, w), r.uniform(0, w)) for _ in range(self.rules.n_trees)]
        self.predators = [
            _Entity(r.uniform(0, w), r.uniform(0, w), r.uniform(-math.pi, math.pi))
            for _ in range(self.rules.n_predators)
        ]
        return self.state()

    # -- dynamics -------------------------------------------------------

    def step(self, actions: dict) -> GameState:
        rules, dt = self.rules, self.dt
        sign = 1.0 if rules.ccw_positive else -1.0

        for agent_id, agent in self.agents.items():
            if not self.alive[agent_id]:
                continue
            action = actions.get(agent_id)
            if action is None:
                continue
            agent.heading += sign * action.turn
            sprinting = action.sprint and rules.sprint_drains
            speed = self.sprint_speed if action.sprint else self.speed
            dist = action.throttle * speed * dt
            agent.x = min(max(agent.x + dist * math.cos(agent.heading), 0.0), rules.world_size)
            agent.y = min(max(agent.y + dist * math.sin(agent.heading), 0.0), rules.world_size)

            cost = rules.idle_cost
            if action.throttle > 0:
                cost = rules.sprint_cost if sprinting else rules.move_cost
            self.energy[agent_id] = max(0.0, self.energy[agent_id] - cost * dt)
            if self.energy[agent_id] <= 0.0:
                self.alive[agent_id] = False

        for predator in self.predators:
            target = self._nearest_live_agent(predator)
            if rules.predators_pursue and target is not None:
                d = math.hypot(target.x - predator.x, target.y - predator.y)
                if d <= rules.predator_aggro_range:
                    predator.heading = math.atan2(target.y - predator.y, target.x - predator.x)
                else:
                    predator.heading += self._rng.uniform(-0.3, 0.3)
            else:
                predator.heading += self._rng.uniform(-0.3, 0.3)
            step = rules.predator_speed * dt
            predator.x = min(max(predator.x + step * math.cos(predator.heading), 0.0), rules.world_size)
            predator.y = min(max(predator.y + step * math.sin(predator.heading), 0.0), rules.world_size)

        for agent_id, agent in self.agents.items():
            if not self.alive[agent_id]:
                continue
            for predator in self.predators:
                if math.hypot(agent.x - predator.x, agent.y - predator.y) <= rules.kill_range:
                    self.alive[agent_id] = False
                    break

        self.t += dt
        self.score += rules.score_per_second * dt * sum(self.alive.values())
        return self.state()

    def _nearest_live_agent(self, entity: _Entity) -> _Entity | None:
        live = [a for i, a in self.agents.items() if self.alive[i]]
        if not live:
            return None
        return min(live, key=lambda a: math.hypot(a.x - entity.x, a.y - entity.y))

    # -- observation ----------------------------------------------------

    def _visible(self, agent: _Entity, target: _Entity) -> tuple[float, float] | None:
        dx, dy = target.x - agent.x, target.y - agent.y
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - agent.heading
        bearing = -((-bearing + math.pi) % (2 * math.pi) - math.pi)
        if distance <= self.hearing_radius:
            return distance, bearing  # heard, regardless of facing
        half = self.vision_angle if self.rules.vision_is_half_angle else self.vision_angle / 2
        if distance <= self.vision_range and abs(bearing) <= half:
            return distance, bearing
        return None

    def state(self) -> GameState:
        sign = 1.0 if self.rules.ccw_positive else -1.0
        agent_status = []
        for agent_id, agent in self.agents.items():
            observations = []
            if self.alive[agent_id]:
                for tree in self.trees:
                    seen = self._visible(agent, tree)
                    if seen:
                        observations.append(
                            {"type": "tree", "distance": seen[0], "angle": sign * seen[1]}
                        )
                for predator in self.predators:
                    seen = self._visible(agent, predator)
                    if seen:
                        rel = predator.heading - agent.heading
                        rel = -((-rel + math.pi) % (2 * math.pi) - math.pi)
                        observations.append(
                            {
                                "type": "predator",
                                "distance": seen[0],
                                "angle": sign * seen[1],
                                "rel_dir": sign * rel,
                            }
                        )
                observations.append({"type": "edge", "coords": self._edge_coords(agent)})
            agent_status.append(
                {
                    "agent_id": agent_id,
                    "observations": observations,
                    "energy": self.energy[agent_id],
                    "max_energy": self.max_energy,
                    "biome": "forest",
                    "age": self.t,
                    "speed": self.speed,
                    "sprint_speed": self.sprint_speed,
                    "hearing_radius": self.hearing_radius,
                    "vision_angle": self.vision_angle,
                    "vision_range": self.vision_range,
                }
            )
        payload = {
            "game_status": "running" if any(self.alive.values()) else "finished",
            "score": self.score,
            "agent_status": agent_status,
        }
        return GameState.parse(payload)

    def _edge_coords(self, agent: _Entity) -> list[list[float]]:
        w = self.rules.world_size
        if self.rules.edges_are_world:
            return [[0.0, 0.0], [w, w]]
        return [[-agent.x, -agent.y], [w - agent.x, w - agent.y]]


def run_episode(policy, rules: RuleDraw, seed: int = 0, max_steps: int = 600) -> dict:
    """Run one episode and return summary metrics."""
    sim = MockSim(rules=rules, seed=seed)
    policy.reset()
    state = sim.state()
    steps = 0
    for steps in range(1, max_steps + 1):
        actions = policy.act(state, dt=sim.dt)
        state = sim.step(actions)
        if not state.is_running:
            break
    return {
        "score": round(sim.score, 2),
        "survived_s": round(sim.t, 2),
        "steps": steps,
        "died": not state.is_running,
    }


def evaluate(policy_factory, n_draws: int = 20, seed: int = 0, max_steps: int = 600) -> dict:
    """Score a policy across many rule draws.

    The headline number is the *median across draws*, not the mean of one
    draw. A policy is only credible here if it holds up when the rules it was
    written under turn out to be false.
    """
    rng = random.Random(seed)
    results = []
    for i in range(n_draws):
        rules = RuleDraw.sample(rng)
        results.append(run_episode(policy_factory(), rules, seed=rng.randrange(10**6), max_steps=max_steps))
    survived = sorted(r["survived_s"] for r in results)
    mid = len(survived) // 2
    return {
        "draws": n_draws,
        "median_survival_s": survived[mid],
        "worst_survival_s": survived[0],
        "best_survival_s": survived[-1],
        "death_rate": round(sum(r["died"] for r in results) / len(results), 3),
    }
