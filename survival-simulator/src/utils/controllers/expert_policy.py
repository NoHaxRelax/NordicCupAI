"""A configurable, deterministic expert shared by every agent."""

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.utils.DTOs import ActionRequest
from src.utils.controllers.crowd_memory import CrowdConfig, CrowdTracker, crowd_direction
from src.utils.controllers.exploration import avoid_edges, _path_clear
from src.utils.controllers.global_planner import GlobalPlanner, PlannerConfig
from src.utils.controllers.harvest import HarvestConfig, HarvestCoordinator
from src.utils.controllers.policy_inputs import ExplorationHint, PolicyInputs, ReproductionHint, SectionHint, prepare_inputs
from src.utils.controllers.population import PopulationConfig, PopulationTracker


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "expert_policy.json"


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PerceptionConfig(ConfigSection):
    nearest_fruits: int = Field(ge=1, strict=True)
    predator_danger_radius: float = Field(ge=0)


class MovementConfig(ConfigSection):
    walk_speed_fraction: float = Field(ge=0, le=1)
    flee_speed_fraction: float = Field(ge=0, le=1)
    max_turn_angle: float = Field(ge=0, le=math.pi)
    turn_deadzone: float = Field(ge=0, le=math.pi)
    food_sprint_enabled: bool = Field(default=False, strict=True)
    food_sprint_max_distance: float = Field(default=70.0, gt=0)
    food_sprint_min_energy: float = Field(default=170.0, ge=0)
    food_sprint_energy_reserve: float = Field(default=110.0, ge=0)
    sprint_clearance_margin: float = Field(default=5.0, ge=0)
    scan_nearby_food: bool = Field(default=False, strict=True)


class MemoryConfig(ConfigSection):
    predator_escape_seconds: float = Field(ge=0)


class ExplorationConfig(ConfigSection):
    speed_fraction: float = Field(ge=0, le=1)
    turn_amplitude: float = Field(ge=0, le=math.pi)
    turn_period_seconds: float = Field(gt=0)


class ReproductionConfig(ConfigSection):
    enabled: bool = Field(strict=True)
    energy_threshold: float = Field(ge=0)
    minimum_age: float = Field(ge=0)
    minimum_energy_reserve: float = Field(ge=0)
    selection: PopulationConfig = Field(default_factory=PopulationConfig)


class MechanicsConfig(ConfigSection):
    walking_energy_per_unit: float = Field(ge=0)
    sprinting_energy_per_unit: float = Field(ge=0)
    spawn_energy_cost: float = Field(gt=0)
    sprint_min_energy_fraction: float = Field(ge=0, le=1)


class ExpertConfig(ConfigSection):
    perception: PerceptionConfig
    memory: MemoryConfig
    crowd: CrowdConfig
    movement: MovementConfig
    exploration: ExplorationConfig
    reproduction: ReproductionConfig
    mechanics: MechanicsConfig
    harvest: HarvestConfig = Field(default_factory=HarvestConfig)


def load_config(path: str | Path | None = None) -> ExpertConfig:
    """Load once per controller; all parameter defaults live in the JSON file."""
    config_path = Path(path or os.environ.get("EXPERT_POLICY_CONFIG") or DEFAULT_CONFIG_PATH)
    return ExpertConfig.model_validate_json(config_path.read_text(encoding="utf-8"))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


@dataclass
class EscapeMemory:
    direction: float
    last_time: float
    last_age: float
    last_turn: float = 0.0
    unseen_since: float | None = None


class ExpertPolicy:
    def __init__(self, config: ExpertConfig | None = None, planner_config: PlannerConfig | None = None):
        self.config = config if config is not None else load_config()
        self.planner = GlobalPlanner(planner_config)
        self.crowd_tracker = CrowdTracker(self.config.crowd)
        self.population = PopulationTracker(self.config.reproduction.selection)
        self.harvest = HarvestCoordinator(self.config.harvest)
        self._escape_memories: dict[int, EscapeMemory] = {}
        self._observation_frames: dict[int, tuple[float, float, bool]] = {}
        self._last_step_time: float | None = None

    def reset(self):
        """Forget all agents at an episode boundary."""
        self._escape_memories.clear()
        self._observation_frames.clear()
        self.planner.reset()
        self.crowd_tracker.reset()
        self.population.reset()
        self.harvest.reset()
        self._last_step_time = None

    def actions_for_step(self, agent_states: list[dict], sim_time: float,
                         game_status: str = "ok") -> list[ActionRequest]:
        """Process a complete population; prune dead agents and detect restarts.

        Use one policy instance per active simulation. Repeated requests at the
        same timestamp do not advance timers or apply the same turn twice.
        """
        if game_status == "game_over" or not agent_states:
            self.reset()
            return []
        if self._last_step_time is not None and sim_time < self._last_step_time:
            self.reset()
        living_ids = {state["agent_id"] for state in agent_states}
        self.crowd_tracker.prune(living_ids)
        self._escape_memories = {
            agent_id: memory for agent_id, memory in self._escape_memories.items()
            if agent_id in living_ids
        }
        self._observation_frames = {key: frame for key, frame in self._observation_frames.items()
                                    if key in living_ids}
        self.population.update(agent_states, sim_time)
        hints = self.planner.instructions(agent_states, sim_time, trait_ratings=self.population.ratings,
            territory_policy=self.config.harvest.enabled and self.config.harvest.coverage.enabled)
        self.population.population_phase = self.planner.population_phase
        self.planner.population_snapshot = self.population.snapshot()
        harvest_hints, breeding_hints = self.harvest.update(
            agent_states, sim_time, self.planner, self.population, self.config.mechanics,
            self.config.reproduction.energy_threshold)
        actions = [self.action_decision(state, sim_time=sim_time,
                                       section_hint=hints.get(state["agent_id"]),
                                       exploration_hint=self.planner.exploration_hints.get(state["agent_id"]),
                                       reproduction_hint=breeding_hints.get(state["agent_id"], self.population.reproduction_hint(
                                           state["agent_id"], sim_time, self.config.reproduction.energy_threshold)),
                                       harvest_hint=harvest_hints.get(state["agent_id"]))
                   for state in agent_states]
        self.planner.remember_actions(actions)
        self.population.remember_actions(actions, sim_time)
        self.harvest.remember_actions(actions, sim_time)
        self.planner.harvest_snapshot = self.harvest.snapshot()
        self._last_step_time = sim_time
        return actions

    def action_decision(self, agent_state: dict, sim_time: float | None = None,
                        section_hint: SectionHint | None = None,
                        exploration_hint: ExplorationHint | None = None,
                        reproduction_hint: ReproductionHint | None = None,
                        harvest_hint=None) -> ActionRequest:
        """Add per-agent memory, then apply the rules to the prepared inputs.

        Single-agent callers may omit sim_time to use agent age as their clock.
        Production runners use actions_for_step with the actual simulation time.
        """
        inputs = prepare_inputs(
            agent_state,
            self.config.perception.nearest_fruits,
            self.config.perception.predator_danger_radius,
        )
        inputs = replace(inputs, section_hint=section_hint, exploration_hint=exploration_hint,
                         reproduction_hint=reproduction_hint, harvest_hint=harvest_hint)
        now = float(agent_state["age"] if sim_time is None else sim_time)
        frame = self._observation_frames.get(inputs.agent_id)
        age = agent_state["age"]
        # The simulator can skip a survivor's observation refresh after a
        # death. A cached cone/edge position cannot authorize a fresh sprint.
        fresh = (frame[2] if frame is not None and now == frame[0] and age == frame[1]
                 else not (frame is not None and now > frame[0] and age == frame[1]))
        self._observation_frames[inputs.agent_id] = (now, age, fresh)
        inputs = replace(inputs, observations_fresh=fresh, population_phase=self.planner.population_phase)
        inputs = replace(inputs, neighbors=self.crowd_tracker.observe(agent_state, now))
        memory = self._escape_memories.get(inputs.agent_id)
        if memory and (now < memory.last_time or age < memory.last_age):
            # Protect direct callers and reused IDs after a restart.
            self._escape_memories.pop(inputs.agent_id)
            memory = None
        if memory and now > memory.last_time:
            # The last action has now been applied. Keep the escape heading
            # fixed in the world while expressing it relative to current facing.
            memory.direction = _wrap_angle(memory.direction - memory.last_turn)

        duration = self.config.memory.predator_escape_seconds
        if inputs.predator is not None and duration > 0:
            memory = EscapeMemory(
                direction=_wrap_angle(inputs.predator.angle + math.pi),
                last_time=now, last_age=age,
            )
            self._escape_memories[inputs.agent_id] = memory
        elif memory:
            if memory.unseen_since is None:
                memory.unseen_since = now
            deadline = memory.unseen_since + duration
            if now >= deadline or math.isclose(now, deadline, rel_tol=0, abs_tol=1e-9):
                self._escape_memories.pop(inputs.agent_id)
                memory = None
            else:
                inputs = replace(
                    inputs, remembered_escape_direction=memory.direction,
                    escape_seconds_remaining=deadline - now,
                )

        action = self.decide(inputs)
        self.crowd_tracker.remember_turn(inputs.agent_id, action.turn_angle)
        if memory:
            memory.last_time = now
            memory.last_age = age
            memory.last_turn = action.turn_angle
        return action

    def decide(self, inputs: PolicyInputs) -> ActionRequest:
        """Flee, forage, or explore; independently reproduce when safe."""
        cfg = self.config
        stats = inputs.stats
        walk_speed = min(stats["speed"], stats["sprint_speed"])
        in_danger = inputs.predator is not None or inputs.remembered_escape_direction is not None
        mapping_sprint_reserve = None
        explore_look = (None if inputs.exploration_hint is None else inputs.exploration_hint.look_direction)
        scan_during_food = False
        fruits = inputs.fruits
        harvest = inputs.harvest_hint
        assigned_harvest = harvest is not None and harvest.vector is not None
        if harvest is not None and (assigned_harvest or stats["energy"] >= cfg.harvest.emergency_energy):
            fruits = ()
        if inputs.exploration_hint is not None and inputs.exploration_hint.food_distance_limit is not None:
            fruits = tuple(fruit for fruit in fruits
                           if fruit.distance <= inputs.exploration_hint.food_distance_limit)

        if in_danger:
            # Movement occurs BEFORE turning: both angles use current facing.
            direction = (
                _wrap_angle(inputs.predator.angle + math.pi) if inputs.predator is not None
                else inputs.remembered_escape_direction
            )
            distance = stats["sprint_speed"] * cfg.movement.flee_speed_fraction
            desired_turn = direction
        elif assigned_harvest:
            x, y = harvest.vector
            direction = math.atan2(y, x)
            distance = min(math.hypot(x, y), walk_speed * cfg.movement.walk_speed_fraction)
            desired_turn = direction
        elif fruits:
            # Compare ripeness only if every candidate supplies it. If any is
            # unknown, choose the nearest rather than treating unknown as unripe.
            if all(fruit.ripeness is not None for fruit in fruits):
                target = min(
                    fruits,
                    key=lambda fruit: (-fruit.ripeness, fruit.distance, fruit.angle),
                )
            else:
                target = min(fruits, key=lambda fruit: (fruit.distance, fruit.angle))
            direction = _wrap_angle(target.angle)
            distance = min(target.distance, walk_speed * cfg.movement.walk_speed_fraction)
            if (cfg.movement.food_sprint_enabled and not inputs.population_phase
                    and stats["energy"] >= cfg.movement.food_sprint_min_energy
                    and walk_speed < target.distance <= cfg.movement.food_sprint_max_distance
                    and abs(direction) <= stats["vision_angle"] / 2
                    and target.distance <= stats["vision_range"]
                    and _path_clear(direction, target.distance, inputs.obstacle_edges,
                                    cfg.movement.sprint_clearance_margin)):
                distance = min(target.distance, stats["sprint_speed"])
                mapping_sprint_reserve = cfg.movement.food_sprint_energy_reserve
            # Nearby fruit remains audible while the scout surveys sideways.
            scan_during_food = (cfg.movement.scan_nearby_food and inputs.observations_fresh
                                and target.distance <= stats["hearing_radius"])
            desired_turn = direction
        elif inputs.exploration_hint is not None:
            hint = inputs.exploration_hint
            x, y = hint.vector
            direction = math.atan2(y, x)
            limit = walk_speed if hint.max_speed is None else min(hint.max_speed, stats["sprint_speed"])
            distance = min(math.hypot(x, y), max(0.0, limit))
            if distance > walk_speed:
                mapping_sprint_reserve = hint.minimum_energy_reserve
            explore_look = hint.look_direction
            desired_turn = direction
        else:
            direction = 0.0
            distance = walk_speed * cfg.exploration.speed_fraction
            # Sweep gently left and right when there is no remembered danger.
            desired_turn = cfg.exploration.turn_amplitude * math.sin(
                math.tau * stats["age"] / cfg.exploration.turn_period_seconds
            )

        if not in_danger and not assigned_harvest and inputs.section_hint is not None:
            hint = inputs.section_hint
            center_distance = math.hypot(*hint.vector)
            if center_distance > 0 and hint.strength > 0:
                # Blend velocities, not angles (which wrap at +/-pi). The push
                # stays gentle, including when it opposes the forage direction.
                push_distance = min(center_distance, walk_speed * cfg.movement.walk_speed_fraction)
                x = (1 - hint.strength) * distance * math.cos(direction)
                y = (1 - hint.strength) * distance * math.sin(direction)
                x += hint.strength * push_distance * hint.vector[0] / center_distance
                y += hint.strength * push_distance * hint.vector[1] / center_distance
                distance = math.hypot(x, y)
                if distance > 0:
                    direction = math.atan2(y, x)
                    desired_turn = direction

        if not in_danger and not assigned_harvest and distance > 0:
            crowd = crowd_direction(inputs.neighbors, direction, cfg.crowd, inputs.agent_id)
            if crowd is not None:
                open_direction, crowd_strength = crowd
                strength = (cfg.crowd.foraging_strength if fruits
                            else cfg.crowd.exploration_strength) * crowd_strength
                if strength > 0:
                    # Angular steering also works when the open direction is
                    # directly behind us; vector blending would only slow down.
                    direction = _wrap_angle(direction + strength * _wrap_angle(open_direction - direction))
                    desired_turn = direction

        if not in_danger and (assigned_harvest or (not fruits and inputs.exploration_hint is not None)):
            direction, distance = avoid_edges(
                direction, distance, inputs.obstacle_edges,
                4.0 if assigned_harvest else inputs.exploration_hint.obstacle_margin, inputs.agent_id,
            )
            desired_turn = direction

        distance = max(0.0, min(distance, stats["sprint_speed"]))
        if stats["energy"] < stats["max_energy"] * cfg.mechanics.sprint_min_energy_fraction:
            distance = min(distance, walk_speed)

        if not in_danger and distance > walk_speed:
            # Check the final heading AFTER crowd/section steering. A map goal
            # is not evidence that an unobserved fast step is clear.
            if (mapping_sprint_reserve is None
                    or not inputs.observations_fresh
                    or abs(_wrap_angle(direction)) > stats["vision_angle"] / 2
                    or distance + cfg.movement.sprint_clearance_margin > stats["vision_range"]
                    or not _path_clear(direction, distance + cfg.movement.sprint_clearance_margin,
                                       inputs.obstacle_edges, cfg.movement.sprint_clearance_margin)):
                distance = min(distance, walk_speed)

        if not in_danger and not assigned_harvest and (not fruits or scan_during_food) and explore_look is not None:
            # Facing is independent of movement. Survey while travelling;
            # food and emergency escape still look where they are going.
            desired_turn = _wrap_angle(explore_look)

        if not in_danger and assigned_harvest and harvest.survey and harvest.look_direction is not None:
            desired_turn = _wrap_angle(harvest.look_direction)

        turn = max(
            -cfg.movement.max_turn_angle,
            min(desired_turn, cfg.movement.max_turn_angle),
        )
        if abs(turn) < cfg.movement.turn_deadzone:
            turn = 0.0

        if not in_danger and distance > walk_speed:
            reserve = max(mapping_sprint_reserve,
                          stats["max_energy"] * cfg.mechanics.sprint_min_energy_fraction)
            budget = (stats["energy"] - reserve - min(math.pi, abs(turn)) / math.tau
                      - walk_speed * cfg.mechanics.walking_energy_per_unit)
            extra = (max(0.0, budget / cfg.mechanics.sprinting_energy_per_unit)
                     if cfg.mechanics.sprinting_energy_per_unit > 0
                     else (stats["sprint_speed"] - walk_speed if budget >= 0 else 0.0))
            distance = min(distance, walk_speed + extra)

        # The engine charges movement and turning before checking reproduction.
        action_cost = (
            min(distance, stats["speed"]) * cfg.mechanics.walking_energy_per_unit
            + max(0.0, distance - stats["speed"]) * cfg.mechanics.sprinting_energy_per_unit
            + min(math.pi, abs(turn)) / math.tau
        )
        energy_after_action = stats["energy"] - action_cost
        breeding = inputs.reproduction_hint
        spawn = (
            cfg.reproduction.enabled
            and (breeding is None or breeding.allowed)
            and not in_danger
            and stats["age"] >= cfg.reproduction.minimum_age
            and stats["energy"] > (cfg.reproduction.energy_threshold if breeding is None else breeding.energy_threshold)
            and energy_after_action > cfg.mechanics.spawn_energy_cost
            and (
                energy_after_action - cfg.mechanics.spawn_energy_cost
                >= max(cfg.reproduction.minimum_energy_reserve,
                       0. if breeding is None else breeding.minimum_energy_reserve)
            )
        )
        return ActionRequest(
            agent_id=inputs.agent_id,
            move_distance=distance,
            move_direction=direction,
            turn_angle=turn,
            spawn_agent=spawn,
        )
