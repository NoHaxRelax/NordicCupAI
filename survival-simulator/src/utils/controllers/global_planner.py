"""Food-balanced territories built exclusively from the competition observations."""

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from src.utils.controllers.biome_estimator import BiomeEstimator, BiomeInferenceConfig
from src.utils.controllers.exploration import ExplorationConfig, ExplorationCoordinator
from src.utils.controllers.policy_inputs import SectionHint
from src.utils.controllers.world_estimator import EstimatorConfig, WorldEstimator, rotate


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "global_planner.json"


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(strict=True)
    mapping_enabled: bool = Field(default=False, strict=True)
    population_after_alignment: bool = Field(default=False, strict=True)
    alignment_hold_seconds: float = Field(default=1.0, ge=0)
    biome_inference: BiomeInferenceConfig = Field(default_factory=BiomeInferenceConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    replan_interval_seconds: float = Field(gt=0)
    grid_cell_size: float = Field(gt=0)
    max_grid_cells: int = Field(ge=1, strict=True)
    max_sections: int = Field(ge=1, strict=True)
    push_strength: float = Field(ge=0, le=1)
    center_distance_margin: float = Field(ge=0)
    tree_food_weight: float = Field(ge=0)
    biome_prior_weight: float = Field(ge=0)
    draw_overlay: bool = Field(strict=True)
    estimator: EstimatorConfig


def load_planner_config(path=None):
    path = Path(path or os.environ.get("GLOBAL_PLANNER_CONFIG") or DEFAULT_CONFIG_PATH)
    return PlannerConfig.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class FoodGrid:
    origin: np.ndarray
    cell_size: float
    weights: np.ndarray
    known: np.ndarray


@dataclass
class Section:
    section_id: int
    bounds: tuple[float, float, float, float]
    center: np.ndarray
    food_weight: float


@dataclass
class GroupPlan:
    sections: list[Section]
    updated_at: float
    revision: int
    assignments: dict[int, int] = field(default_factory=dict)


def food_grid(group, config):
    """Observed trees dominate; visited biomes provide a weak food prior.

    Unknown cells have zero weight. No map bounds or unseen biome values are
    inferred. Coarsen the grid if necessary to bound memory and partition cost.
    """
    samples = [(tree.position, config.tree_food_weight) for tree in group.trees]
    samples.extend((position, potential * config.biome_prior_weight)
                   for position, potential, _ in group.visited.values())
    if not samples:
        return None
    positions = np.array([position for position, _ in samples])
    cell_size = config.grid_cell_size
    # An origin at the lower observed extent lets even a one-cell budget work.
    origin = positions.min(axis=0)
    while True:
        indices = np.floor((positions - origin) / cell_size).astype(int)
        nx, ny = indices.max(axis=0) + 1
        if int(nx) * int(ny) <= config.max_grid_cells:
            break
        cell_size *= 2
    weights = np.zeros((ny, nx))
    known = np.zeros((ny, nx), dtype=bool)
    for (x, y), (_, weight) in zip(indices, samples):
        weights[y, x] += weight
        known[y, x] = True
    return FoodGrid(origin, cell_size, weights, known)


def partition_food(grid, count):
    """Recursively cut rectangles into approximately equal food shares.

    A cell is indivisible, so a concentrated food source can prevent exact
    balance. Every leaf contains a known cell; with no food, balance known area.
    """
    count = min(count, int(grid.known.sum()))
    sections = []
    if count < 1:
        return sections

    def split(x0, y0, x1, y1, n):
        weights = grid.weights[y0:y1, x0:x1]
        known = grid.known[y0:y1, x0:x1]
        total = float(weights.sum())
        balance = weights if total > 0 else known.astype(float)
        if n == 1:
            yy, xx = np.nonzero(known)
            points = grid.origin + (np.column_stack((xx + x0, yy + y0)) + 0.5) * grid.cell_size
            point_weights = weights[yy, xx]
            center = np.average(points, axis=0, weights=point_weights if total > 0 else None)
            # Keep the center in a discovered cell rather than in an unknown gap.
            center = points[np.argmin(np.linalg.norm(points - center, axis=1))].copy()
            lower = grid.origin + np.array((x0, y0)) * grid.cell_size
            upper = grid.origin + np.array((x1, y1)) * grid.cell_size
            sections.append(Section(len(sections), (*lower, *upper), center, total))
            return
        candidates = []
        # Prefer the longer dimension when both cuts balance equally well.
        axes = (0, 1) if x1 - x0 >= y1 - y0 else (1, 0)
        for preference, axis in enumerate(axes):
            mass = balance.sum(axis=axis).cumsum()
            capacity = known.sum(axis=axis).cumsum()
            for cut in range(1, len(mass)):
                minimum = max(1, n - int(capacity[-1] - capacity[cut - 1]))
                maximum = min(n - 1, int(capacity[cut - 1]))
                if minimum > maximum:
                    continue
                # Sparse grids may not admit exactly n//2 leaves on either
                # side. Choose the feasible allocation closest to its food share.
                fraction = float(mass[cut - 1] / mass[-1])
                left_count = max(minimum, min(maximum, round(fraction * n)))
                error = abs(fraction - left_count / n)
                candidates.append((error, abs(n - 2 * left_count), preference, cut, axis, left_count))
        _, _, _, cut, axis, left_count = min(candidates)
        right_count = n - left_count
        if axis == 0:
            split(x0, y0, x0 + cut, y1, left_count)
            split(x0 + cut, y0, x1, y1, right_count)
        else:
            split(x0, y0, x1, y0 + cut, left_count)
            split(x0, y0 + cut, x1, y1, right_count)

    ny, nx = grid.weights.shape
    split(0, 0, nx, ny, count)
    return sections


class GlobalPlanner:
    def __init__(self, config=None):
        self.config = config if config is not None else load_planner_config()
        self.estimator = WorldEstimator(self.config.estimator)
        self.exploration = ExplorationCoordinator(self.config.exploration,
                                                 self.config.estimator.max_position_uncertainty)
        self.biome_estimator = BiomeEstimator(self.config.biome_inference)
        self.reset()

    def reset(self):
        self.estimator.reset()
        self.exploration.reset()
        self.biome_estimator.reset()
        self.plans: dict[int, GroupPlan] = {}
        self.hints: dict[int, SectionHint] = {}
        self.exploration_hints = {}
        self.trait_ratings = {}
        self.population_snapshot = {}
        self.harvest_snapshot = {}
        self.population_phase = False
        self.shared_frame_since = None
        self.shared_frame_established_at = None
        self.last_time = None

    def _update_phase(self, now):
        if not self.config.population_after_alignment:
            return
        groups = {pose.group_id for pose in self.estimator.poses.values()}
        anchored = {key for key in groups if self.estimator.groups[key].anchored}
        if not anchored:
            self.population_phase = False
            self.shared_frame_established_at = None
        if self.population_phase:
            # Newborns can briefly have their own local frame. Joining them
            # should not restart the whole colony's initial scouting phase.
            return
        if len(groups) != 1 or not anchored:
            self.shared_frame_since = None
            return
        if self.shared_frame_since is None:
            self.shared_frame_since = now
        if now - self.shared_frame_since >= self.config.alignment_hold_seconds - 1e-9:
            self.population_phase = True
            self.shared_frame_established_at = now

    def remember_actions(self, actions):
        if self.config.enabled or self.config.mapping_enabled or self.config.exploration.enabled:
            self.estimator.remember_actions(actions)

    def _assign(self, plan, poses, previous=None):
        living = {pose.agent_id for pose in poses}
        if previous and previous.sections and plan.sections:
            costs = np.linalg.norm(
                np.array([s.center for s in previous.sections])[:, None, :]
                - np.array([s.center for s in plan.sections])[None, :, :], axis=2,
            )
            old, new = linear_sum_assignment(costs)
            remap = dict(zip(old.tolist(), new.tolist()))
            plan.assignments = {agent_id: remap[section_id]
                                for agent_id, section_id in previous.assignments.items()
                                if agent_id in living and section_id in remap}
        plan.assignments = {key: value for key, value in plan.assignments.items() if key in living}
        loads = [0] * len(plan.sections)
        pending = []
        # Preserve existing owners; births get free sections before sharing.
        for pose in sorted(poses, key=lambda p: p.agent_id):
            section_id = plan.assignments.get(pose.agent_id)
            if section_id is None or (previous is not None and loads[section_id]):
                pending.append(pose)
            else:
                loads[section_id] += 1
        for pose in pending:
            section_id = min(range(len(plan.sections)), key=lambda index: (
                loads[index], float(np.linalg.norm(plan.sections[index].center - pose.position)), index,
            ))
            plan.assignments[pose.agent_id] = section_id
            loads[section_id] += 1

    def _hint(self, pose, plan):
        if pose.uncertainty > self.config.estimator.max_position_uncertainty:
            return None
        own_id = plan.assignments[pose.agent_id]
        occupied = set(plan.assignments.values()) - {own_id}
        if not occupied:
            return None
        own = plan.sections[own_id].center - pose.position
        nearest_other = min(np.linalg.norm(plan.sections[index].center - pose.position) for index in occupied)
        # Distance differences can move by twice the position error. This dead
        # band also prevents jitter on boundaries when estimates are accurate.
        if np.linalg.norm(own) <= nearest_other + self.config.center_distance_margin + 2 * pose.uncertainty:
            return None
        vector = rotate(own, -pose.heading)
        return SectionHint(tuple(float(value) for value in vector), self.config.push_strength)

    def instructions(self, agent_states, sim_time, trait_ratings=None, territory_policy=False):
        if not agent_states:
            self.reset()
            return {}
        if not (self.config.enabled or self.config.mapping_enabled or self.config.exploration.enabled):
            return {}
        if (self.last_time is not None and sim_time < self.last_time) or any(
            state["agent_id"] in self.estimator.poses
            and state["age"] < self.estimator.poses[state["agent_id"]].age for state in agent_states
        ):
            self.reset()
        if self.last_time == sim_time:
            return self.hints.copy()
        self.trait_ratings = {} if trait_ratings is None else dict(trait_ratings)
        self.estimator.update(agent_states, sim_time)
        self._update_phase(sim_time)
        self.biome_estimator.update(self.estimator.groups, self.estimator.poses, sim_time)
        managed_groups = {key for key, group in self.estimator.groups.items()
                          if territory_policy and self.population_phase and group.anchored
                          and (group.world_size is not None or
                               (group.known_width is not None and group.known_height is not None))}
        self.exploration_hints = self.exploration.instructions(
            agent_states, self.estimator, sim_time, trait_ratings=self.trait_ratings,
            population_phase=self.population_phase, managed_groups=managed_groups)
        self.plans = {key: plan for key, plan in self.plans.items() if key in self.estimator.groups}
        self.hints = {}
        if not self.config.enabled:
            self.last_time = sim_time
            return {}
        for group_id, group in self.estimator.groups.items():
            if group_id in managed_groups:
                self.plans.pop(group_id, None)
                continue
            poses = [pose for pose in self.estimator.poses.values() if pose.group_id == group_id]
            plan = self.plans.get(group_id)
            if (plan is None or plan.revision != group.revision
                    or sim_time - plan.updated_at >= self.config.replan_interval_seconds):
                grid = food_grid(group, self.config)
                if grid is None:
                    continue
                sections = partition_food(grid, min(len(poses), self.config.max_sections))
                if not sections:
                    continue
                previous = plan
                plan = GroupPlan(sections, sim_time, group.revision)
                self._assign(plan, poses, previous)
                self.plans[group_id] = plan
            else:
                self._assign(plan, poses)
            for pose in poses:
                hint = self._hint(pose, plan)
                if hint and hint.strength > 0:
                    self.hints[pose.agent_id] = hint
        self.last_time = sim_time
        return self.hints.copy()

    def snapshot(self):
        """JSON-compatible estimated state for diagnostics; never ground truth."""
        snapshot = self.estimator.snapshot()
        snapshot["sim_time"] = self.last_time
        snapshot["population"] = dict(self.population_snapshot)
        snapshot["harvest"] = getattr(self, "harvest_snapshot", {})
        snapshot["phase"] = ("population" if self.population_phase else "alignment"
                             if self.config.population_after_alignment else "exploration")
        snapshot["shared_frame_established_at"] = self.shared_frame_established_at
        for group in snapshot["groups"]:
            group["biome_estimate"] = self.biome_estimator.snapshot(group["group_id"])
            plan = self.plans.get(group["group_id"])
            group["updated_at"] = None if plan is None else plan.updated_at
            group["sections"] = [] if plan is None else [
                {"section_id": s.section_id, "center": s.center.tolist(),
                 "bounds": list(s.bounds), "food_weight": s.food_weight} for s in plan.sections]
            group["assignments"] = {} if plan is None else dict(plan.assignments)
            frontier = self.exploration.frontiers.get(group["group_id"])
            group["mapping_coverage"] = None if frontier is None else frontier.coverage
        roles = self.exploration.snapshot()
        for agent in snapshot["agents"]:
            agent["push_requested"] = agent["agent_id"] in self.hints
            agent.update(roles.get(agent["agent_id"], {"role": "unassigned"}))
            task = self.harvest_snapshot.get("tasks", {}).get(agent["agent_id"])
            if task is not None:
                agent.update(role="owner", objective=task["kind"])
            rating = self.trait_ratings.get(agent["agent_id"])
            if rating is not None:
                agent.update(trait_score=rating.score, elite=rating.elite, low_rank=rating.low_rank,
                             trait_percentile=rating.percentile, trait_ratios=dict(rating.ratios))
        return snapshot
