"""Three colony rules: limit births, share productive space, collect food.

Only public observations and the estimated map enter this coordinator. Local
collision avoidance and predator escape remain in ExpertPolicy.
"""

from dataclasses import dataclass
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import linear_sum_assignment

from src.utils.controllers.metabolism import MetabolismTracker
from src.utils.controllers.navigation import Navigator
from src.utils.controllers.policy_inputs import HarvestHint, ReproductionHint
from src.utils.controllers.world_estimator import relative_vector, rotate


class SimpleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    population_early: int = Field(default=12, ge=1)
    population_middle: int = Field(default=6, ge=1)
    population_late: int = Field(default=3, ge=1)
    middle_seconds: float = Field(default=900., gt=0)
    late_seconds: float = Field(default=1800., gt=0)
    # Counting cutoff only; retirement requires measured senescence.
    renewal_age: float = Field(default=40., gt=0)
    parent_reserve: float = Field(default=75., ge=0)
    birth_interval_seconds: float = Field(default=8., gt=0)
    territory_interval_seconds: float = Field(default=30., gt=0)
    territory_cell_size: float = Field(default=120., gt=0)
    food_interval_seconds: float = Field(default=.5, gt=0)
    fruit_memory_seconds: float = Field(default=8., gt=0)
    map_interval_seconds: float = Field(default=.5, gt=0)
    late_map_interval_seconds: float = Field(default=2., gt=0)
    scan_interval_seconds: float = Field(default=4., gt=0)
    late_scan_interval_seconds: float = Field(default=15., gt=0)
    patrol_interval_seconds: float = Field(default=12., gt=0)
    late_patrol_interval_seconds: float = Field(default=40., gt=0)

    @model_validator(mode="after")
    def valid_schedule(self):
        if not self.population_early >= self.population_middle >= self.population_late:
            raise ValueError("population targets must decrease or stay constant")
        if self.late_seconds <= self.middle_seconds:
            raise ValueError("late_seconds must exceed middle_seconds")
        return self

    def fraction(self, now):
        return min(1., max(0., now / self.late_seconds))

    def interpolate(self, now, early, late):
        return early + (late - early) * self.fraction(now)


@dataclass
class KnownFruit:
    position: np.ndarray
    last_seen: float


class SimpleCoordinator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.active = False
        self.frame = None
        self.last_time = None
        self.last_birth = -math.inf
        self.next_territories = self.next_food = 0.
        self.population_target = self.config.population_early
        self.population_plan = {}
        self.homes, self.goals, self.scans, self.patrols = {}, {}, {}, {}
        self.sweeps, self.blocked_until = {}, {}
        self.tracks, self.assignments, self.tasks, self.last_ages = {}, {}, {}, {}
        self.next_id = 0
        self.points = np.empty((0, 2))
        self.owners = np.array([], dtype=int)
        self.navigator = Navigator()
        self.metabolic_tracker = MetabolismTracker()
        self.retired = set()
        self.scout_goals = {}
        self.cached = ({}, {})

    def _births(self, states, now, population, mechanics, aligned):
        cfg = self.config
        self.population_target = (cfg.population_late if now >= cfg.late_seconds else
                                  cfg.population_middle if now >= cfg.middle_seconds else cfg.population_early)
        young = sum(s["age"] < cfg.renewal_age for s in states)
        # Older living agents do not consume a birth slot. Space births fast
        # enough to replenish this age window, with modest headroom for deaths.
        needed = young < self.population_target
        interval = min(cfg.birth_interval_seconds, .8 * cfg.renewal_age / self.population_target)
        reserve = cfg.parent_reserve
        threshold = mechanics.spawn_energy_cost + reserve + 2.
        hints = {s["agent_id"]: ReproductionHint(threshold, False, reserve) for s in states}
        candidates = [s for s in states if s["energy"] > threshold and s["age"] >= 5.
                      and s["agent_id"] not in self.retired]
        selected = None
        if (needed and (aligned or len(states) < 2) and candidates
                and now - self.last_birth >= interval - 1e-9):
            parent = max(candidates, key=lambda s: (population.ratings[s["agent_id"]].score,
                                                   s["energy"], -s["agent_id"]))
            selected = parent["agent_id"]
            hints[selected] = ReproductionHint(threshold, True, reserve)
        self.population_plan = dict(population=len(states), target=self.population_target,
                                    young=young, counted_age_limit=cfg.renewal_age, retired=len(self.retired),
                                    birth_interval_seconds=interval, selected_parent=selected,
                                    reason="birth slot available" if selected is not None else "hold population")
        return hints

    def _territories(self, states, poses, group, planner, now):
        ids = sorted(s["agent_id"] for s in states)
        if not ids:
            self.homes.clear()
            self.goals.clear()
            self.points = np.empty((0, 2))
            self.owners = np.array([], dtype=int)
            return
        if set(ids) == set(self.homes) and now < self.next_territories:
            return
        self.next_territories = now + self.config.territory_interval_seconds
        pitch = self.config.territory_cell_size
        potentials = planner.config.estimator.biome_food_potential
        # One candidate per coarse cell bounds work and avoids counting a
        # repeatedly visited location as more productive.
        cells = {}
        layer = planner.biome_estimator.layers.get(group.group_id)
        if layer:
            labels, confidence = np.array(layer["labels"]), np.array(layer["confidence"])
            x0, y0, x1, y1 = layer["bounds"]
            ny, nx = labels.shape
            for y in range(0, ny, max(1, round(pitch * ny / (y1 - y0)))):
                for x in range(0, nx, max(1, round(pitch * nx / (x1 - x0)))):
                    label = labels[y, x]
                    potential = potentials.get(layer["palette"][label], 0.) if label >= 0 else 0.
                    if potential <= 0 or confidence[y, x] < .3:
                        continue
                    point = np.array([x0 + (x + .5) * (x1 - x0) / nx,
                                      y0 + (y + .5) * (y1 - y0) / ny])
                    cells[tuple(np.floor(point / pitch).astype(int))] = (point, potential)
        for sample in group.biomes.values():
            potential = potentials.get(sample.biome, 0.)
            key = tuple(np.floor(sample.position / pitch).astype(int))
            if potential > 0:
                cells[key] = (sample.position.copy(), potential)
            else:
                cells.pop(key, None)
        if not cells:
            cells = {i: (poses[i].position.copy(), 1.) for i in ids}
        points, weights = zip(*cells.values())
        self.points, weights = np.array(points), np.array(weights)
        # Weighted farthest-point seeding spreads homes across productive land.
        selected = [int(np.argmax(weights))]
        while len(selected) < min(len(ids), len(self.points)):
            distances = np.min(np.sum((self.points[:, None] - self.points[selected]) ** 2, axis=2), axis=1)
            scores = distances * weights
            scores[selected] = -1
            selected.append(int(np.argmax(scores)))
        centers = self.points[selected]
        # Match homes to agents with minimal relocation, keeping old ownership
        # as the reference when the map gets a better biome estimate.
        origins = np.array([self.homes.get(i, poses[i].position) for i in ids])
        rows, columns = linear_sum_assignment(np.sum((origins[:, None] - centers) ** 2, axis=2))
        homes = {ids[r]: centers[c].copy() for r, c in zip(rows, columns)}
        for i in ids:
            homes.setdefault(i, poses[i].position.copy())
        self.homes = homes
        nearest = np.argmin(np.sum((self.points[:, None] - np.array([homes[i] for i in ids])) ** 2, axis=2), axis=1)
        self.owners = np.array(ids)[nearest]
        self.goals = {i: p for i, p in self.goals.items() if i in homes}

    def _food(self, states, poses, now, mechanics):
        seen, hearing = set(), []
        for state in sorted(states, key=lambda s: (poses[s["agent_id"]].uncertainty, s["agent_id"])):
            i = state["agent_id"]
            pose = poses[i]
            if pose.uncertainty > 8. or self.last_ages.get(i) == state["age"]:
                continue
            hearing.append((pose.position, max(0., state["hearing_radius"] - 8. - pose.uncertainty)))
            used = set()
            for observation in state["observations"]:
                if observation["type"] != "Fruit":
                    continue
                point = pose.position + rotate(relative_vector(observation), pose.heading)
                candidates = [(float(np.sum((t.position - point) ** 2)), key)
                              for key, t in self.tracks.items() if key not in used]
                distance, key = min(candidates, default=(math.inf, -1))
                if distance > 16.:
                    if len(self.tracks) >= 128:
                        continue
                    key = self.next_id
                    self.next_id += 1
                    self.tracks[key] = KnownFruit(point, now)
                if key not in seen:
                    self.tracks[key].position = point
                self.tracks[key].last_seen = now
                seen.add(key)
                used.add(key)
        self.tracks = {key: t for key, t in self.tracks.items()
                       if now - t.last_seen <= self.config.fruit_memory_seconds
                       and (key in seen or not any(np.sum((t.position - p) ** 2) < radius ** 2
                                                  for p, radius in hearing))}
        self.last_ages = {s["agent_id"]: s["age"] for s in states}
        self.assignments = {i: key for i, key in self.assignments.items()
                            if i in poses and key in self.tracks and i not in self.retired}
        if now < self.next_food:
            return
        self.next_food = now + self.config.food_interval_seconds
        # Nearest useful agent wins; a small preference retains existing goals.
        pairs = []
        for state in states:
            i = state["agent_id"]
            if i in self.retired or state["energy"] > state["max_energy"] - 25. or poses[i].uncertainty > 8.:
                continue
            factor = max(.3, self.movement_factors.get(state["biome"], 1.))
            for key, track in self.tracks.items():
                if self._blocked(i, track.position, now):
                    continue
                distance = float(np.linalg.norm(track.position - poses[i].position))
                cost = distance / factor * mechanics.walking_energy_per_unit
                if cost > min(35., state["energy"] - 5.):
                    continue
                pairs.append((distance - (30. if self.assignments.get(i) == key else 0.), i, key))
        self.assignments = {}
        claimed = set()
        for _, i, key in sorted(pairs):
            if i not in self.assignments and key not in claimed:
                self.assignments[i] = key
                claimed.add(key)

    @staticmethod
    def _goal_key(agent_id, point):
        return (agent_id, *np.round(np.asarray(point) / 10.).astype(int))

    def _blocked(self, agent_id, point, now):
        return now < self.blocked_until.get(self._goal_key(agent_id, point), 0.)

    def _scout_hint(self, state, pose, frontier, now):
        """Give each aging scout a persistent, distinct unvisited frontier."""
        i = state["agent_id"]
        if pose.uncertainty > 8.:
            self.tasks[i]["kind"] = "aging scout; relocalize"
            return HarvestHint((0., 0.), None, 0., False, look_direction=math.pi / 4,
                               scan_while_stationary=True, allow_local_food=False, retired=True)
        goal = self.scout_goals.get(i)
        if goal is not None and (np.linalg.norm(goal - pose.position) <= 25.
                or not np.any(np.linalg.norm(frontier.targets - goal, axis=1) < 1.)
                or self._blocked(i, goal, now)):
            self.scout_goals.pop(i, None)
            self.navigator.release(i)
            goal = None
        if goal is None:
            current_cell = np.floor(pose.position / frontier.cell_size).astype(int)
            candidates = [p for p, cell in zip(frontier.targets, frontier.cells)
                          if np.any(cell != current_cell) and not self._blocked(i, p, now)
                          and all(np.linalg.norm(p - other) >= .65 * frontier.cell_size
                                  for owner, other in self.scout_goals.items() if owner != i)]
            goal = min(candidates, key=lambda p: float(np.linalg.norm(p - pose.position)), default=None)
            if goal is not None:
                self.scout_goals[i] = goal.copy()
        vector = (0., 0.)
        if goal is not None:
            route = self.navigator.steer(i, pose.position, goal, now, arrival_radius=15.)
            if route.blocked:
                self.blocked_until[self._goal_key(i, goal)] = now + 30.
                self.scout_goals.pop(i, None)
                self.navigator.release(i)
            elif route.waypoint is not None:
                vector = tuple(rotate(route.waypoint - pose.position, -pose.heading))
        self.tasks[i].update(kind="aging scout; explore unseen areas" if goal is not None else "aging scout; no free frontier",
                             destination=None if goal is None else goal.tolist())
        return HarvestHint(vector, None, 0., False, survey=True, allow_local_food=False, retired=True)

    def update(self, states, now, planner, population, mechanics, base_threshold):
        if self.last_time is not None and (now < self.last_time or any(
                s["agent_id"] in self.metabolic_tracker.history
                and s["age"] < self.metabolic_tracker.history[s["agent_id"]].state["age"] for s in states)):
            self.reset()
        if now == self.last_time:
            return self.cached
        self.metabolic_tracker.walking_cost = mechanics.walking_energy_per_unit
        self.metabolic_tracker.sprinting_cost = mechanics.sprinting_energy_per_unit
        self.metabolic_tracker.spawn_cost = mechanics.spawn_energy_cost
        self.metabolic_tracker.sprint_min_energy_fraction = mechanics.sprint_min_energy_fraction
        self.metabolic_tracker.update(states, now)
        previous_retired = self.retired
        self.retired = {i for i, record in self.metabolic_tracker.history.items() if record.aging}
        self.scout_goals = {i: p for i, p in self.scout_goals.items() if i in self.retired}
        breeding = self._births(states, now, population, mechanics, planner.population_phase)
        groups = [g for g in planner.estimator.groups.values() if g.anchored]
        group = max(groups, key=lambda g: sum(p.group_id == g.group_id for p in planner.estimator.poses.values()), default=None)
        self.active = planner.population_phase and group is not None
        self.tasks = {}
        hints = {}
        for state in states:
            i = state["agent_id"]
            if i in self.retired:
                self.assignments.pop(i, None)
                for memory in (self.homes, self.goals, self.scans, self.patrols, self.sweeps):
                    memory.pop(i, None)
                if i not in previous_retired:
                    self.navigator.release(i)
                hints[i] = HarvestHint((min(state["speed"], state["sprint_speed"]), 0.), None, 0., False,
                                       survey=True, allow_local_food=False, retired=True)
                self.tasks[i] = dict(kind="aging scout; align map", passive_energy_per_second=self.metabolic_tracker.rates[i])
        if self.active:
            frame = (group.group_id, group.frame_revision)
            if frame != self.frame:
                self.homes.clear()
                self.goals.clear()
                self.tracks.clear()
                self.assignments.clear()
                self.blocked_until.clear()
                self.scout_goals.clear()
                self.next_territories = self.next_food = now
                self.frame = frame
            poses = {i: p for i, p in planner.estimator.poses.items() if p.group_id == group.group_id}
            members = [s for s in states if s["agent_id"] in poses]
            self.movement_factors = planner.config.estimator.biome_movement_factors
            self._territories([s for s in members if s["agent_id"] not in self.retired], poses, group, planner, now)
            self._food(members, poses, now, mechanics)
            self.navigator.update(group, now)
            self.navigator.prune(poses)
            self.scans = {i: t for i, t in self.scans.items() if i in poses}
            self.patrols = {i: t for i, t in self.patrols.items() if i in poses}
            self.sweeps = {i: t for i, t in self.sweeps.items() if i in poses}
            self.blocked_until = {key: t for key, t in self.blocked_until.items() if key[0] in poses and now < t}
            scouts = sorted((s for s in members if s["agent_id"] in self.retired), key=lambda s: s["agent_id"])
            if scouts:
                frontier = planner.exploration._frontier_plan(group, list(poses.values()), now)
                for state in scouts:
                    hints[state["agent_id"]] = self._scout_hint(state, poses[state["agent_id"]], frontier, now)
            for state in members:
                i, energy = state["agent_id"], state["energy"]
                if i in self.retired:
                    continue
                pose, home = poses[i], self.homes[state["agent_id"]]
                if pose.uncertainty > 8.:
                    continue  # Keep local food/obstacle behavior until relocalized.
                key = self.assignments.get(i)
                destination = self.tracks[key].position if key is not None else None
                kind = "collect fruit" if key is not None else "rest and watch"
                if destination is None and energy > 60.:
                    if np.linalg.norm(home - pose.position) > 100. and not self._blocked(i, home, now):
                        destination, kind = home, "spread to food area"
                    elif now >= self.patrols.get(i, 0.):
                        choices = self.points[self.owners == i]
                        choices = choices[np.linalg.norm(choices - home, axis=1) <= 180.]
                        if len(choices):
                            # Deterministic short patrol, different phase per agent.
                            self.goals[i] = choices[(int(now / self.config.patrol_interval_seconds) + i) % len(choices)]
                        interval = self.config.interpolate(now, self.config.patrol_interval_seconds,
                                                           self.config.late_patrol_interval_seconds)
                        self.patrols[i] = now + interval
                    if destination is None and i in self.goals:
                        destination, kind = self.goals[i], "patrol food area"
                vector = (0., 0.)
                if destination is not None and self._blocked(i, destination, now):
                    destination = None
                    self.goals.pop(i, None)
                if destination is not None:
                    route = self.navigator.steer(i, pose.position, destination, now,
                                                 arrival_radius=5. if key is not None else 25.)
                    if route.blocked or route.status == "arrived":
                        self.goals.pop(i, None)
                        if route.blocked:
                            # Remember failed goals: rebuilding the same A* path
                            # every tick is expensive and cannot free the agent.
                            self.blocked_until[self._goal_key(i, destination)] = now + 30.
                        if route.blocked and key is not None:
                            self.tracks.pop(key, None)
                            self.assignments.pop(i, None)
                        self.navigator.release(i)
                    elif route.waypoint is not None:
                        vector = tuple(rotate(route.waypoint - pose.position, -pose.heading))
                else:
                    self.navigator.pause(i, now)
                if key is None and now >= self.scans.get(i, 0.) and not self.sweeps.get(i, 0):
                    self.sweeps[i] = 8
                    self.scans[i] = now + self.config.interpolate(now, self.config.scan_interval_seconds,
                                                                self.config.late_scan_interval_seconds)
                scan = key is None and self.sweeps.get(i, 0) > 0
                if scan:
                    self.sweeps[i] -= 1
                hints[i] = HarvestHint(vector, key, 0., False, survey=key is None,
                                       look_direction=math.pi / 4 if scan else None,
                                       scan_while_stationary=scan, allow_local_food=False)
                self.tasks[i] = dict(kind=kind, destination=None if destination is None else destination.tolist())
        self.last_time = now
        self.cached = hints, breeding
        return self.cached

    def remember_actions(self, actions, now):
        self.metabolic_tracker.remember_actions(actions)
        if any(action.spawn_agent for action in actions):
            self.last_birth = now

    def snapshot(self):
        return dict(enabled=True, mode="simple", active=self.active, population_target=self.population_target,
                    retired_ids=sorted(self.retired), scout_targets={i: p.tolist() for i, p in self.scout_goals.items()},
                    reproduction_plan=dict(self.population_plan), group_id=None if self.frame is None else self.frame[0],
                    tasks=self.tasks, navigation=self.navigator.snapshot(), assignments=dict(self.assignments),
                    coverage=dict(homes={i: p.tolist() for i, p in self.homes.items()},
                                  points=[dict(position=p.tolist(), owner=int(i)) for p, i in zip(self.points, self.owners)]),
                    fruits=[dict(track_id=i, position=t.position.tolist(), ripe_in_seconds=0.)
                            for i, t in self.tracks.items()])
