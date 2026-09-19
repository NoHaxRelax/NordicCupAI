"""Central fruit reservations and breeding, using observations and estimated poses.

Fruit has no public ID, age, size, or energy. Spatial tracks therefore provide
age estimates, not privileged simulator data. Time since first sighting is a
lower bound; an empty preceding observation brackets the birth more closely.
"""

from dataclasses import dataclass
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial import cKDTree

from models.exploration.exploration import _path_clear
from models.exploration.coverage import CoverageConfig, CoverageCoordinator
from models.exploration.navigation import Navigator
from models.exploration.policy_inputs import HarvestHint, ReproductionHint, observed_edges
from models.exploration.world_estimator import rotate


class HarvestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=False, strict=True)
    replan_seconds: float = Field(default=1.0, gt=0)
    match_radius: float = Field(default=4.0, gt=0)
    max_pose_uncertainty: float = Field(default=4.0, gt=0)
    max_tracks: int = Field(default=1500, ge=1)
    max_unseen_seconds: float = Field(default=20.0, gt=0)
    unknown_initial_age_seconds: float = Field(default=5.0, ge=0, le=20)
    initial_energy: float = Field(default=20.0, gt=0)
    maximum_energy: float = Field(default=60.0, gt=0)
    growth_per_second: float = Field(default=2.0, gt=0)
    lifetime_seconds: float = Field(default=50.0, gt=0)
    ripe_energy: float = Field(default=56.0, gt=0)
    pickup_clearance: float = Field(default=18.0, ge=15)
    emergency_energy: float = Field(default=90.0, gt=0)
    survival_reserve: float = Field(default=35.0, ge=0)
    minimum_population: int = Field(default=12, ge=1)
    maximum_population: int = Field(default=60, ge=1)
    food_budget_per_agent_second: float = Field(default=6.0, gt=0)
    birth_interval_seconds: float = Field(default=0.5, gt=0)
    parent_reserve: float = Field(default=100.0, ge=0)
    minimum_breeder_trait_score: float = Field(default=1.0, ge=0)
    gene_backup_age: float = Field(default=40.0, ge=0)
    gene_backup_interval_seconds: float = Field(default=35.0, gt=0)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)

    @model_validator(mode="after")
    def valid_limits(self):
        if not self.initial_energy <= self.ripe_energy <= self.maximum_energy:
            raise ValueError("Require initial_energy <= ripe_energy <= maximum_energy")
        if self.minimum_population > self.maximum_population:
            raise ValueError("minimum_population cannot exceed maximum_population")
        if (self.ripe_energy - self.initial_energy) / self.growth_per_second >= self.lifetime_seconds:
            raise ValueError("Fruit must ripen before its lifetime ends")
        return self


@dataclass
class FruitTrack:
    track_id: int
    position: np.ndarray
    first_seen: float
    last_seen: float
    estimated_birth: float
    birth_bracketed: bool
    uncertainty: float


class HarvestCoordinator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 0
        self.frame = None
        self.previous_views = {}
        self.assignments = {}
        self.blocked_until = {}
        self.last_ages = {}
        self.last_time = None
        self.dt = .1
        self.next_plan = 0.
        self.next_birth = 0.
        self.next_gene_backup = 0.
        self.gene_backup_ids = set()
        self.discoveries = []
        self.population_target = self.config.minimum_population
        self.hints = {}
        self.breeding = {}
        self.active = False
        self.recovering_population = False
        self.movement_factors = {}
        self.coverage = CoverageCoordinator(self.config.coverage)
        self.navigator = Navigator()
        self.tasks = {}

    def energy(self, track, now):
        cfg = self.config
        return min(cfg.maximum_energy, cfg.initial_energy + cfg.growth_per_second * max(0., now - track.estimated_birth))

    def _visible(self, position, view, margin):
        _, origin, heading, hearing, vision, cone, edges, _ = view
        vector = rotate(position - origin, -heading)
        distance = float(np.linalg.norm(vector))
        if distance < hearing - margin:
            return True
        angle = math.atan2(vector[1], vector[0])
        return (distance < vision - margin and abs(angle) < cone / 2 - .03
                and _path_clear(angle, distance, edges, 0.))

    def _observe(self, states, poses, now):
        cfg = self.config
        seen, views = set(), {}
        for state in sorted(states, key=lambda s: (poses[s["agent_id"]].uncertainty, s["agent_id"])):
            agent_id = state["agent_id"]
            pose = poses[agent_id]
            if (pose.uncertainty > cfg.max_pose_uncertainty
                    or self.last_ages.get(agent_id) == state["age"]):
                continue
            edges = observed_edges(state["observations"])
            observations = sorted((o for o in state["observations"] if o["type"] == "Fruit"),
                                  key=lambda o: (o["distance"], o["angle"]))
            positions = [pose.position + rotate(np.array([o["distance"] * math.cos(o["angle"]),
                                                         o["distance"] * math.sin(o["angle"])]), pose.heading)
                         for o in observations]
            view = (now, pose.position.copy(), pose.heading, state["hearing_radius"], state["vision_range"],
                    state["vision_angle"], edges, positions)
            views[agent_id] = view
            ids = sorted(self.tracks)
            tree = cKDTree([self.tracks[i].position for i in ids]) if ids else None
            used = set()
            for position in positions:
                candidates = [] if tree is None else tree.query_ball_point(position, cfg.match_radius)
                candidates = [ids[index] for index in candidates if ids[index] not in used]
                track_id = min(candidates, key=lambda i: (float(np.linalg.norm(self.tracks[i].position - position)), i), default=None)
                if track_id is None:
                    if len(self.tracks) >= cfg.max_tracks:
                        continue
                    previous = self.previous_views.get(agent_id)
                    bracketed = (previous is not None and now - previous[0] <= 1.
                                 and self._visible(position, previous, cfg.match_radius)
                                 and all(np.linalg.norm(position - old) > cfg.match_radius for old in previous[-1]))
                    birth = ((previous[0] + now) / 2 if bracketed else
                             max(0., now - cfg.unknown_initial_age_seconds))
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = FruitTrack(track_id, position.copy(), now, now, birth, bracketed, pose.uncertainty)
                    if bracketed:
                        self.discoveries.append(now)
                track = self.tracks[track_id]
                # Prefer the best observer in this step; repeated sightings
                # neither increase age nor turn two adjacent fruits into one.
                if track_id not in seen:
                    track.position = position.copy()
                    track.uncertainty = pose.uncertainty
                track.last_seen = now
                seen.add(track_id)
                used.add(track_id)
        expired = []
        for track_id, track in self.tracks.items():
            if (now - track.first_seen > cfg.lifetime_seconds or now - track.last_seen > cfg.max_unseen_seconds
                    or (track_id not in seen and any(self._visible(track.position, view,
                        cfg.match_radius + track.uncertainty) for view in views.values()))):
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]
        self.previous_views = views
        self.discoveries = [time for time in self.discoveries if now - time < 30]

    def _metabolism(self, state):
        # Max lifespan is not observed. Old agents need a conservative reserve
        # because the engine can add a per-step age drain after age 60.
        return 2. + (state["age"] * .01 / self.dt if state["age"] > 80 else 0.)

    def _candidate(self, state, pose, track, now, walking_cost):
        distance = float(np.linalg.norm(track.position - pose.position))
        speed = max(.01, min(state["speed"], state["sprint_speed"]))
        factor = max(.05, self.movement_factors.get(state["biome"], 1.))
        travel = distance / (speed * factor) * self.dt
        energy = self.energy(track, now + travel)
        wait = max(0., (self.config.ripe_energy - energy) / self.config.growth_per_second)
        movement_cost = distance / factor * walking_cost * 1.2
        urgent = (state["energy"] < self.config.emergency_energy or state["energy"] - movement_cost
                  - (travel + wait) * self._metabolism(state) < self.config.survival_reserve)
        # Establish/recover the colony before delaying its food for ripening.
        urgent = urgent or (self.recovering_population and state["energy"] < 260.)
        if urgent:
            wait = 0.
        energy = self.energy(track, now + travel + wait)
        net = energy - movement_cost
        if now + travel + wait > track.first_seen + self.config.lifetime_seconds - 1:
            return -1., urgent, wait
        urgency = 1. + 2. * max(0., 1. - state["energy"] / self.config.emergency_energy)
        utility = net * urgency / (3. + travel + wait)
        if state["energy"] > state["max_energy"] - energy:
            utility *= .75
        utility *= math.exp(-(now - track.last_seen) / self.config.max_unseen_seconds)
        return utility, urgent, wait

    def _assign(self, states, poses, now, population, mechanics):
        """Each owner chooses food locally; only emergencies cross territories.

        Keep a viable ripe target until it is collected. Hungry agents choose
        first and can borrow a reservation; no colony-wide assignment solver.
        """
        choices, emergencies = {}, set()
        # Ownership is shared across agents; query the map once per fruit.
        # Repeating the fine-grid lookup for every agent dominates large colonies.
        owners = {track_id: self.coverage.owner(track.position) for track_id, track in self.tracks.items()}
        for state in states:
            agent_id = state["agent_id"]
            pose = poses[agent_id]
            if pose.uncertainty > self.config.max_pose_uncertainty:
                continue
            emergency = state["energy"] < self.config.emergency_energy
            if emergency:
                emergencies.add(agent_id)
            patrol = self.coverage.hint(agent_id, pose)
            options = {}
            for track_id, track in self.tracks.items():
                if self.blocked_until.get((agent_id, track_id), 0.) > now:
                    continue
                owner = owners[track_id]
                if not emergency and owner is not None and owner != agent_id:
                    continue
                value, urgent, wait = self._candidate(state, pose, track, now, mechanics.walking_energy_per_unit)
                if value <= 0:
                    continue
                # Patrol the territory while its fruit develops. With no work
                # due, waiting near a developing fruit is an explicit task.
                if wait > 0 and not urgent and patrol is not None:
                    continue
                options[track_id] = value
            choices[agent_id] = options
        assigned, claimed = {}, set()
        order = sorted(choices, key=lambda i: (i not in emergencies,
                       -max(choices[i].values(), default=0), i))
        for agent_id in order:
            options = {t: v for t, v in choices[agent_id].items() if t not in claimed}
            previous = self.assignments.get(agent_id)
            if previous in options:
                target = previous
            elif options:
                target = min(options, key=lambda t: (-options[t], t))
            else:
                continue
            assigned[agent_id] = target
            claimed.add(target)
        self.assignments = assigned
        self.next_plan = now + self.config.replan_seconds

    def _breeders(self, states, poses, now, population, mechanics, base_threshold):
        cfg = self.config
        # Observed new-fruit arrivals plus the standing stock fund a bounded
        # population. This is an estimate of food supply, not a truth census.
        stock = sum(self.energy(track, now) for track in self.tracks.values())
        supply = len(self.discoveries) * cfg.maximum_energy / 30 + stock / 30
        self.population_target = max(cfg.minimum_population, min(cfg.maximum_population,
                                     int(supply / cfg.food_budget_per_agent_second)))
        hints = {s["agent_id"]: ReproductionHint(base_threshold, False) for s in states}
        self.gene_backup_ids = set()
        if len(population.ratings) >= cfg.maximum_population or now < self.next_birth - 1e-9:
            return hints
        best_score = max((r.score for r in population.ratings.values()), default=1.)
        candidates = []
        for state in states:
            agent_id = state["agent_id"]
            if poses[agent_id].uncertainty > cfg.max_pose_uncertainty:
                continue
            rating = population.ratings[agent_id]
            valuable = rating.elite or rating.score >= best_score - 1e-9
            young_carriers = sum(other_id != agent_id and population.ages[other_id] < cfg.gene_backup_age
                                 and other.score >= .95 * rating.score
                                 for other_id, other in population.ratings.items()) if valuable else 2
            backup = (valuable and rating.score >= cfg.minimum_breeder_trait_score
                      and young_carriers < 2
                      and state["age"] >= cfg.gene_backup_age
                      and now - population.last_birth_request.get(agent_id, -math.inf) >= cfg.gene_backup_interval_seconds
                      and now >= self.next_gene_backup)
            if len(population.ratings) >= self.population_target and not backup:
                continue
            if (len(population.ratings) >= cfg.minimum_population
                    and rating.score < cfg.minimum_breeder_trait_score):
                continue
            # One source of trait thresholds and normal birth cooldowns.
            ordinary = population.reproduction_hint(agent_id, now, base_threshold)
            if ordinary is not None and not ordinary.allowed:
                continue
            threshold = ordinary.energy_threshold if ordinary is not None else base_threshold
            if state["energy"] <= max(threshold, mechanics.spawn_energy_cost + cfg.parent_reserve + 2.):
                continue
            candidates.append((-rating.score, -state["energy"], agent_id, threshold, backup))
        if candidates:
            _, _, agent_id, threshold, backup = min(candidates)
            hints[agent_id] = ReproductionHint(threshold, True, cfg.parent_reserve)
            if backup:
                self.gene_backup_ids.add(agent_id)
        return hints

    def update(self, states, now, planner, population, mechanics, base_threshold):
        if not self.config.enabled:
            return {}, {}
        if self.last_time is not None and (now < self.last_time or any(
                s["age"] < self.last_ages.get(s["agent_id"], -1) for s in states)):
            self.reset()
        if now == self.last_time:
            return self.hints.copy(), self.breeding.copy()
        if self.last_time is not None and now > self.last_time:
            self.dt = now - self.last_time
        self.last_time = now
        poses = planner.estimator.poses
        if hasattr(planner.estimator, "config"):
            self.movement_factors = planner.estimator.config.biome_movement_factors
        groups = planner.estimator.groups
        eligible = [s for s in states if s["agent_id"] in poses and groups[poses[s["agent_id"]].group_id].anchored]
        self.active = planner.population_phase and bool(eligible)
        self.recovering_population = len(population.ratings) < self.config.minimum_population
        self.hints, self.breeding = {}, {}
        self.tasks = {}
        if not self.active:
            return {}, {}
        # Newborns can briefly remain disconnected while the established
        # absolute group continues harvesting. Never mix independent frames.
        group_id = min((poses[s["agent_id"]].group_id for s in eligible),
                       key=lambda key: (-sum(poses[s["agent_id"]].group_id == key for s in eligible), key))
        eligible = [s for s in eligible if poses[s["agent_id"]].group_id == group_id]
        frame = (group_id, groups[group_id].frame_revision)
        if frame != self.frame:
            self.tracks.clear()
            self.previous_views.clear()
            self.assignments.clear()
            self.discoveries.clear()
            self.next_plan = now
            self.frame = frame
            self.coverage.reset()
            self.navigator.reset()
        self._observe(eligible, poses, now)
        group = groups[group_id]
        biome_layer = getattr(getattr(planner, "biome_estimator", None), "layers", {}).get(group_id)
        self.coverage.update(eligible, poses, groups[group_id], self.previous_views,
                             self._visible, population.ratings, now, self.config.max_pose_uncertainty,
                             biome_layer=biome_layer, fruits=self.tracks.values())
        self.navigator.update(group, now)
        self.last_ages = {s["agent_id"]: s["age"] for s in states}
        self.blocked_until = {key: end for key, end in self.blocked_until.items() if end > now}
        living = {s["agent_id"] for s in eligible}
        self.navigator.prune(living)
        self.assignments = {i: t for i, t in self.assignments.items() if i in living and t in self.tracks}
        if now >= self.next_plan - 1e-9:
            self._assign(eligible, poses, now, population, mechanics)
        for state in eligible:
            agent_id = state["agent_id"]
            if poses[agent_id].uncertainty > self.config.max_pose_uncertainty:
                continue
            track = self.tracks.get(self.assignments.get(agent_id))
            pose = poses[agent_id]
            waiting, look = False, None
            if track is None:
                patrol = self.coverage.hint(agent_id, pose)
                home = self.coverage.homes.get(agent_id)
                if (patrol is None and home is not None and self.coverage.owner(pose.position) != agent_id
                        and self.blocked_until.get((agent_id, "home"), 0.) <= now):
                    destination = np.array(home)
                    kind = "return to territory"
                elif patrol is None:
                    self.navigator.release(agent_id)
                    # The old exploration policy is only a fallback until a
                    # territory map exists; afterwards idling is deliberate.
                    vector = (0., 0.) if self.coverage.homes else None
                    self.hints[agent_id] = HarvestHint(vector, None, 0., False)
                    self.tasks[agent_id] = dict(kind="idle" if vector else "local exploration")
                    continue
                else:
                    destination = pose.position + rotate(np.array(patrol.vector), pose.heading)
                    look = patrol.look_direction
                    kind = "scout patch" if agent_id in self.coverage.gap_targets else "patrol territory"
            else:
                _, urgent, _ = self._candidate(state, pose, track, now, mechanics.walking_energy_per_unit)
                offset = track.position - pose.position
                distance = float(np.linalg.norm(offset))
                waiting = not urgent and self.energy(track, now) < self.config.ripe_energy
                if waiting:
                    offset *= max(0., distance - self.config.pickup_clearance) / max(distance, 1e-9)
                destination = pose.position + offset
                kind = "wait for ripening" if waiting and np.linalg.norm(offset) < 1 else "collect fruit"
            route = self.navigator.steer(agent_id, pose.position, destination, now,
                                         waiting=kind == "wait for ripening")
            self.tasks[agent_id] = dict(kind=kind, destination=destination.tolist(),
                                       navigation=route.status,
                                       remaining=route.remaining if math.isfinite(route.remaining) else None)
            if route.blocked:
                if track is not None:
                    self.blocked_until[(agent_id, track.track_id)] = now + 20.
                    self.assignments.pop(agent_id, None)
                elif kind == "return to territory":
                    self.blocked_until[(agent_id, "home")] = now + 20.
                else:
                    self.coverage.reject_target(agent_id, now)
                self.navigator.release(agent_id)
                self.next_plan = now
                self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False)
                self.tasks[agent_id]["kind"] = "blocked; choose another target"
                continue
            vector = np.zeros(2) if route.waypoint is None else rotate(route.waypoint - pose.position, -pose.heading)
            self.hints[agent_id] = HarvestHint(tuple(float(v) for v in vector),
                None if track is None else track.track_id, 0. if track is None else self.energy(track, now),
                waiting, survey=track is None, look_direction=look)
        self.breeding = self._breeders(eligible, poses, now, population, mechanics, base_threshold)
        # The colony's food budget and birth slot also cover newborns which
        # briefly have their own coordinate frame.
        for state in states:
            self.breeding.setdefault(state["agent_id"], ReproductionHint(base_threshold, False))
        return self.hints.copy(), self.breeding.copy()

    def remember_actions(self, actions, now):
        if self.active and any(action.spawn_agent for action in actions):
            self.next_birth = now + self.config.birth_interval_seconds
            if any(a.spawn_agent and a.agent_id in self.gene_backup_ids for a in actions):
                self.next_gene_backup = now + 5.

    def snapshot(self):
        now = self.last_time or 0.
        return dict(enabled=self.config.enabled, active=self.active, population_target=self.population_target,
                    group_id=None if self.frame is None else self.frame[0],
                    gene_backup_ids=sorted(self.gene_backup_ids),
                    tasks=self.tasks, navigation=self.navigator.snapshot(),
                    coverage=self.coverage.snapshot(),
                    assignments=dict(self.assignments), fruits=[dict(track_id=t.track_id,
                    position=t.position.tolist(), first_seen=t.first_seen, last_seen=t.last_seen,
                    estimated_energy=self.energy(t, now), birth_bracketed=t.birth_bracketed,
                    ripe_in_seconds=max(0., (self.config.ripe_energy - self.energy(t, now)) / self.config.growth_per_second))
                    for t in sorted(self.tracks.values(), key=lambda t: t.track_id)])
