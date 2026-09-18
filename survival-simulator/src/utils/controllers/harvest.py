"""Central fruit reservations and breeding, using observations and estimated poses.

Fruit has no public ID, age, size, or energy. Spatial tracks therefore provide
age estimates, not privileged simulator data. Time since first sighting is a
lower bound; an empty preceding observation brackets the birth more closely.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial import cKDTree

from src.utils.controllers.exploration import _path_clear
from src.utils.controllers.coverage import CoverageConfig, CoverageCoordinator
from src.utils.controllers.navigation import Navigator
from src.utils.controllers.metabolism import MetabolismTracker
from src.utils.controllers.travel_cost import PublicBiomeTravel
from src.utils.controllers.conservation import ConservationConfig, ConservationSchedule
from src.utils.controllers.policy_inputs import HarvestHint, ReproductionHint, observed_edges
from src.utils.controllers.world_estimator import rotate


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
    survival_population: int = Field(default=2, ge=1)
    scouting_population_early: int = Field(default=6, ge=1)
    scouting_population_middle: int = Field(default=4, ge=1)
    maximum_population: int = Field(default=60, ge=1)
    population_cap_early: int = Field(default=12, ge=1)
    population_cap_middle: int = Field(default=4, ge=1)
    population_cap_late: int = Field(default=2, ge=1)
    population_middle_seconds: float = Field(default=900., gt=0)
    population_late_seconds: float = Field(default=1800., gt=0)
    food_budget_per_agent_second: float = Field(default=6.0, gt=0)
    birth_interval_seconds: float = Field(default=0.5, gt=0)
    parent_reserve: float = Field(default=100.0, ge=0)
    minimum_breeder_trait_score: float = Field(default=1.0, ge=0)
    gene_backup_age: float = Field(default=40.0, ge=0)
    gene_backup_interval_seconds: float = Field(default=35.0, gt=0)
    renewal_age: float = Field(default=55.0, gt=0)
    renewal_parent_reserve: float = Field(default=40.0, ge=0)
    emergency_parent_reserve: float = Field(default=20.0, ge=0)
    lineage_transfer_reserve: float = Field(default=5.0, ge=0)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    conservation: ConservationConfig = Field(default_factory=ConservationConfig)

    @model_validator(mode="after")
    def valid_limits(self):
        if not self.initial_energy <= self.ripe_energy <= self.maximum_energy:
            raise ValueError("Require initial_energy <= ripe_energy <= maximum_energy")
        if self.minimum_population > self.maximum_population:
            raise ValueError("minimum_population cannot exceed maximum_population")
        if not self.population_cap_early >= self.population_cap_middle >= self.population_cap_late:
            raise ValueError("Population caps must decrease or stay equal over time")
        if self.population_late_seconds <= self.population_middle_seconds:
            raise ValueError("population_late_seconds must be greater than population_middle_seconds")
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
        self.metabolic_tracker = MetabolismTracker()
        self.travel = PublicBiomeTravel()
        self.conservation = ConservationSchedule(self.config.conservation)
        self.next_plan = 0.
        self.next_birth = 0.
        self.next_gene_backup = 0.
        self.gene_backup_ids = set()
        self.discoveries = []
        self.population_target = self.config.minimum_population
        self.population_plan = {}
        self.population_supply = None
        self.population_arrivals_supply = None
        self.population_plan_time = None
        self.planned_birth_interval = self.config.birth_interval_seconds
        self.hints = {}
        self.breeding = {}
        self.active = False
        self.recovering_population = False
        self.movement_factors = {}
        self.orchard_watches = {}
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
        return self.metabolic_tracker.estimate(state, self.dt)

    def _hungry(self, state):
        # React before a long trip becomes fatal, including the public lower
        # bound on senescence. Energy alone understates an old agent's risk.
        return state["energy"] < max(self.config.emergency_energy,
                                      self.config.survival_reserve + 12. * self._metabolism(state))

    def _candidate(self, state, pose, track, now, walking_cost):
        offset = track.position - pose.position
        length = float(np.linalg.norm(offset))
        distance = max(0., length - max(0., 8. - pose.uncertainty))
        destination = pose.position + offset * distance / max(length, 1e-9)
        speed = max(.01, min(state["speed"], state["sprint_speed"]))
        factor = max(.05, self.movement_factors.get(state["biome"], 1.))
        requested = self.travel.requested_distance([pose.position, destination], factor)
        travel = requested / speed * self.dt
        energy = self.energy(track, now + travel)
        movement_cost = requested * walking_cost * 1.2
        travel_metabolism = travel * self._metabolism(state)
        if movement_cost + travel_metabolism >= state["energy"] - 2.:
            return -1., True, 0.
        urgent = (self._hungry(state) or state["energy"] - movement_cost
                  - travel * self._metabolism(state) < self.config.survival_reserve)
        # Young agents can turn a small meal into enough energy to fund the
        # next generation. Wait only if the entire wait preserves a survival
        # reserve and observation bounds its birth. Newly discovered fruit
        # of unknown age may already be ripe and about to rot.
        wait = max(0., (self.config.ripe_energy - energy) / self.config.growth_per_second)
        if (not track.birth_bracketed
                or state["age"] + travel + wait >= self.config.renewal_age
                or state["energy"] - movement_cost - (travel + wait) * self._metabolism(state)
                    < self.config.survival_reserve):
            wait = 0.
        energy = self.energy(track, now + travel + wait)
        energy = min(energy, max(0., state["max_energy"] - state["energy"]
                                 + movement_cost + travel * self._metabolism(state)))
        # Passive drain continues if the agent stays still. A reachable meal
        # that pays for its extra walking extends life even when its energy
        # does not cover all metabolism incurred during the journey.
        net = energy - movement_cost - wait * self._metabolism(state)
        if now + travel + wait > track.first_seen + self.config.lifetime_seconds - 1:
            return -1., urgent, wait
        urgency = 1. + 2. * max(0., 1. - state["energy"] / self.config.emergency_energy)
        utility = net * urgency / (3. + travel + wait)
        if state["energy"] > state["max_energy"] - energy:
            utility *= .75
        utility *= math.exp(-(now - track.last_seen) / self.config.max_unseen_seconds)
        return utility, urgent, wait

    def _retiring_agents(self, states):
        """Conserve food once public evidence confirms costly senescence.

        Only carriers without an allocated meal rest. Aging agents may
        harvest food younger agents leave unclaimed; existing stored energy
        can still fund a replacement through the ordinary breeding planner.
        """
        if sum(state["age"] < 55. and state["energy"] > 60. for state in states) < 2:
            return set()
        tracker = self.metabolic_tracker
        return {state["agent_id"] for state in states
                if state["agent_id"] in tracker.history
                and tracker.history[state["agent_id"]].aging
                and tracker.rates.get(state["agent_id"], 0.) > 4.}

    def _assign(self, states, poses, now, population, mechanics):
        """Share useful food, with territories as a gentle locality preference.

        Keep a viable ripe target until it is collected. Hungry agents choose
        first and can borrow a reservation; no colony-wide assignment solver.
        """
        choices, emergencies = {}, set()
        by_id = {state["agent_id"]: state for state in states}
        # Ownership is shared across agents; query the map once per fruit.
        # Repeating the fine-grid lookup for every agent dominates large colonies.
        owners = {track_id: self.coverage.owner(track.position) for track_id, track in self.tracks.items()}
        for state in states:
            agent_id = state["agent_id"]
            pose = poses[agent_id]
            if pose.uncertainty > self.config.max_pose_uncertainty:
                continue
            emergency = self._hungry(state)
            if emergency:
                emergencies.add(agent_id)
            options = {}
            for track_id, track in self.tracks.items():
                if self.blocked_until.get((agent_id, track_id), 0.) > now:
                    continue
                owner = owners[track_id]
                value, urgent, wait = self._candidate(state, pose, track, now, mechanics.walking_energy_per_unit)
                if value <= 0:
                    continue
                if not emergency and owner is not None and owner != agent_id:
                    value *= .8
                options[track_id] = value
            choices[agent_id] = options
        assigned, claimed = {}, set()
        # A senescent parent can burn a fruit in a few seconds. Give younger
        # carriers first access so keeping an old parent alive cannot erase
        # the next generation when only one meal remains.
        senescent = {i for i in choices if
                    (record := self.metabolic_tracker.history.get(i)) is not None and record.aging}
        # One reservation does not cover a young carrier's next meals. Keep
        # its nearby viable options available instead of letting a costly
        # old parent consume the second fruit during the first pickup.
        # Remote food remains available when no healthy carrier can use it.
        protected_food = {track_id for agent_id, options in choices.items()
            if agent_id not in senescent
            and by_id[agent_id]["max_energy"] - by_id[agent_id]["energy"] >= self.config.initial_energy
            for track_id in options
            if np.linalg.norm(self.tracks[track_id].position - poses[agent_id].position) <= 160.}
        for agent_id in senescent:
            choices[agent_id] = {track_id: value for track_id, value in choices[agent_id].items()
                                 if track_id not in protected_food}
        order = sorted(choices, key=lambda i: (i in senescent or by_id[i]["age"] >= 90,
                       i not in emergencies,
                       by_id[i]["energy"] / self._metabolism(by_id[i]) if i in emergencies else 0.,
                       -max(choices[i].values(), default=0), i))
        for agent_id in order:
            options = {t: v for t, v in choices[agent_id].items() if t not in claimed}
            previous = self.assignments.get(agent_id)
            if previous in options and options[previous] >= .8 * max(options.values()):
                target = previous
            elif options:
                target = min(options, key=lambda t: (-options[t], t))
            else:
                continue
            assigned[agent_id] = target
            claimed.add(target)
        self.assignments = assigned
        self.next_plan = now + self.config.replan_seconds

    @staticmethod
    def _expected_age_drain(age, horizon):
        """Integrate expected age drain over public onset bounds, 60--120s.

        No individual's private maximum age is read. The integral averages
        over the documented onset distribution and .1s simulation ticks.
        """
        def primitive(value):
            middle = min(120., max(60., value))
            result = ((middle ** 3 - 60. ** 3) / 3.
                      - 30. * (middle ** 2 - 60. ** 2)) / 600.
            return result + .05 * max(0., value ** 2 - 120. ** 2)
        return primitive(age + horizon) - primitive(age)

    def _breeding_food(self, states, poses, now):
        """Share fresh nearby fruit evidence once across competing parents."""
        food = {state["agent_id"]: 0. for state in states}
        fresh = [track for track in self.tracks.values() if now - track.last_seen <= 5.]
        if not states or not fresh:
            return food
        positions = np.asarray([poses[state["agent_id"]].position for state in states])
        distances = np.linalg.norm(positions[:, None] - np.asarray([t.position for t in fresh])[None], axis=2)
        weights = np.where(distances <= 160., np.exp(-distances / 80.), 0.)
        # Neither a newborn nor a remote uncertain pose is proof of access to
        # a global food cluster. Uncertain agents remain in the head count.
        for row, state in enumerate(states):
            if poses[state["agent_id"]].uncertainty > self.config.max_pose_uncertainty:
                weights[row] = 0.
        shares = weights / np.maximum(1., weights.sum(axis=0))
        amounts = shares @ np.asarray([self.energy(track, now) for track in fresh])
        return {state["agent_id"]: float(amounts[row]) for row, state in enumerate(states)}

    def _forecast_age_drain(self, state, horizon):
        """Integrate future aging conditioned on public energy observations."""
        age = state["age"]
        end = age + horizon
        rate = .01 / self.dt
        record = self.metabolic_tracker.history.get(state["agent_id"])
        if record is not None and record.aging:
            return rate * (age * horizon + .5 * horizon ** 2)
        lower = max(60., record.healthy_at_age if record is not None
                    and record.healthy_at_age is not None else 60.)
        # A clean normal-upkeep sample rules out onset before that age.
        # The remaining public onset interval is uniform up to age 120.
        if lower >= 120.:
            start = max(age, lower)
            return .5 * rate * max(0., end ** 2 - start ** 2)
        start, stop = max(age, lower), min(end, 120.)
        result = 0.
        if stop > start:
            result = ((stop ** 3 - start ** 3) / 3.
                      - lower * (stop ** 2 - start ** 2) / 2.) / (120. - lower)
        start = max(age, 120.)
        if end > start:
            result += .5 * (end ** 2 - start ** 2)
        return rate * max(0., result)

    def _breeders(self, states, poses, now, population, mechanics, base_threshold):
        cfg = self.config
        count = len(population.ratings)
        # A tiny colony observes little food simply because it sees little of
        # the world. Retain enough scouts before treating that missing evidence
        # as a reason to shrink further. Late tree production supports fewer.
        if now < cfg.population_middle_seconds:
            phase, spatial_floor, scheduled_cap = "early", cfg.scouting_population_early, cfg.population_cap_early
        elif now < cfg.population_late_seconds:
            phase, spatial_floor, scheduled_cap = "middle", cfg.scouting_population_middle, cfg.population_cap_middle
        else:
            phase, spatial_floor, scheduled_cap = "late", cfg.survival_population, cfg.population_cap_late
        target_cap = min(cfg.maximum_population, scheduled_cap)
        floor = min(cfg.minimum_population, target_cap,
                    max(cfg.survival_population, spatial_floor))
        stock = sum(self.energy(track, now) for track in self.tracks.values())
        arrivals = len(self.discoveries) * .75 * cfg.ripe_energy / 30.
        supply = arrivals + stock / 45.
        elapsed = 0. if self.population_plan_time is None else max(0., now - self.population_plan_time)
        if self.population_supply is None:
            self.population_supply = supply
        else:
            self.population_supply += (supply - self.population_supply) * (1. - math.exp(-elapsed / 45.))
        if self.population_arrivals_supply is None:
            self.population_arrivals_supply = arrivals
        else:
            self.population_arrivals_supply += (arrivals - self.population_arrivals_supply) * (1. - math.exp(-elapsed / 45.))
        self.population_plan_time = now
        self.population_target = max(floor, min(target_cap,
                                      int(self.population_supply / cfg.food_budget_per_agent_second)))
        local_food = self._breeding_food(states, poses, now)
        # Existing stock is already counted once below. Credit only half the
        # independently observed arrival rate, distributed where fresh local
        # food supports access and capped at ordinary forecast upkeep.
        local_total = sum(local_food.values())
        forecast_income = {agent_id: min(3., .5 * self.population_arrivals_supply
            * food / max(local_total, 1e-9)) for agent_id, food in local_food.items()}
        forecasts = {}
        visible_ids = {state["agent_id"] for state in states}
        for horizon in (30., 60.):
            forecasts[horizon] = sum(
                state["energy"] + local_food[state["agent_id"]]
                + forecast_income[state["agent_id"]] * horizon - 3. * horizon
                - self._forecast_age_drain(state, horizon) > cfg.survival_reserve / 2.
                for state in states)
            # Briefly disconnected newborns still count against birth demand.
            forecasts[horizon] += sum(agent_id not in visible_ids and age + horizon < 60.
                                      for agent_id, age in population.ages.items())
        young = sum(state["age"] < cfg.renewal_age and state["energy"] > cfg.survival_reserve
                    for state in states)
        young += sum(agent_id not in visible_ids and age < cfg.renewal_age
                     for agent_id, age in population.ages.items())
        desired_young = min(self.population_target, max(cfg.survival_population,
                                                       math.ceil(.6 * self.population_target)))
        growth = count < self.population_target
        replacement = (young < desired_young or forecasts[30.] < self.population_target
                       or forecasts[60.] < desired_young)
        critical = forecasts[30.] < min(floor, max(2, math.ceil(count / 2.)))
        critical = critical or (count <= cfg.survival_population and any(
            state["age"] >= .6 * cfg.renewal_age for state in states))
        barren_emergency = count <= cfg.survival_population or (young == 0 and any(
            state["age"] >= cfg.renewal_age for state in states))
        overlap_cap = min(cfg.maximum_population, self.population_target + max(1, math.ceil(.5 * self.population_target)))
        # Stored energy in a confirmed senescent parent disappears quickly.
        # Preserve enough successor slots before old headcount falls, rather
        # than waiting until every potential parent is too weak to reproduce.
        # Actual healthy children, not the pessimistic 30s energy forecast,
        # close these slots. Existing weak children still occupy capacity.
        senescent_ids = {state["agent_id"] for state in states
            if (record := self.metabolic_tracker.history.get(state["agent_id"])) is not None
            and record.aging and self._metabolism(state) > 4.}
        old_count = sum(age >= cfg.renewal_age for age in population.ages.values())
        successor_cap = min(cfg.maximum_population, max(overlap_cap, old_count + desired_young))
        successor_slots = max(0, desired_young - young)
        funded_successors = critical and successor_slots > 0 and bool(senescent_ids)
        ordinary_capacity = count < (overlap_cap if replacement else self.population_target)
        funded_capacity = funded_successors and count < successor_cap
        cohort_rescue = funded_capacity and young == 0
        can_expand = ordinary_capacity or funded_capacity
        # Several seconds between successful births create staggered cohorts.
        # Immediate extinction risk can use the faster configured birth slot.
        self.planned_birth_interval = max(cfg.birth_interval_seconds,
            1. if critical else 2.5 if growth else min(8., 30. / max(1, desired_young)))
        self.population_plan = dict(population=count, target=self.population_target, scouting_floor=floor,
            population_phase=phase, target_cap=target_cap,
            estimated_food_per_second=self.population_supply, viable_in_30_seconds=forecasts[30.],
            observed_arrival_energy_per_second=self.population_arrivals_supply,
            credited_forecast_food_per_second=sum(forecast_income.values()),
            viable_in_60_seconds=forecasts[60.], healthy_young=young, desired_young=desired_young,
            replacement_needed=replacement, critical=critical, overlap_cap=overlap_cap,
            successor_cap=successor_cap, missing_healthy_successors=successor_slots,
            funded_successor_transfer=funded_capacity,
            barren_birth_allowed=barren_emergency,
            last_cohort_rescue=cohort_rescue,
            birth_interval_seconds=self.planned_birth_interval, selected_parent=None,
            reason="capacity reached" if not can_expand else "no suitable parent")
        hints = {s["agent_id"]: ReproductionHint(base_threshold, False) for s in states}
        self.gene_backup_ids = set()
        if not can_expand or count >= cfg.maximum_population or now < self.next_birth - 1e-9:
            if now < self.next_birth - 1e-9:
                self.population_plan["reason"] = "spacing generations"
            return hints
        if not growth and not replacement:
            self.population_plan["reason"] = "population sustainable"
            return hints
        candidates = []
        for state in states:
            agent_id = state["agent_id"]
            if poses[agent_id].uncertainty > cfg.max_pose_uncertainty:
                continue
            funded_transfer = funded_capacity and agent_id in senescent_ids
            ordinary = population.reproduction_hint(agent_id, now, base_threshold)
            # A confirmed aging parent can burn another child's entire
            # energy cost during the ordinary cooldown. The global birth
            # interval and funded-successor bounds still limit transfers.
            if ordinary is not None and not ordinary.allowed and not funded_transfer:
                continue
            if not ordinary_capacity and not funded_transfer:
                continue
            food_ready = local_food[agent_id] >= cfg.initial_energy
            # Avoid releasing a 75-energy newborn into a barren neighbourhood
            # unless the current population or its forecast is already unsafe.
            if not food_ready and not barren_emergency and not funded_transfer:
                continue
            # A low long-range forecast is not permission to split every
            # young parent's travelling reserve. Spend the smaller reserve
            # primarily to replace aging carriers; young parents use it only
            # when the actual colony is close to extinction.
            renewal = funded_transfer or (replacement and (state["age"] >= cfg.renewal_age
                                       or (critical and count <= cfg.survival_population)))
            reserve = cfg.renewal_parent_reserve if renewal else cfg.parent_reserve
            if renewal and critical and count <= cfg.survival_population:
                reserve = min(reserve, cfg.emergency_parent_reserve)
            if funded_transfer:
                reserve = min(reserve, cfg.lineage_transfer_reserve)
            threshold = ordinary.energy_threshold if ordinary is not None else base_threshold
            if renewal:
                threshold = mechanics.spawn_energy_cost + reserve + 2.
            if state["energy"] <= max(threshold, mechanics.spawn_energy_cost + reserve + 2.):
                continue
            rating = population.ratings[agent_id]
            # Nutrition and replacing aging carriers come before gene ranking;
            # among similarly prepared parents, preserve the strongest traits.
            candidates.append((not food_ready, int(state["age"] < cfg.renewal_age) if renewal else 0,
                -min(3, int(local_food[agent_id] / 60.)), -rating.score, -state["energy"],
                agent_id, threshold, reserve, renewal, funded_transfer))
        if candidates:
            _, _, _, _, _, agent_id, threshold, reserve, renewal, funded_transfer = min(candidates)
            hints[agent_id] = ReproductionHint(threshold, True, reserve, preserve_lineage=renewal)
            self.population_plan.update(selected_parent=agent_id,
                reason="transfer senescent energy to successors" if funded_transfer else
                       "emergency lineage renewal" if renewal and critical else "replace aging cohort" if renewal else "restore scouting population",
                parent_local_food=local_food[agent_id])
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
        self.metabolic_tracker.walking_cost = mechanics.walking_energy_per_unit
        self.metabolic_tracker.sprinting_cost = mechanics.sprinting_energy_per_unit
        self.metabolic_tracker.spawn_cost = mechanics.spawn_energy_cost
        self.metabolic_tracker.sprint_min_energy_fraction = mechanics.sprint_min_energy_fraction
        metabolic_rates = self.metabolic_tracker.update(states, now, self.dt)
        poses = planner.estimator.poses
        if hasattr(planner.estimator, "config"):
            self.movement_factors = planner.estimator.config.biome_movement_factors
            travel_factors = {"forest": 1., "grassland": 1., **self.movement_factors}
            if travel_factors != self.travel.movement_factors:
                self.travel = PublicBiomeTravel(travel_factors)
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
            self.orchard_watches.clear()
        self._observe(eligible, poses, now)
        group = groups[group_id]
        biome_layer = getattr(getattr(planner, "biome_estimator", None), "layers", {}).get(group_id)
        self.travel.update(biome_layer)
        self.coverage.update(eligible, poses, groups[group_id], self.previous_views,
                             self._visible, population.ratings, now, self.config.max_pose_uncertainty,
                             biome_layer=biome_layer, fruits=self.tracks.values(), metabolic_rates=metabolic_rates)
        self.navigator.update(group, now)
        self.last_ages = {s["agent_id"]: s["age"] for s in states}
        self.blocked_until = {key: end for key, end in self.blocked_until.items() if end > now}
        living = {s["agent_id"] for s in eligible}
        self.navigator.prune(living)
        self.conservation.prune(living)
        self.orchard_watches = {i: watch for i, watch in self.orchard_watches.items() if i in living}
        self.assignments = {i: t for i, t in self.assignments.items() if i in living and t in self.tracks}
        if now >= self.next_plan - 1e-9:
            self._assign(eligible, poses, now, population, mechanics)
        retiring = self._retiring_agents(states)
        recent_trees = [tree for tree in group.trees if now - tree.last_seen <= 6.]
        if recent_trees:
            tree_positions = np.asarray([tree.position for tree in recent_trees])
            nearby_agents = cKDTree([poses[s["agent_id"]].position for s in eligible])
            tree_crowds = nearby_agents.query_ball_point(tree_positions, 65., return_length=True)
        for state in eligible:
            agent_id = state["agent_id"]
            if poses[agent_id].uncertainty > self.config.max_pose_uncertainty:
                continue
            track = self.tracks.get(self.assignments.get(agent_id))
            pose = poses[agent_id]
            waiting, look, scan_while_stationary = False, None, False
            if track is None:
                if agent_id in retiring:
                    self.navigator.release(agent_id)
                    scanning = self.conservation.scanning(agent_id, now, self.dt)
                    self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False,
                        look_direction=math.pi / 4 if scanning else None,
                        scan_while_stationary=scanning, allow_local_food=False)
                    self.tasks[agent_id] = dict(kind="retired; preserve food for young")
                    continue
                patrol = self.coverage.hint(agent_id, pose)
                home = self.coverage.homes.get(agent_id)
                relocating = getattr(self.coverage, "tasks", {}).get(agent_id) == "seek productive biome"
                # Recent, publicly seen trees are a better place to look for
                # the next meal than empty territory centers. Stay within
                # hearing range instead of repeatedly pacing around a tree.
                orchard = None
                travel_budget = CoverageCoordinator._travel_budget(state, self._metabolism(state))
                if recent_trees and state["biome"] != "river" and not relocating:
                    distances = np.linalg.norm(tree_positions - pose.position, axis=1)
                    costs = distances + 100. * (tree_crowds - (distances < 65.))
                    for index in np.argsort(costs):
                        candidate = recent_trees[index]
                        other_campers = tree_crowds[index] - int(distances[index] < 65.)
                        if (distances[index] <= travel_budget and other_campers < 1
                                and self.blocked_until.get((agent_id, ("orchard", tuple(candidate.position))), 0.) <= now):
                            orchard = candidate
                            break
                if orchard is not None:
                    destination = orchard.position.copy()
                    distance = float(np.linalg.norm(destination - pose.position))
                    if distance < max(20., state["hearing_radius"] * .6):
                        key = tuple(destination)
                        previous, since = self.orchard_watches.get(agent_id, (None, now))
                        if previous != key:
                            since = now
                        self.orchard_watches[agent_id] = (key, since)
                        if now - since >= 8.:
                            self.blocked_until[(agent_id, ("orchard", key))] = now + 20.
                            self.orchard_watches.pop(agent_id, None)
                            orchard = None
                        else:
                            self.navigator.release(agent_id)
                            # Fruit can grow beyond hearing range behind a
                            # stationary camper. Sweep once per six seconds,
                            # spending about one energy per complete turn.
                            scanning = self.conservation.scanning(agent_id, now, self.dt)
                            self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False,
                                look_direction=math.pi / 4 if scanning else None,
                                scan_while_stationary=scanning)
                            self.tasks[agent_id] = dict(kind="watch orchard", destination=destination.tolist())
                            continue
                    if orchard is not None and (state["energy"] < 180. or distance < 150.):
                        kind = "find food at orchard"
                    else:
                        orchard = None
                if (orchard is None and state["energy"] >= 300. and state["age"] < 50.
                        and state["biome"] != "river" and not relocating and not self._hungry(state)):
                    self.navigator.release(agent_id)
                    scanning = self.conservation.scanning(agent_id, now, self.dt)
                    self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False,
                        look_direction=math.pi / 4 if scanning else None,
                        scan_while_stationary=scanning)
                    self.tasks[agent_id] = dict(kind="rest with food reserve")
                    continue
                elif orchard is not None:
                    pass
                elif (patrol is None and home is not None and self.coverage.owner(pose.position) != agent_id
                        and self.blocked_until.get((agent_id, "home"), 0.) <= now
                        and np.linalg.norm(np.asarray(home) - pose.position) <= travel_budget):
                    destination = np.array(home)
                    kind = "return to territory"
                elif patrol is None:
                    self.navigator.release(agent_id)
                    # The old exploration policy is only a fallback until a
                    # territory map exists; afterwards idling is deliberate.
                    vector = (0., 0.) if self.coverage.homes else None
                    scanning = vector is not None and self.conservation.scanning(agent_id, now, self.dt)
                    self.hints[agent_id] = HarvestHint(vector, None, 0., False,
                        look_direction=math.pi / 4 if scanning else None,
                        scan_while_stationary=scanning)
                    self.tasks[agent_id] = dict(kind="idle" if vector else "local exploration")
                    continue
                else:
                    destination = pose.position + rotate(np.array(patrol.vector), pose.heading)
                    look = patrol.look_direction
                    scan_while_stationary = patrol.scan_while_stationary
                    kind = self.coverage.tasks.get(agent_id, "patrol territory")
                    if np.linalg.norm(patrol.vector) <= 1.:
                        scan_while_stationary = self.conservation.scanning(agent_id, now, self.dt)
                        look = math.pi / 4 if scan_while_stationary else None
                    elif look is not None:
                        # Keep facing the route, but gradually remove the
                        # optional side-to-side scanning while walking.
                        heading = math.atan2(patrol.vector[1], patrol.vector[0])
                        sweep = (look - heading + math.pi) % math.tau - math.pi
                        look = heading + (1. - self.conservation.fraction(now)) * sweep
            else:
                self.orchard_watches.pop(agent_id, None)
                _, _, wait = self._candidate(state, pose, track, now, mechanics.walking_energy_per_unit)
                offset = track.position - pose.position
                distance = float(np.linalg.norm(offset))
                waiting = wait > 0.
                if waiting:
                    offset *= max(0., distance - self.config.pickup_clearance) / max(distance, 1e-9)
                destination = pose.position + offset
                kind = "wait for ripening" if waiting and np.linalg.norm(offset) < 1 else "collect fruit"
            if (track is None and kind != "find food at orchard" and not relocating
                    and not self.conservation.should_scout(agent_id, now)):
                # Rest without abandoning the shared survey destination or
                # charging deliberate inactivity against navigation progress.
                self.navigator.pause(agent_id, now)
                scanning = self.conservation.scanning(agent_id, now, self.dt)
                self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False,
                    look_direction=math.pi / 4 if scanning else None,
                    scan_while_stationary=scanning, allow_local_food=False)
                self.tasks[agent_id] = dict(kind="rest; conserve energy", destination=destination.tolist())
                continue
            route = self.navigator.steer(agent_id, pose.position, destination, now,
                                         waiting=kind == "wait for ripening",
                                         arrival_radius=max(0., 8. - pose.uncertainty)
                                         if track is not None and not waiting else 0.)
            self.tasks[agent_id] = dict(kind=kind, destination=destination.tolist(),
                                       navigation=route.status,
                                       remaining=route.remaining if math.isfinite(route.remaining) else None)
            factor = max(.05, self.movement_factors.get(state["biome"], 1.))
            speed = max(.01, min(state["speed"], state["sprint_speed"]))
            planned = self.navigator.routes.get(agent_id)
            points = ([pose.position] + planned.points if planned is not None
                      and route.status not in ("arrived", "waiting") else [pose.position, destination])
            if track is not None and waiting:
                approach = max(0., distance - max(0., 8. - pose.uncertainty))
                points = points + [pose.position + (track.position - pose.position)
                                   * approach / max(distance, 1e-9)]
            requested = (math.inf if route.blocked else
                         self.travel.requested_distance(points, factor))
            route_cost = requested * (mechanics.walking_energy_per_unit * 1.2
                                      + self.dt / speed * self._metabolism(state))
            if track is not None:
                # A fruit just beyond a rock can require a long detour. The
                # straight-line assignment estimate is only a preliminary
                # screen; pay for the actual remaining route before claiming
                # its reward. Passive metabolism still matters for solvency.
                unaffordable = route_cost >= state["energy"] - 2.
            else:
                search_budget = .8 * max(0., state["energy"] - min(35., state["energy"] * .25))
                unaffordable = route_cost > search_budget
            if route.blocked or unaffordable:
                if track is not None:
                    self.blocked_until[(agent_id, track.track_id)] = now + 20.
                    self.assignments.pop(agent_id, None)
                elif kind == "return to territory":
                    self.blocked_until[(agent_id, "home")] = now + 20.
                elif kind == "find food at orchard":
                    self.blocked_until[(agent_id, ("orchard", tuple(destination)))] = now + 20.
                else:
                    self.coverage.reject_target(agent_id, now)
                self.navigator.release(agent_id)
                self.next_plan = now
                scanning = self.conservation.scanning(agent_id, now, self.dt)
                self.hints[agent_id] = HarvestHint((0., 0.), None, 0., False,
                    look_direction=math.pi / 4 if scanning else None,
                    scan_while_stationary=scanning)
                self.tasks[agent_id]["kind"] = ("route exceeds energy budget; choose another target"
                                                if unaffordable and not route.blocked else "blocked; choose another target")
                continue
            vector = np.zeros(2) if route.waypoint is None else rotate(route.waypoint - pose.position, -pose.heading)
            self.hints[agent_id] = HarvestHint(tuple(float(v) for v in vector),
                None if track is None else track.track_id, 0. if track is None else self.energy(track, now),
                waiting, survey=track is None, look_direction=look,
                scan_while_stationary=scan_while_stationary)
        self.breeding = self._breeders(eligible, poses, now, population, mechanics, base_threshold)
        # The colony's food budget and birth slot also cover newborns which
        # briefly have their own coordinate frame.
        for state in states:
            self.breeding.setdefault(state["agent_id"], ReproductionHint(base_threshold, False))
        # Central assignments already consider every shared observation and
        # remembered wall. A local greedy pickup must not undo reservations
        # or retry an unaffordable/blocked route through an unseen wall.
        self.hints = {agent_id: replace(hint, allow_local_food=False)
                      if hint.vector is not None else hint for agent_id, hint in self.hints.items()}
        return self.hints.copy(), self.breeding.copy()

    def remember_actions(self, actions, now):
        self.metabolic_tracker.remember_actions(actions)
        if self.active and any(action.spawn_agent for action in actions):
            self.next_birth = now + self.planned_birth_interval
            if any(a.spawn_agent and a.agent_id in self.gene_backup_ids for a in actions):
                self.next_gene_backup = now + 5.

    def snapshot(self):
        now = self.last_time or 0.
        return dict(enabled=self.config.enabled, active=self.active, population_target=self.population_target,
                    conservation=dict(fraction=self.conservation.fraction(now),
                        scouting_fraction=self.conservation.scouting_fraction(now),
                        scan_interval_seconds=self.conservation.scan_interval(now)),
                    reproduction_plan=dict(self.population_plan),
                    group_id=None if self.frame is None else self.frame[0],
                    gene_backup_ids=sorted(self.gene_backup_ids),
                    tasks=self.tasks, navigation=self.navigator.snapshot(),
                    coverage=self.coverage.snapshot(),
                    assignments=dict(self.assignments), fruits=[dict(track_id=t.track_id,
                    position=t.position.tolist(), first_seen=t.first_seen, last_seen=t.last_seen,
                    estimated_energy=self.energy(t, now), birth_bracketed=t.birth_bracketed,
                    ripe_in_seconds=max(0., (self.config.ripe_energy - self.energy(t, now)) / self.config.growth_per_second))
                    for t in sorted(self.tracks.values(), key=lambda t: t.track_id)])
