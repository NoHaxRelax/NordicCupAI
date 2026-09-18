"""Stable food territories and one persistent survey task per owner."""

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from src.utils.controllers.world_estimator import rotate
from src.utils.controllers.survey_gaps import SurveyGaps
from src.utils.controllers.territories import allocate, connected_parts, corridor_graph


class CoverageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=True, strict=True)
    cell_size: float = Field(default=100., gt=0)
    max_cells: int = Field(default=512, ge=4)
    observation_interval_seconds: float = Field(default=.5, gt=0)
    replan_seconds: float = Field(default=2., gt=0)
    territory_seconds: float = Field(default=30., gt=0)
    food_imbalance_fraction: float = Field(default=.35, gt=0, le=1)
    orchard_revisit_seconds: float = Field(default=20., gt=0)
    other_revisit_seconds: float = Field(default=60., gt=0)
    minimum_patch_width: float = Field(default=25., gt=0)
    gap_cell_size: float = Field(default=10., gt=0)
    max_gap_cells: int = Field(default=24000, ge=16)
    nearby_gap_distance: float = Field(default=220., gt=0)
    rejected_target_seconds: float = Field(default=20., gt=0)


# Relative production from public environment mechanics, used only as a weak
# prior. Labels and confidence come from the observation-based biome estimate.
BIOME_POTENTIAL = dict(forest=1., grassland=.5, swamp=.72, desert=.05, river=0.)
MOVEMENT_FACTOR = dict(swamp=.5, desert=.8, river=.3)


class CoverageCoordinator:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.frame = None
        self.points = np.empty((0, 2))
        self.seen = np.empty(0)
        self.orchards = np.empty(0, dtype=bool)
        self.surveyable = np.empty(0, dtype=bool)
        self.food_weights = np.empty(0)
        self.travel_efficiency = np.empty(0)
        self.biome_quality = np.empty(0)
        self.biome_confidence = np.empty(0)
        self.homes = {}
        self.owners = np.empty(0, dtype=int)
        self.targets = {}
        self.gap_targets = {}
        self.tasks = {}
        self.target_since = {}
        self.blocked = {}
        self.patrol_ids = set()
        self.fruit_sites = {}
        self.members = set()
        self.next_observation = self.next_plan = self.next_territory = 0.
        self.now = 0.
        self.gaps = None
        self.graph = []
        self.cell_labels = None
        self.grid_pitch = None

    def _grid(self, group):
        size = group.world_size
        if size is None and group.known_width is not None and group.known_height is not None:
            size = (group.known_width, group.known_height)
        if size is None:
            return False
        frame = (group.group_id, group.frame_revision, tuple(size))
        if self.frame == frame:
            return True
        self.reset()
        self.frame = frame
        self.gaps = SurveyGaps(size, self.config.gap_cell_size, self.config.max_gap_cells, self.config.minimum_patch_width)
        margin = min(40., min(size) / 4)
        extent = np.asarray(size) - 2 * margin
        spacing = self.config.cell_size
        while np.prod(np.ceil(extent / spacing)) > self.config.max_cells:
            spacing *= 1.2
        nx, ny = np.maximum(1, np.ceil(extent / spacing).astype(int))
        self.grid_pitch = extent / (nx, ny)
        xx, yy = np.meshgrid(margin + (np.arange(nx) + .5) * self.grid_pitch[0],
                             margin + (np.arange(ny) + .5) * self.grid_pitch[1])
        self.points = np.column_stack((xx.ravel(), yy.ravel()))
        self.seen = np.full(len(self.points), -1.)
        self.orchards = np.zeros(len(self.points), dtype=bool)
        self.surveyable = np.ones(len(self.points), dtype=bool)
        self.food_weights = np.ones(len(self.points))
        self.travel_efficiency = np.ones(len(self.points))
        self.biome_quality = np.full(len(self.points), .5)
        self.biome_confidence = np.zeros(len(self.points))
        self.owners = np.full(len(self.points), -1, dtype=int)
        return True

    def _fine_indices(self, points):
        indices = np.floor((np.asarray(points) - self.gaps.origin) / self.gaps.pitch).astype(int)
        return np.clip(indices, [0, 0], [self.gaps.nx - 1, self.gaps.ny - 1])

    def _food_weights(self, group, biome_layer):
        # A nonzero unknown-area budget keeps unexplored terrain represented.
        prior = np.full(len(self.points), .5)
        self.travel_efficiency[:] = 1.
        self.biome_confidence[:] = 0.
        if biome_layer is not None and biome_layer.get("labels"):
            labels = np.asarray(biome_layer["labels"], dtype=int)
            confidence = np.asarray(biome_layer["confidence"], dtype=float)
            bounds = np.asarray(biome_layer["bounds"])
            indices = np.floor((self.points - bounds[:2]) / (bounds[2:] - bounds[:2])
                               * [labels.shape[1], labels.shape[0]]).astype(int)
            valid = ((indices[:, 0] >= 0) & (indices[:, 0] < labels.shape[1])
                     & (indices[:, 1] >= 0) & (indices[:, 1] < labels.shape[0]))
            palette = biome_layer["palette"]
            for cell in np.flatnonzero(valid):
                x, y = indices[cell]
                label = int(labels[y, x])
                if 0 <= label < len(palette):
                    certainty = float(np.clip(confidence[y, x], 0, 1))
                    self.biome_confidence[cell] = certainty
                    prior[cell] = .5 * (1 - certainty) + BIOME_POTENTIAL.get(palette[label], .5) * certainty
                    self.travel_efficiency[cell] = (1 - certainty) + certainty * MOVEMENT_FACTOR.get(palette[label], 1.)
        self.biome_quality = prior * self.travel_efficiency
        weights = .1 + .3 * prior
        recent_trees = [tree for tree in group.trees if self.now - tree.last_seen < 45.]
        tree_positions = np.asarray([tree.position for tree in recent_trees])
        self.orchards[:] = False
        if len(tree_positions):
            distances, _ = cKDTree(tree_positions).query(self.points)
            self.orchards = distances < 150.
        # Spread evidence across adjacent cells so one prolific tree can be
        # shared between neighboring territories without creating a giant cell.
        for tree in recent_trees:
            position = tree.position
            kernel = np.exp(-np.sum((self.points - position) ** 2, axis=1) / (2 * 70. ** 2))
            kernel[~self.surveyable] = 0
            if kernel.sum() > 0:
                freshness = max(0., 1. - (self.now - tree.last_seen) / 45.)
                weights += 4. * freshness * kernel / kernel.sum()
        fruit_weight = np.zeros(len(self.points))
        for position, observed in self.fruit_sites.values():
            distance = np.linalg.norm(self.points - position, axis=1)
            nearest = int(np.argmin(np.where(self.surveyable, distance, np.inf)))
            fruit_weight[nearest] += .5 * max(0., 1 - (self.now - observed) / 30.)
        self.orchards |= fruit_weight > .25
        weights += np.minimum(fruit_weight, .5)
        self.food_weights = np.where(self.surveyable, weights, 0.)

    def _clear_target(self, agent_id):
        self.targets.pop(agent_id, None)
        self.gap_targets.pop(agent_id, None)
        self.target_since.pop(agent_id, None)
        self.tasks.pop(agent_id, None)

    def destination(self, agent_id):
        if agent_id in self.gap_targets:
            return self.gap_targets[agent_id].copy()
        target = self.targets.get(agent_id)
        return self.points[target].copy() if target is not None else None

    def _is_blocked(self, agent_id, position):
        return any(owner == agent_id and np.linalg.norm(np.asarray(point) - position) < self.config.minimum_patch_width
                   for (owner, point), expiry in self.blocked.items() if expiry > self.now)

    def reject_target(self, agent_id, now):
        destination = self.destination(agent_id)
        if destination is not None:
            self.blocked[(agent_id, tuple(destination))] = now + self.config.rejected_target_seconds
        self._clear_target(agent_id)
        self.patrol_ids.discard(agent_id)
        self.next_plan = min(self.next_plan, now)

    def owner(self, position):
        if not self.config.enabled or self.cell_labels is None:
            return None
        point = np.asarray(position)
        if np.any(point < 0) or np.any(point >= np.asarray(self.frame[2])):
            return None
        x, y = self._fine_indices([point])[0]
        cell = int(self.cell_labels[y, x])
        if cell < 0 or not self.surveyable[cell]:
            return None
        owner = int(self.owners[cell])
        return owner if owner >= 0 else None

    @staticmethod
    def _travel_budget(state, metabolism=None):
        """Conservative walking range while retaining energy for another meal.

        Public mechanics charge .05 per requested walking unit. Slow biomes
        increase the number of commands, and ageing can add drain per .1s
        simulation step. These are planning bounds, not hidden agent state.
        """
        factor = MOVEMENT_FACTOR.get(state["biome"], 1.)
        if metabolism is None:
            metabolism = 2. + (state["age"] * .1 if state["age"] >= 60. else 0.)
        speed = max(.01, min(state["speed"], state["sprint_speed"]))
        cost_per_pixel = (.06 + .1 * metabolism / speed) / factor
        reserve = min(35., state["energy"] * .25)
        return max(0., .8 * (state["energy"] - reserve) / cost_per_pixel)

    def _patrol_score(self, cell, distance, urgency, agent_id):
        # Unknown cells retain their nonzero prior. Observed productive areas
        # receive more attention, and a long walk must justify its energy cost.
        locality = 1. if self.owners[cell] == agent_id else .8
        return (locality * (.1 + self.food_weights[cell]) * self.travel_efficiency[cell]
                * urgency / (75. + distance))

    def update(self, states, poses, group, views, visible, ratings, now, max_uncertainty,
               biome_layer=None, fruits=(), metabolic_rates=None):
        if not self.config.enabled or not group.anchored or not self._grid(group):
            self.patrol_ids = set()
            return
        self.now = now
        metabolic_rates = metabolic_rates or {}
        if now >= self.next_observation - 1e-9:
            self.gaps.observe(views)
            for view in views.values():
                indices = np.flatnonzero(np.linalg.norm(self.points - view[1], axis=1) <= max(view[3], view[4]))
                for index in indices:
                    if visible(self.points[index], view, 4.):
                        self.seen[index] = now
            for key, (position, observed) in list(self.fruit_sites.items()):
                if observed < now and any(visible(position, view, 4.)
                        and all(np.linalg.norm(position - fruit_position) > 30. for fruit_position in view[-1])
                        for view in views.values()):
                    self.fruit_sites[key] = (position, min(observed, now - 25.))
            self.next_observation = now + self.config.observation_interval_seconds
        for fruit in fruits:
            position = np.asarray(fruit.position)
            key = tuple(np.round(position / 20).astype(int))
            observed = float(getattr(fruit, "last_seen", now))
            if now - observed > 1.:
                continue
            if key not in self.fruit_sites or observed > self.fruit_sites[key][1]:
                # A retained track is not a fresh sighting. Only actual new
                # evidence may renew a site's productivity estimate.
                self.fruit_sites[key] = (position.copy(), observed)
        self.fruit_sites = {key: value for key, value in self.fruit_sites.items() if now - value[1] < 30}
        eligible = {s["agent_id"] for s in states
                    if s["agent_id"] in poses and poses[s["agent_id"]].uncertainty <= max_uncertainty}
        by_id = {state["agent_id"]: state for state in states}
        for agent_id in set(self.targets) | set(self.gap_targets):
            if (agent_id in eligible and np.linalg.norm(self.destination(agent_id) - poses[agent_id].position)
                    > self._travel_budget(by_id[agent_id], metabolic_rates.get(agent_id))):
                self._clear_target(agent_id)
                self.next_plan = min(self.next_plan, now)
        membership_changed = eligible != self.members
        if now < self.next_plan - 1e-9 and not membership_changed:
            return
        self.blocked = {key: expiry for key, expiry in self.blocked.items() if expiry > now}
        self.gaps.find(group, [poses[i] for i in sorted(eligible)])
        cells = self._fine_indices(self.points)
        previous_valid = self.surveyable.copy()
        self.surveyable = self.gaps.reachable[cells[:, 1], cells[:, 0]]
        refresh = (membership_changed or now >= self.next_territory or self.cell_labels is None
                   or not np.array_equal(previous_valid, self.surveyable))
        self._food_weights(group, biome_layer)
        if refresh:
            self.graph, self.cell_labels = corridor_graph(self.gaps, self.points, self.surveyable)
            agent_cells = {}
            for agent_id in eligible:
                x, y = self._fine_indices([poses[agent_id].position])[0]
                agent_cells[agent_id] = int(self.cell_labels[y, x])
            self.owners, self.homes = allocate(self.points, self.food_weights, self.graph, self.surveyable,
                                               eligible, poses, self.owners, self.config.food_imbalance_fraction,
                                               agent_cells)
            self.members = eligible
            self.next_territory = now + self.config.territory_seconds
        for agent_id in set(self.targets) | set(self.gap_targets):
            destination = self.destination(agent_id)
            if agent_id not in self.homes or (self.owner(destination) != agent_id
                                             and by_id[agent_id]["energy"] >= 180.):
                self._clear_target(agent_id)

        # Tiny unsensed raster holes are not useful destinations, even if a
        # coarse survey sample happens to lie inside one.
        unknown = ~self.gaps.seen & self.gaps.reachable
        clearance = distance_transform_edt(np.pad(unknown, 1), sampling=self.gaps.pitch[::-1])[1:-1, 1:-1]
        clearance -= max(self.gaps.pitch) / 2
        worthwhile = clearance[cells[:, 1], cells[:, 0]] >= self.config.minimum_patch_width / 2
        periods = np.where(self.orchards, self.config.orchard_revisit_seconds, self.config.other_revisit_seconds)
        components = np.full(len(self.points), -1, dtype=int)
        for number, part in enumerate(connected_parts(np.flatnonzero(self.surveyable), self.graph)):
            components[list(part)] = number
        for agent_id in sorted(self.homes):
            state = by_id[agent_id]
            budget = self._travel_budget(state, metabolic_rates.get(agent_id))
            hungry = state["energy"] < 180.
            old = self.destination(agent_id)
            if old is not None:
                if self.tasks.get(agent_id) == "seek productive biome":
                    completed = (state["biome"] != "river"
                                 or np.linalg.norm(old - poses[agent_id].position) < 15.)
                elif agent_id in self.gap_targets:
                    x, y = self._fine_indices([old])[0]
                    completed = self.gaps.seen[y, x] or clearance[y, x] < self.config.minimum_patch_width / 2
                else:
                    completed = self.seen[self.targets[agent_id]] >= self.target_since[agent_id]
                if not completed and not self._is_blocked(agent_id, old):
                    continue
                self._clear_target(agent_id)
            position = poses[agent_id].position
            x, y = self._fine_indices([position])[0]
            current_cell = int(self.cell_labels[y, x])
            component = components[current_cell] if current_cell >= 0 else -1
            if state["biome"] == "river":
                # Rivers produce no food. Swamps can support productive
                # orchards despite slow travel, so their movement penalty
                # belongs in route budgets and soft patrol preferences.
                current_quality = BIOME_POTENTIAL[state["biome"]] * MOVEMENT_FACTOR[state["biome"]]
                distances = np.linalg.norm(self.points - position, axis=1)
                better = np.flatnonzero(self.surveyable & (components == component)
                    & (self.biome_confidence >= .6)
                    & (self.biome_quality > max(.2, current_quality * 1.25))
                    & (distances <= budget) & (distances >= 15.))
                better = [cell for cell in better if not self._is_blocked(agent_id, self.points[cell])]
                if better:
                    cell = min(better, key=lambda c: distances[c] / self.biome_quality[c])
                    self.gap_targets[agent_id] = self.points[cell].copy()
                    self.tasks[agent_id] = "seek productive biome"
                    self.target_since[agent_id] = now
                    continue
            candidates = []
            for patch in self.gaps.patches:
                destination = np.asarray(patch["position"])
                distance = float(np.linalg.norm(destination - position))
                owner = self.owner(destination)
                if ((owner == agent_id or hungry) and distance <= min(budget, self.config.nearby_gap_distance)
                        and not self._is_blocked(agent_id, destination)):
                    x, y = self._fine_indices([destination])[0]
                    cell = int(self.cell_labels[y, x])
                    if cell >= 0 and components[cell] == component:
                        candidates.append((self._patrol_score(cell, distance, 3.5, agent_id), -1, destination, "scout"))
            available = np.flatnonzero(self.surveyable & (components == component)
                                        & ((self.owners == agent_id) | hungry))
            fallback = []
            partial = []
            for cell in available:
                unseen = self.seen[cell] < 0
                age = now - self.seen[cell]
                destination = self.points[cell]
                distance = float(np.linalg.norm(destination - position))
                if self._is_blocked(agent_id, destination):
                    continue
                if distance > budget:
                    if hungry and budget >= 12.:
                        # Coarse centers may all be out of range in a slow
                        # biome. Take a short step toward a promising area
                        # instead of waiting in an empty patch until death.
                        step = min(60., budget * .8)
                        waypoint = position + (destination - position) * step / distance
                        wx, wy = self._fine_indices([waypoint])[0]
                        waypoint_cell = int(self.cell_labels[wy, wx])
                        if (self.gaps.reachable[wy, wx] and waypoint_cell >= 0
                                and components[waypoint_cell] == component
                                and not self._is_blocked(agent_id, waypoint)):
                            urgency = 3. if unseen else min(4., max(1., age / periods[cell]))
                            partial.append((self._patrol_score(cell, distance, urgency, agent_id),
                                            -1, waypoint, "short food search"))
                    continue
                # Even a recently surveyed productive area is preferable to
                # sending a hungry newborn across an unaffordable territory.
                if hungry and distance >= self.config.minimum_patch_width:
                    fallback.append((self._patrol_score(cell, distance, 1., agent_id), int(cell), destination, "forage"))
                if (unseen and not worthwhile[cell]) or (not unseen and age < periods[cell]):
                    continue
                urgency = 3. if unseen else min(4., age / periods[cell])
                candidates.append((self._patrol_score(cell, distance, urgency, agent_id), int(cell), destination,
                                   "scout" if unseen else "orchard" if self.orchards[cell] else "patrol"))
            if not candidates and hungry:
                candidates = fallback or partial
            if candidates:
                _, cell, destination, task = max(candidates, key=lambda item: (item[0], item[1]))
                if cell < 0:
                    self.gap_targets[agent_id] = destination.copy()
                else:
                    self.targets[agent_id] = cell
                self.tasks[agent_id] = task
                self.target_since[agent_id] = now
            elif hungry:
                # Explicitly stop an unaffordable trip; the caller can still
                # assign observed food, and the next plan tries again soon.
                self.gap_targets[agent_id] = position.copy()
                self.tasks[agent_id] = "conserve energy"
                self.target_since[agent_id] = now
        self.patrol_ids = set(self.targets) | set(self.gap_targets)
        self.next_plan = now + self.config.replan_seconds

    def survey_fraction(self):
        return float(np.mean(self.seen[self.surveyable] >= 0)) if self.surveyable.any() else 0.

    def hint(self, agent_id, pose):
        from src.utils.controllers.policy_inputs import HarvestHint
        destination = self.destination(agent_id)
        if destination is None:
            return None
        vector = rotate(destination - pose.position, -pose.heading)
        look = math.atan2(vector[1], vector[0]) + .7 * math.sin(self.now * 2 + agent_id * 2.4)
        conserving = self.tasks.get(agent_id) == "conserve energy"
        scanning = conserving and (self.now + agent_id * .37) % 6. < .8
        return HarvestHint(tuple(float(v) for v in vector), None, 0., False, survey=True,
                           look_direction=math.pi / 4 if scanning else None if conserving else look,
                           scan_while_stationary=scanning)

    def snapshot(self):
        if not len(self.points):
            return dict(enabled=self.config.enabled, survey_percent=None, recent_percent=None, homes={}, targets={})
        return dict(enabled=self.config.enabled, survey_percent=100 * self.survey_fraction(),
                    recent_percent=100 * float(np.sum(self.surveyable & (self.seen >= 0)
                        & (self.now - self.seen <= self.config.orchard_revisit_seconds))) / max(1, int(self.surveyable.sum())),
                    metric="observed survey points; not a guarantee of complete area coverage",
                    grid_pitch=self.grid_pitch.tolist(),
                    homes={i: p.tolist() for i, p in self.homes.items()},
                    targets={i: self.points[t].tolist() for i, t in self.targets.items()},
                    gap_targets={i: p.tolist() for i, p in self.gap_targets.items()},
                    tasks=dict(self.tasks),
                    food_by_owner={i: float(self.food_weights[self.owners == i].sum()) for i in self.homes},
                    unknown_patches=self.gaps.patches,
                    patrol_ids=sorted(self.patrol_ids),
                    points=[dict(position=p.tolist(), last_seen=float(t), orchard=bool(o),
                                 owner=int(owner) if owner >= 0 else None, food_weight=float(weight))
                            for p, t, o, owner, weight in zip(self.points, self.seen, self.orchards,
                                                            self.owners, self.food_weights)])
