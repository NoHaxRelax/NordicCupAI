"""Stable food territories and one persistent survey task per owner."""

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from models.exploration.world_estimator import rotate
from models.exploration.survey_gaps import SurveyGaps
from models.exploration.territories import allocate, corridor_graph


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
        self.owners = np.full(len(self.points), -1, dtype=int)
        return True

    def _fine_indices(self, points):
        indices = np.floor((np.asarray(points) - self.gaps.origin) / self.gaps.pitch).astype(int)
        return np.clip(indices, [0, 0], [self.gaps.nx - 1, self.gaps.ny - 1])

    def _food_weights(self, group, biome_layer):
        # A nonzero unknown-area budget keeps unexplored terrain represented.
        prior = np.full(len(self.points), .5)
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
                    prior[cell] = .5 * (1 - certainty) + BIOME_POTENTIAL.get(palette[label], .5) * certainty
        weights = .1 + .3 * prior
        tree_positions = np.asarray([tree.position for tree in group.trees])
        self.orchards[:] = False
        if len(tree_positions):
            distances, _ = cKDTree(tree_positions).query(self.points)
            self.orchards = distances < 150.
        # Spread evidence across adjacent cells so one prolific tree can be
        # shared between neighboring territories without creating a giant cell.
        for position in tree_positions:
            kernel = np.exp(-np.sum((self.points - position) ** 2, axis=1) / (2 * 70. ** 2))
            kernel[~self.surveyable] = 0
            if kernel.sum() > 0:
                weights += 4. * kernel / kernel.sum()
        fruit_weight = np.zeros(len(self.points))
        for position, observed in self.fruit_sites.values():
            distance = np.linalg.norm(self.points - position, axis=1)
            nearest = int(np.argmin(np.where(self.surveyable, distance, np.inf)))
            fruit_weight[nearest] += .5 * max(0., 1 - (self.now - observed) / 120.)
        self.orchards |= fruit_weight > .25
        weights += np.minimum(fruit_weight, 4.)
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

    def update(self, states, poses, group, views, visible, ratings, now, max_uncertainty,
               biome_layer=None, fruits=()):
        if not self.config.enabled or not group.anchored or not self._grid(group):
            self.patrol_ids = set()
            return
        self.now = now
        if now >= self.next_observation - 1e-9:
            self.gaps.observe(views)
            for view in views.values():
                indices = np.flatnonzero(np.linalg.norm(self.points - view[1], axis=1) <= max(view[3], view[4]))
                for index in indices:
                    if visible(self.points[index], view, 4.):
                        self.seen[index] = now
            self.next_observation = now + self.config.observation_interval_seconds
        for fruit in fruits:
            position = np.asarray(fruit.position)
            key = tuple(np.round(position / 20).astype(int))
            self.fruit_sites[key] = (position.copy(), now)
        self.fruit_sites = {key: value for key, value in self.fruit_sites.items() if now - value[1] < 120}
        eligible = {s["agent_id"] for s in states
                    if s["agent_id"] in poses and poses[s["agent_id"]].uncertainty <= max_uncertainty}
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
        if refresh:
            self.graph, self.cell_labels = corridor_graph(self.gaps, self.points, self.surveyable)
            self._food_weights(group, biome_layer)
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
            if agent_id not in self.homes or self.owner(destination) != agent_id:
                self._clear_target(agent_id)

        # Tiny unsensed raster holes are not useful destinations, even if a
        # coarse survey sample happens to lie inside one.
        unknown = ~self.gaps.seen & self.gaps.reachable
        clearance = distance_transform_edt(np.pad(unknown, 1), sampling=self.gaps.pitch[::-1])[1:-1, 1:-1]
        clearance -= max(self.gaps.pitch) / 2
        worthwhile = clearance[cells[:, 1], cells[:, 0]] >= self.config.minimum_patch_width / 2
        periods = np.where(self.orchards, self.config.orchard_revisit_seconds, self.config.other_revisit_seconds)
        for agent_id in sorted(self.homes):
            old = self.destination(agent_id)
            if old is not None:
                if agent_id in self.gap_targets:
                    x, y = self._fine_indices([old])[0]
                    completed = self.gaps.seen[y, x] or clearance[y, x] < self.config.minimum_patch_width / 2
                else:
                    completed = self.seen[self.targets[agent_id]] >= self.target_since[agent_id]
                if not completed and not self._is_blocked(agent_id, old):
                    continue
                self._clear_target(agent_id)
            position = poses[agent_id].position
            candidates = []
            for patch in self.gaps.patches:
                destination = np.asarray(patch["position"])
                distance = float(np.linalg.norm(destination - position))
                if (self.owner(destination) == agent_id and distance <= self.config.nearby_gap_distance
                        and not self._is_blocked(agent_id, destination)):
                    candidates.append((4. - distance / 500., -1, destination, "scout"))
            owned = np.flatnonzero((self.owners == agent_id) & self.surveyable)
            for cell in owned:
                unseen = self.seen[cell] < 0
                age = now - self.seen[cell]
                if (unseen and not worthwhile[cell]) or (not unseen and age < periods[cell]):
                    continue
                destination = self.points[cell]
                if self._is_blocked(agent_id, destination):
                    continue
                urgency = 3. if unseen else min(4., age / periods[cell])
                distance = float(np.linalg.norm(destination - position))
                candidates.append((urgency - distance / 500., int(cell), destination,
                                   "scout" if unseen else "orchard" if self.orchards[cell] else "patrol"))
            if candidates:
                _, cell, destination, task = max(candidates, key=lambda item: (item[0], item[1]))
                if cell < 0:
                    self.gap_targets[agent_id] = destination.copy()
                else:
                    self.targets[agent_id] = cell
                self.tasks[agent_id] = task
                self.target_since[agent_id] = now
        self.patrol_ids = set(self.targets) | set(self.gap_targets)
        self.next_plan = now + self.config.replan_seconds

    def survey_fraction(self):
        return float(np.mean(self.seen[self.surveyable] >= 0)) if self.surveyable.any() else 0.

    def hint(self, agent_id, pose):
        from models.exploration.policy_inputs import HarvestHint
        destination = self.destination(agent_id)
        if destination is None:
            return None
        vector = rotate(destination - pose.position, -pose.heading)
        look = math.atan2(vector[1], vector[0]) + .7 * math.sin(self.now * 2 + agent_id * 2.4)
        return HarvestHint(tuple(float(v) for v in vector), None, 0., False, survey=True, look_direction=look)

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
