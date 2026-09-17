"""Robust bounded-uncertainty safety variant of v21 replaceable-site guiding.

Runtime inputs remain the detached static map, ordinary native observation DTOs,
and public simulation time. Predator energy, rest state, identity, and hidden
coordinates are neither requested nor inferred as facts.
"""
from __future__ import annotations

import math

from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import (
    add, mul, wrap,
)
from simple_chase.policy_v21_replaceable import SimpleChase as V21Replaceable


class SimpleChase(V21Replaceable):
    """V21 with stationarity-safe control and clustered motion uncertainty."""

    MAX_PREDATOR_MODELS = 6
    PREDATOR_SPEEDS = (0.0, 11.0, 15.0)
    ESCAPE_DEPTH = 6
    ESCAPE_BEAM = 12
    CONTACT_MARGIN = 19.0
    HEARING_TARGET = 55.0

    @property
    def stationary_ticks(self):
        # Native DTOs contain no energy/rest bit. A motionless observation may
        # be an awake predator temporarily blocked by a wall, so inherited
        # safety gates must always retain the active-predator path.
        return 0

    @stationary_ticks.setter
    def stationary_ticks(self, value):
        # The inherited increment reads the safety-facing getter above, so this
        # is deliberately only a last-assignment diagnostic, not a duration.
        self.stationary_observed_last_tick = bool(value)

    @staticmethod
    def _cluster_models(models, guide_point, goal, limit=6):
        """Keep bounded geometric extrema; safety was checked before pruning."""
        unique = {}
        for point, heading in models:
            key = (round(point[0], 5), round(point[1], 5), round(heading, 5))
            unique[key] = (point, heading)
        rows = list(unique.values())
        if len(rows) <= limit:
            return rows
        gx, gy = guide_point
        ux, uy = goal[0] - gx, goal[1] - gy
        length = max(math.hypot(ux, uy), 1e-12)
        ux, uy = ux / length, uy / length
        cross = (-uy, ux)

        def distance(row):
            return math.dist(row[0], guide_point)

        def along(row):
            delta = (row[0][0] - gx, row[0][1] - gy)
            return delta[0] * ux + delta[1] * uy

        def lateral(row):
            delta = (row[0][0] - gx, row[0][1] - gy)
            return delta[0] * cross[0] + delta[1] * cross[1]

        selected = []
        for key in (distance, along, lateral):
            for row in (min(rows, key=key), max(rows, key=key)):
                if row not in selected:
                    selected.append(row)
        if len(selected) < limit:
            for row in sorted(rows, key=distance):
                if row not in selected:
                    selected.append(row)
                if len(selected) == limit:
                    break
        return selected[:limit]

    def _advance_uncertain_model(self, point, heading, target, speed):
        # Native resting predators skip Predator.step entirely: no movement and
        # no turn. Calling _model_predator(..., speed=0) would incorrectly turn.
        if speed == 0.0:
            return point, heading
        return self._model_predator(point, heading, target, speed)

    def _branched_models(self, models, target):
        return [self._advance_uncertain_model(point, heading, target, speed)
                for point, heading in models
                for speed in self.PREDATOR_SPEEDS]

    def _escape_plan(self, state, pose, desired):
        observation = getattr(self, "_mpc_observation", None)
        if observation is None or self.last_predator_point is None:
            return None
        predator = self.last_predator_point
        heading = wrap(math.atan2(pose.p[1] - predator[1],
                                  pose.p[0] - predator[0])
                       - observation.get("rel_dir", 0.0))
        goal = desired if desired is not None else tuple(self.site["far"])

        # The DTO point precedes one native predator movement. Branch that
        # hidden movement as rest/stationary, energy-capped walk, or sprint.
        hidden = [self._advance_uncertain_model(predator, heading, pose.p, speed)
                  for speed in self.PREDATOR_SPEEDS]
        models = self._cluster_models(hidden, pose.p, goal,
                                     self.MAX_PREDATOR_MODELS)
        initial_goal_distance = math.dist(pose.p, goal)
        beam = [(0.0, pose.p, models, math.inf, None)]

        for _depth in range(self.ESCAPE_DEPTH):
            expanded = []
            for _, position, current_models, minimum, first in beam:
                step = state["sprint_speed"] * self._terrain_modifier(position)
                for index in range(17):
                    candidate = (position if index == 16 else
                                 add(position, (step * math.cos(math.tau * index / 16),
                                                step * math.sin(math.tau * index / 16))))
                    if not self._clear(position, candidate, 5.01):
                        continue
                    # Preserve every 0/11/15 child for the immediate safety
                    # decision; clustering happens only after the candidate is
                    # rejected or accepted.
                    all_next = self._branched_models(current_models, candidate)
                    separations = [math.dist(candidate, point)
                                   for point, _heading in all_next]
                    nearest = min(separations)
                    if nearest < self.CONTACT_MARGIN:
                        continue
                    farthest = max(separations)
                    low = min(minimum, nearest)
                    next_models = self._cluster_models(
                        all_next, candidate, goal, self.MAX_PREDATOR_MODELS)
                    progress = initial_goal_distance - math.dist(candidate, goal)
                    hearing_penalty = 2.0 * max(0.0, farthest - self.HEARING_TARGET)
                    score = (0.5 * progress + min(low, 45.0)
                             + (15.0 if self._free(candidate, 20.0) else 0.0)
                             - hearing_penalty)
                    expanded.append((score, candidate, next_models, low,
                                     candidate if first is None else first))
            if not expanded:
                break
            expanded.sort(key=lambda row: row[0], reverse=True)
            beam = expanded[:self.ESCAPE_BEAM]
        return max(beam, key=lambda row: row[0])[4]
