"""Recover an encounter with at most three sequential bait-born scouts.

The wrapper has one active guide at a time.  If that guide disappears from the
ordinary DTO stream before release, the stationary bait requests one native
child.  A new child performs a static-map sweep until its own DTO observes a
predator, then the existing v24 guide logic handles approach, delivery, and
release.  No pose or predator observation is shared between guide instances.
"""
from __future__ import annotations

import math

from reacquisition_sol.policy_global_sweep_gaze import GlobalSweepGaze
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import wrap
from integrated_guide.spatial_geometry import SpatialGeometry


class ColonyScout(SpatialGeometry, GlobalSweepGaze):
    """GlobalSweepGaze with projections capped to the child's native traits."""

    def act(self, observations, sim_time):
        if observations:
            state = observations[0]
            self._native_speed = float(state["speed"])
            self._native_sprint_speed = float(state["sprint_speed"])
        actions = super().act(observations, sim_time)
        # A bait-born scout may never have seen the lost predator, whereas the
        # inherited reacquisition rule starts only after an encounter. Give
        # that initial state the same deterministic static-map lawnmower path.
        if observations and self.last_predator_point is None:
            state = observations[0]
            aid = state["agent_id"]
            pose = self.poses.get(aid)
            if pose is not None:
                points = self._search_points()
                if self._sweep_index is None:
                    self._sweep_index = min(range(len(points)),
                                            key=lambda i: math.dist(pose.p, points[i]))
                target = points[self._sweep_index]
                if math.dist(pose.p, target) < 18:
                    self._sweep_index = (self._sweep_index + 1) % len(points)
                    target = points[self._sweep_index]
                route = self._path(pose.p, target, 5.01)
                waypoint = route[0] if route else pose.p
                action = self._move(aid, state, pose, waypoint,
                                    "initial_static_sweep_acquisition",
                                    turn=.3, sprint=True)
                old_turn = action["turn_angle"]
                action["turn_angle"] = action["move_direction"]
                pose.theta = wrap(pose.theta + action["turn_angle"] - old_turn)
                self.decisions[aid]["forward_sweep_gaze"] = True
                actions = [action]
        return actions

    def _project_endpoint(self, state, pose, target, sprint=False):
        bounded = dict(state)
        bounded["speed"] = min(float(state["speed"]), self._native_speed)
        bounded["sprint_speed"] = min(float(state["sprint_speed"]),
                                       self._native_sprint_speed)
        return super()._project_endpoint(bounded, pose, target, sprint)


class RecoveryColony:
    """Single-active-guide controller with three sequential native replacements."""

    MAX_REPLACEMENTS = 3

    def __init__(self, static_map, bait_id=0, guide_id=1):
        self.static_map = static_map
        self.bait_id = bait_id
        self.original_guide_id = guide_id
        self.active_guide_id = guide_id
        self.guide_role_ids = [guide_id]
        self.controllers = {guide_id: ColonyScout(static_map, bait_id=bait_id,
                                                   guide_id=guide_id)}
        self.pending_birth = False
        self.pending_since = None
        self.birth_action_due = False
        self.missing_since = None
        self.replacements_requested = 0
        self.finished = False
        self.events = []
        self.decisions = {}
        self.site = self.controllers[guide_id].site
        self.bait_ready = False

    def _register_new_child(self, states, sim_time):
        known = set(self.guide_role_ids) | {self.bait_id}
        children = sorted(set(states) - known)
        if not children:
            return
        # Only one request can be outstanding, so the first new native ID is
        # the replacement. Any unexpected extras remain stationary below.
        aid = children[0]
        self.guide_role_ids.append(aid)
        scout = ColonyScout(self.static_map, bait_id=self.bait_id, guide_id=aid)
        predecessor = self.controllers.get(self.active_guide_id)
        inherited_point = (getattr(predecessor, "last_predator_point", None)
                           if predecessor is not None else None)
        inherited_time = (getattr(predecessor, "last_predator_seen", -1e9)
                          if predecessor is not None else -1e9)
        # This is policy memory produced from an earlier ordinary DTO and
        # static-map localization. It gives the replacement a route toward the
        # last public sighting; the global sweep takes over when it is stale.
        if inherited_point is not None:
            scout.last_predator_point = tuple(inherited_point)
            scout.last_predator_seen = float(inherited_time)
        self.controllers[aid] = scout
        self.active_guide_id = aid
        self.pending_birth = False
        self.pending_since = None
        self.missing_since = None
        self.events.append({"time": round(sim_time, 1),
                            "kind": "native_colony_scout_registered",
                            "agent_id": aid,
                            "replacement_number": self.replacements_requested,
                            "inherited_public_last_sighting": inherited_point is not None})

    @staticmethod
    def _still(agent_id):
        return {"agent_id": agent_id, "move_distance": 0.,
                "move_direction": 0., "turn_angle": 0., "spawn_agent": False}

    @staticmethod
    def _nearest_predator(state):
        return min((float(row["distance"]) for row in state["observations"]
                    if row.get("type") == "Predator"), default=None)

    def act(self, observations, sim_time):
        states = {state["agent_id"]: state for state in observations}
        self._register_new_child(states, sim_time)
        active = self.active_guide_id

        controller = self.controllers.get(active)
        if controller is not None and getattr(controller, "released", False):
            self.finished = True

        if active in states:
            self.missing_since = None
        elif self.missing_since is None:
            self.missing_since = sim_time
        if (self.pending_birth and self.pending_since is not None
                and sim_time - self.pending_since >= 1.):
            # A request may produce a child that dies before reaching a DTO.
            # Permit a later attempt, while the action-count budget remains.
            self.pending_birth = False
            self.pending_since = None
        bait_nearest = (self._nearest_predator(states[self.bait_id])
                        if self.bait_id in states else None)
        bait_holding = bait_nearest is not None and bait_nearest < 30.
        settled = self.missing_since is not None and sim_time - self.missing_since >= 3.
        if (not self.finished and active not in states and settled and not bait_holding
                and not self.pending_birth
                and self.replacements_requested < self.MAX_REPLACEMENTS
                and self.bait_id in states):
            self.replacements_requested += 1
            self.pending_birth = True
            self.pending_since = sim_time
            self.birth_action_due = True
            self.events.append({"time": round(sim_time, 1),
                                "kind": "bait_colony_replacement_request",
                                "dead_guide_id": active,
                                "replacement_number": self.replacements_requested})

        actions = []
        self.decisions = {}
        for aid in sorted(states):
            if aid == self.bait_id:
                action = self._still(aid)
                if self.birth_action_due:
                    action["spawn_agent"] = True
                    self.birth_action_due = False
                self.decisions[aid] = {"rule": "stationary_bait_colony",
                                      "replacement_pending": self.pending_birth,
                                      "bait_close_predator_hold": bait_holding}
            elif aid == self.active_guide_id and not self.finished:
                scout = self.controllers[aid]
                action = scout.act([states[aid]], sim_time)[0]
                action["spawn_agent"] = False
                self.decisions[aid] = dict(scout.decisions.get(aid, {}))
                self.decisions[aid]["sequential_colony_scout"] = True
            else:
                action = self._still(aid)
                self.decisions[aid] = {"rule": "inactive_colony_member"}
            actions.append(action)
        return actions
