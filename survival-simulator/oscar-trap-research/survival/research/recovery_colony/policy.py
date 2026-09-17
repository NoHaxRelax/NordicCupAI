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


class ColonyScout(GlobalSweepGaze):
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
        self.controllers[aid] = ColonyScout(self.static_map, bait_id=self.bait_id,
                                             guide_id=aid)
        self.active_guide_id = aid
        self.pending_birth = False
        self.events.append({"time": round(sim_time, 1),
                            "kind": "native_colony_scout_registered",
                            "agent_id": aid,
                            "replacement_number": self.replacements_requested})

    @staticmethod
    def _still(agent_id):
        return {"agent_id": agent_id, "move_distance": 0.,
                "move_direction": 0., "turn_angle": 0., "spawn_agent": False}

    def act(self, observations, sim_time):
        states = {state["agent_id"]: state for state in observations}
        self._register_new_child(states, sim_time)
        active = self.active_guide_id

        controller = self.controllers.get(active)
        if controller is not None and getattr(controller, "released", False):
            self.finished = True

        if (not self.finished and active not in states and not self.pending_birth
                and self.replacements_requested < self.MAX_REPLACEMENTS
                and self.bait_id in states):
            self.replacements_requested += 1
            self.pending_birth = True
            self.events.append({"time": round(sim_time, 1),
                                "kind": "bait_colony_replacement_request",
                                "dead_guide_id": active,
                                "replacement_number": self.replacements_requested})

        actions = []
        self.decisions = {}
        for aid in sorted(states):
            if aid == self.bait_id:
                action = self._still(aid)
                if self.pending_birth:
                    action["spawn_agent"] = True
                self.decisions[aid] = {"rule": "stationary_bait_colony",
                                      "replacement_pending": self.pending_birth}
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
