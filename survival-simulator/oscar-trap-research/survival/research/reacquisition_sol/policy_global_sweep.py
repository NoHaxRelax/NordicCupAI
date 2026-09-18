"""v24 with a global static-map sweep after a stale predator observation."""
from __future__ import annotations

import math

from simple_chase.policy_v24_short_fallback import SimpleChase as Base
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import Pose


class GlobalSweepGuide(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sweep_index = None

    def act(self, observations, sim_time):
        snapshots = {aid: Pose(tuple(pose.p), pose.theta)
                     for aid, pose in self.poses.items()}
        actions = super().act(observations, sim_time)
        states = {state["agent_id"]: state for state in observations}
        for index, action in enumerate(actions):
            aid = action["agent_id"]
            if aid == self.bait_id or self.decisions.get(aid, {}).get("rule") != "bounded_last_observation_reacquisition":
                continue
            if sim_time - self.last_predator_seen <= 3.:
                continue
            state = states[aid]
            measured = self._localize(state)
            pose = measured or snapshots.get(aid)
            if pose is None:
                continue
            self.poses[aid] = pose
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
            actions[index] = self._move(aid, state, pose, waypoint,
                                        "global_static_sweep_reacquisition",
                                        turn=.3, sprint=True)
        return actions
