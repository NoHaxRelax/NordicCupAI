"""Frozen-v15 probe with bounded emergency-direction hysteresis.

This changes only `_evade`.  Once an emergency escape direction is selected,
the controller keeps it for at most four emergency decisions.  The endpoint is
projected from the current pose and checked against static geometry afresh on
every decision.  A blocked committed direction is discarded immediately.
"""
import math

from simple_chase.policy_v15_direct_escape import SimpleChase as V15
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add


class SimpleChase(V15):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emergency_direction = None
        self.emergency_ticks_left = 0

    def _evade(self, aid, state, pose, pred, rule, turn=0., desired=None):
        observed = self.last_predator_point if self.last_predator_point is not None else pred

        # Continue the prior world-space direction, but recompute the native
        # endpoint and clearance from this tick's localized pose.
        if self.emergency_direction is not None and self.emergency_ticks_left > 0:
            target = add(pose.p, (25. * self.emergency_direction[0],
                                  25. * self.emergency_direction[1]))
            endpoint = self._project_endpoint(state, pose, target, True)
            if self._clear(pose.p, endpoint, 5.01):
                self.emergency_ticks_left -= 1
                return self._move(aid, state, pose, target,
                                  'committed_observed_escape_direction',
                                  turn=turn, sprint=True)

        candidates = []
        for k in range(48):
            angle = math.tau * k / 48
            direction = (math.cos(angle), math.sin(angle))
            target = add(pose.p, (25. * direction[0], 25. * direction[1]))
            endpoint = self._project_endpoint(state, pose, target, True)
            if self._clear(pose.p, endpoint, 5.01):
                candidates.append((math.dist(endpoint, observed), endpoint, direction))
        if not candidates:
            self.emergency_direction = None
            self.emergency_ticks_left = 0
            target = pose.p
        else:
            _, target, self.emergency_direction = max(candidates, key=lambda row: row[0])
            self.emergency_ticks_left = 3
        return self._move(aid, state, pose, target,
                          'start_committed_observed_escape_direction',
                          turn=turn, sprint=True)
