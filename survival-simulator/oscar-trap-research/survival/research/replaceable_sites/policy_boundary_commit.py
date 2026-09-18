"""Untested follow-up: commit a close emergency direction for four ticks."""
import math

from replaceable_sites.policy_boundary_no_hold import BoundaryNoHold as Base
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add, sub, unit


class BoundaryCommittedEscape(Base):
    """Prevent the fitted no-hold repair from reversing on its next tick."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boundary_escape_vector = None
        self._boundary_escape_ticks = 0

    def _escape_plan(self, state, pose, desired):
        if self._boundary_escape_ticks and self._boundary_escape_vector is not None:
            step = state["sprint_speed"] * self._terrain_modifier(pose.p)
            q = add(pose.p, (self._boundary_escape_vector[0] * step,
                             self._boundary_escape_vector[1] * step))
            if self._clear(pose.p, q, 5.01):
                self._boundary_escape_ticks -= 1
                return q
            self._boundary_escape_ticks = 0
        target = super()._escape_plan(state, pose, desired)
        if (target is not None and self.last_predator_point is not None
                and math.dist(pose.p, self.last_predator_point) < 50
                and math.dist(target, pose.p) > .5):
            self._boundary_escape_vector = unit(sub(target, pose.p))
            self._boundary_escape_ticks = 3
        return target
