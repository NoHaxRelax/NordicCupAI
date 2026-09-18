"""Fitted boundary diagnostic: forbid close-pursuit MPC first-step holds."""
import math

from simple_chase.policy_v21_boundary import BoundaryGuide as Base
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add


class BoundaryNoHold(Base):
    """Keep v21 unchanged except for a stationary close-escape result.

    v21's six-step beam may choose its explicit hold action as the first step
    even while a moving predator is inside the early-escape threshold. On the
    fitted map this happened four times beside obstacle 49 and consumed the
    guide's remaining separation. The fallback below uses only the same DTO-
    derived predator point, static map, public terrain, and current pose.
    """
    def _escape_plan(self, state, pose, desired):
        target = super()._escape_plan(state, pose, desired)
        pred = self.last_predator_point
        if (target is None or pred is None or math.dist(target, pose.p) > .5
                or math.dist(pred, pose.p) >= 50 or self.stationary_ticks >= 3):
            return target
        step = state["sprint_speed"] * self._terrain_modifier(pose.p)
        candidates = []
        goal = desired if desired is not None else tuple(self.site["far"])
        initial = math.dist(pose.p, goal)
        for k in range(32):
            angle = math.tau * k / 32
            q = add(pose.p, (step * math.cos(angle), step * math.sin(angle)))
            if not self._clear(pose.p, q, 5.01):
                continue
            separation = math.dist(q, pred)
            progress = initial - math.dist(q, goal)
            clearance = 1. if self._free(q, 10.) else 0.
            candidates.append((separation + .25 * progress + clearance, q))
        return max(candidates, key=lambda row: row[0])[1] if candidates else target
