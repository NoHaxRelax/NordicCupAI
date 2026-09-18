"""DTO-only prepared handoff policy using an immutable static map."""
from __future__ import annotations

import math

from real_map_guide_sol.policy import RealMapGuidePolicy, Pose, add, rot, sub, wrap
from replaceable_sites.selector import enumerate_sites


class StaticHandoffPolicy(RealMapGuidePolicy):
    def __init__(self, static_map, old_bait_id=0, replacement_id=1,
                 site_kind="interior"):
        self.width = float(static_map["width"]); self.height = float(static_map["height"])
        self.rects = [(float(o["x"]), float(o["y"]), float(o["width"]), float(o["height"]))
                      for o in static_map["obstacles"]]
        self.edges = []
        for i, (x, y, w, h) in enumerate(self.rects):
            self.edges += [((x,y),(x+w,y),i),((x+w,y),(x+w,y+h),i),
                           ((x+w,y+h),(x,y+h),i),((x,y+h),(x,y),i)]
        if site_kind == "expanded_narrowest":
            from replaceable_sites.expanded_selector import enumerate_sites as expanded_sites
            candidates = expanded_sites(static_map)
        else:
            candidates = enumerate_sites(static_map)
        if site_kind == "boundary":
            eligible = [s for s in candidates if s["boundary_indices"]]
        elif site_kind == "interior":
            eligible = [s for s in candidates if not s["boundary_indices"]
                        and s["overlap"] >= 55 and not s["offset_approach"]]
        elif site_kind == "expanded_narrowest":
            eligible = sorted(candidates, key=lambda s: (s["gap"], -s["overlap"]))
        else:
            raise ValueError("site_kind must be interior or boundary")
        if not eligible:
            raise ValueError("handoff proof requires a long interior replaceable site")
        self.site = eligible[0]
        self.old_bait_id, self.replacement_id = old_bait_id, replacement_id
        self.poses = {}; self.decisions = {}; self.last_actions = {}

    def _action(self, aid, state, target, rule):
        pose = self.poses.get(aid)
        if pose is None:
            self.decisions[aid] = {"rule": "localize_from_native_edges"}
            return dict(agent_id=aid, move_distance=0., move_direction=0.,
                        turn_angle=.25, spawn_agent=False)
        delta = sub(target, pose.p); distance = math.hypot(*delta)
        direction = wrap(math.atan2(delta[1], delta[0]) - pose.theta) if distance > .05 else 0.
        move = min(float(state["speed"]), distance)
        desired = math.atan2(self.site["inward"][1], self.site["inward"][0])
        action = dict(agent_id=aid, move_distance=move, move_direction=direction,
                      turn_angle=wrap(desired-pose.theta), spawn_agent=False)
        self.decisions[aid] = {"rule": rule}
        self.last_actions[aid] = action
        pose.p = add(pose.p, rot((move, 0.), pose.theta + direction))
        pose.theta = wrap(pose.theta + action["turn_angle"])
        return action

    def act(self, observations, sim_time):
        self.decisions = {}
        states = {s["agent_id"]: s for s in observations}
        for aid, state in states.items():
            localized = self._localize(state)
            if localized is not None:
                self.poses[aid] = localized
        actions = []
        for aid in sorted(states):
            if aid == self.replacement_id:
                target = self.site["replacement_entry"] if sim_time < 12 else self.site["goal"]
                rule = "rear_hold_before_handoff" if sim_time < 12 else "replacement_enter_opposite_mouth"
            elif aid == self.old_bait_id:
                target = self.site["goal"] if sim_time < 28 else self.site["replacement_entry"]
                rule = "old_bait_hold" if sim_time < 28 else "old_bait_retreat_opposite_mouth"
            else:
                target = self.site["replacement_entry"]; rule = "unassigned_hold"
            actions.append(self._action(aid, states[aid], tuple(target), rule))
        return actions
