"""Eight native reserves with exactly one stateful pursuing guide.

Election and reserve behavior use only each agent's ordinary DTO. Inactive
reserves never call their v30 planner, preventing pose integration for actions
that the wrapper replaces. They either move directly away from an observed
near predator or hold and scan. Only the elected guide plans toward the bait.
"""
from __future__ import annotations

import math

from integrated_guide.policy_v30_wide_route import Guide as WideGuide
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import wrap


class NativeTraitWideGuide(WideGuide):
    """Bound hypothetical endpoints by the elected native agent's traits."""

    def act(self, observations, sim_time):
        if observations:
            state = observations[0]
            self._native_speed = float(state["speed"])
            self._native_sprint_speed = float(state["sprint_speed"])
        return super().act(observations, sim_time)

    def _project_endpoint(self, state, pose, target, sprint=False):
        bounded = dict(state)
        bounded["speed"] = min(float(state["speed"]), self._native_speed)
        bounded["sprint_speed"] = min(float(state["sprint_speed"]),
                                       self._native_sprint_speed)
        return super()._project_endpoint(bounded, pose, target, sprint)


class LeaderReserves:
    MAX_BIRTHS = 8
    ELECTION_HYSTERESIS = 5.0

    def __init__(self, static_map, bait_id=0, guide_id=1):
        self.static_map = static_map
        self.bait_id = bait_id
        self.original_guide_id = guide_id
        self.guide_role_ids = [guide_id]
        self.active_guide_id = guide_id
        self.controllers = {}
        self.births_requested = 0
        self.parent_births = {}
        self.events = []
        self.decisions = {}
        self.site = self._controller(guide_id).site
        self.bait_ready = False

    def _controller(self, aid):
        if aid not in self.controllers:
            self.controllers[aid] = NativeTraitWideGuide(
                self.static_map, bait_id=self.bait_id, guide_id=aid)
        return self.controllers[aid]

    @staticmethod
    def _nearest(state):
        predators = [row for row in state["observations"]
                     if row.get("type") == "Predator"]
        return min(predators, key=lambda row: float(row["distance"])) if predators else None

    @staticmethod
    def _still(aid, turn=0.3):
        return {"agent_id": aid, "move_distance": 0., "move_direction": 0.,
                "turn_angle": turn, "spawn_agent": False}

    def _elect(self, states, visible, sim_time):
        live = [aid for aid in self.guide_role_ids if aid in states]
        if not live:
            return
        old = self.active_guide_id
        if visible:
            challenger = min(visible, key=lambda aid: float(visible[aid]["distance"]))
            if old not in visible:
                chosen = challenger
            else:
                chosen = (challenger
                          if float(visible[challenger]["distance"]) + self.ELECTION_HYSTERESIS
                          < float(visible[old]["distance"]) else old)
        else:
            chosen = old if old in live else live[0]
        if chosen != old:
            self.active_guide_id = chosen
            self.events.append({"time": round(sim_time, 1),
                                "kind": "dto_closest_informed_leader_elected",
                                "agent_id": chosen, "previous_agent_id": old})

    def _reserve_action(self, state):
        aid = state["agent_id"]
        nearest = self._nearest(state)
        if nearest is None:
            return self._still(aid), "uninformed_reserve_hold_scan"
        distance = float(nearest["distance"])
        angle = float(nearest["angle"])
        if distance >= 100.:
            return self._still(aid, turn=angle), "informed_reserve_hold_gaze"
        # Movement is relative to the current heading and native turning occurs
        # afterward. Move away now, then face the predator for the next DTO.
        action = {"agent_id": aid,
                  "move_distance": float(state["sprint_speed"]),
                  "move_direction": wrap(angle + math.pi),
                  "turn_angle": angle,
                  "spawn_agent": False}
        return action, "informed_reserve_direct_evade"

    def act(self, observations, sim_time):
        states = {state["agent_id"]: state for state in observations}
        for aid in sorted(states):
            if aid != self.bait_id and aid not in self.guide_role_ids:
                self.guide_role_ids.append(aid)
                self._controller(aid)
                self.events.append({"time": round(sim_time, 1),
                                    "kind": "native_leader_reserve_registered",
                                    "agent_id": aid})
        visible = {aid: nearest for aid, state in states.items()
                   if aid != self.bait_id
                   for nearest in [self._nearest(state)] if nearest is not None}
        self._elect(states, visible, sim_time)

        actions = []
        self.decisions = {}
        for aid in sorted(states):
            if aid == self.bait_id:
                action = self._still(aid, turn=0.)
                rule = "stationary_predeployed_bait"
            elif aid == self.active_guide_id:
                controller = self._controller(aid)
                action = controller.act([states[aid]], sim_time)[0]
                action["spawn_agent"] = False
                decision = dict(controller.decisions.get(aid, {}))
                decision["single_active_reserve_leader"] = True
                self.decisions[aid] = decision
                actions.append(action)
                continue
            else:
                action, rule = self._reserve_action(states[aid])
            self.decisions[aid] = {"rule": rule,
                                   "single_active_reserve_leader": False}
            actions.append(action)

        if self.births_requested < self.MAX_BIRTHS:
            candidates = [(float(row["distance"]),
                           -self.parent_births.get(aid, 0), aid)
                          for aid, row in visible.items()
                          if float(row["distance"]) < 75.]
            if candidates:
                distance, _, parent = max(candidates)
                for action in actions:
                    if action["agent_id"] == parent:
                        action["spawn_agent"] = True
                        break
                self.births_requested += 1
                self.parent_births[parent] = self.parent_births.get(parent, 0) + 1
                self.events.append({"time": round(sim_time, 1),
                                    "kind": "bounded_leader_reserve_birth",
                                    "parent_id": parent,
                                    "request_number": self.births_requested,
                                    "observed_distance": distance})
        return actions
