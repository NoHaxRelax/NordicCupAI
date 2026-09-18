"""Independent v24 guides with at most three native reserve births.

Each guide controller receives only that guide's native DTO, public time, its
role ID, and the immutable static map. The wrapper shares no inferred pose or
predator state between controllers. Active-guide reporting is derived only
from Predator distances present in public DTOs and never affects actions.
"""
from __future__ import annotations

from simple_chase.policy_v24_short_fallback import SimpleChase as Guide


class BackupGuides:
    def __init__(self, static_map, bait_id=0, guide_id=1):
        self.static_map = static_map
        self.bait_id, self.original_guide_id = bait_id, guide_id
        self.controllers = {}
        self.guide_role_ids = [guide_id]
        self.child_ids = set()
        self.active_guide_id = guide_id
        self.events = []
        self.decisions = {}
        self.births_requested = 0
        self.last_birth_time = -1e9
        self.first_predator_seen = False
        self.site = self._controller(guide_id).site
        self.bait_ready = False

    def _controller(self, aid):
        if aid not in self.controllers:
            controller = Guide(self.static_map, bait_id=self.bait_id, guide_id=aid)
            # Instances intentionally retain separate localization, route,
            # pursuit, velocity, phase, and escape-planner state.
            self.controllers[aid] = controller
        return self.controllers[aid]

    @staticmethod
    def _nearest_predator(state):
        rows = [o for o in state["observations"] if o.get("type") == "Predator"]
        return min((float(o["distance"]) for o in rows), default=None)

    def _update_active(self, states, sim_time):
        live = [aid for aid in self.guide_role_ids if aid in states]
        if not live:
            return
        distances = {aid: self._nearest_predator(states[aid]) for aid in live}
        visible = {aid: d for aid, d in distances.items() if d is not None}
        current = self.active_guide_id
        if current not in live:
            chosen = min(visible, key=visible.get) if visible else live[0]
        elif visible:
            challenger = min(visible, key=visible.get)
            current_distance = distances.get(current)
            # Five-unit hysteresis avoids role chatter between nearby siblings.
            chosen = (challenger if current_distance is None
                      or visible[challenger] + 5 < current_distance else current)
        else:
            chosen = current
        if chosen != current:
            self.active_guide_id = chosen
            self.events.append({"time": round(sim_time, 1),
                                "kind": "public_dto_active_guide",
                                "agent_id": chosen,
                                "reason": "minimum_observed_predator_distance_hysteresis"})

    def act(self, observations, sim_time):
        states = {s["agent_id"]: s for s in observations}
        for aid in sorted(states):
            if aid != self.bait_id and aid not in self.guide_role_ids:
                self.guide_role_ids.append(aid)
                self.child_ids.add(aid)
                self.events.append({"time": round(sim_time, 1), "kind": "native_backup_registered",
                                    "agent_id": aid})
        self._update_active(states, sim_time)
        actions = []
        self.decisions = {}
        if self.bait_id in states:
            actions.append({"agent_id": self.bait_id, "move_distance": 0.,
                            "move_direction": 0., "turn_angle": 0.,
                            "spawn_agent": False})
            self.decisions[self.bait_id] = {"rule": "stationary_predeployed_bait"}
        for aid in sorted(states):
            if aid == self.bait_id:
                continue
            controller = self._controller(aid)
            action = controller.act([states[aid]], sim_time)[0]
            action["spawn_agent"] = False
            decision = dict(controller.decisions.get(aid, {}))
            decision["independent_backup_guide"] = True
            self.decisions[aid] = decision
            actions.append(action)

        original = states.get(self.original_guide_id)
        if original is not None and self.births_requested < 3:
            nearest = self._nearest_predator(original)
            first = nearest is not None and not self.first_predator_seen
            close = (nearest is not None and nearest < 50
                     and sim_time - self.last_birth_time >= 2.)
            if first or close:
                self.first_predator_seen = True
                self.births_requested += 1
                self.last_birth_time = sim_time
                for action in actions:
                    if action["agent_id"] == self.original_guide_id:
                        action["spawn_agent"] = True
                        break
                self.events.append({"time": round(sim_time, 1), "kind": "backup_birth_request",
                                    "agent_id": self.original_guide_id,
                                    "request_number": self.births_requested,
                                    "observed_distance": round(nearest, 3)})
        return sorted(actions, key=lambda row: row["agent_id"])
