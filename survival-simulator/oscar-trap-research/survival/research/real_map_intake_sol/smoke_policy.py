"""Contract-only policy used to test the harness, not an intake candidate."""
from __future__ import annotations


class SmokePolicy:
    def __init__(self, static_map, bait_id=0, guide_id=1):
        self.bait_id = bait_id
        self.guide_id = guide_id
        # This central site is only for harness plumbing checks. A routing
        # candidate must replace it with a qualifying generated-map gap.
        self.site = {
            "mouth": [static_map["width"] / 2, static_map["height"] / 2],
            "inward": [1.0, 0.0],
            "cross": [0.0, 1.0],
            "goal": [static_map["width"] / 2, static_map["height"] / 2],
            "gap": 0.0,
            "overlap": 0.0,
            "obstacle_indices": [],
        }
        self.decisions = {}

    def act(self, observations, sim_time):
        self.decisions = {state["agent_id"]: "contract smoke: hold"
                          for state in observations}
        return [{"agent_id": state["agent_id"], "move_distance": 0.0,
                 "move_direction": 0.0, "turn_angle": 0.0,
                 "spawn_agent": False} for state in observations]

