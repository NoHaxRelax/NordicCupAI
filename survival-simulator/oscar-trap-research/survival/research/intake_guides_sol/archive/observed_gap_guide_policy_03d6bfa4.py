"""Observation-only fast entry into an observed predator-excluding gap."""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]/"intake_geometry_sol"/"observed_gap_policy.py"
spec=importlib.util.spec_from_file_location("observed_gap_policy_base",BASE)
base=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(base)


class ObservedGapGuidePolicy(base.ObservedGapPolicy):
    """Use the public sprint cap while traversing the mapped passage route."""
    def act(self,observations,sim_time):
        actions=super().act(observations,sim_time)
        states={s["agent_id"]:s for s in observations}
        for action in actions:
            state=states[action["agent_id"]]
            if action["move_distance"] < state["speed"]-.01:
                continue
            extra=state["sprint_speed"]-action["move_distance"]
            if extra<=0:continue
            action["move_distance"]+=extra
            pose=self.poses[action["agent_id"]]
            modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
            pose.p=base.add(pose.p,base.rot((extra*modifier,0.),pose.theta+action["move_direction"]))
            self.decisions[action["agent_id"]]={"rule":"sprint_into_observed_gap"}
        return actions

