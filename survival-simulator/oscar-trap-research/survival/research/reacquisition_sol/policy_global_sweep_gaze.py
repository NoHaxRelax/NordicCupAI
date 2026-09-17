"""Global sweep that faces its movement direction for forward acquisition."""
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import wrap
from reacquisition_sol.policy_global_sweep import GlobalSweepGuide as Base


class GlobalSweepGaze(Base):
    def act(self, observations, sim_time):
        actions = super().act(observations, sim_time)
        for action in actions:
            aid = action["agent_id"]
            if self.decisions.get(aid, {}).get("rule") != "global_static_sweep_reacquisition":
                continue
            old_turn = action["turn_angle"]
            # move_direction is relative to the heading used for this action.
            # Native turning occurs after movement, so matching it points the
            # next observation cone along the current sweep segment.
            new_turn = action["move_direction"]
            action["turn_angle"] = new_turn
            pose = self.poses.get(aid)
            if pose is not None:
                pose.theta = wrap(pose.theta + new_turn - old_turn)
            self.decisions[aid]["forward_sweep_gaze"] = True
        return actions
