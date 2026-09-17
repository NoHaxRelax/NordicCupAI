"""Integrated V28 terrain barrier with bounded emergency-only reserves."""
from short_overlap_sol.policy_v28_terrain_barrier import Guide as TerrainGuide
from backup_guides.emergency import EmergencyGuides as EmergencyBase


class Guide(TerrainGuide):
    def act(self, observations, sim_time):
        # V27's sweep may otherwise fall back to its pre-action pose even when
        # current native edges have just disproved that pose.  Remove rejected
        # cache entries before the cooperative release/sweep chain snapshots.
        for state in observations:
            aid=state['agent_id'];pose=self.poses.get(aid)
            if (pose is not None and self._local_edges(state)
                    and not self._edges_consistent(state,pose,tolerance=.00001)):
                self.poses.pop(aid,None)
        return super().act(observations,sim_time)


class EmergencyGuides(EmergencyBase):
    def _controller(self,aid):
        if aid not in self.controllers:
            self.controllers[aid]=Guide(self.static_map,bait_id=self.bait_id,guide_id=aid)
        return self.controllers[aid]
