"""Spend native reserve births only when ordinary observations show an emergency.

Normal encounters retain one guide. A static-map blocked escape cone, or close
pursuit beside a wall, authorizes the already-bounded eight-child reserve rule.
No fixture position, predator target, energy, or rest state enters this policy.
"""
import math
from backup_guides.burst import BurstGuides


class EmergencyGuides(BurstGuides):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emergency_reserves = False
        self.burst_used = 8

    def act(self, observations, sim_time):
        if not self.emergency_reserves:
            for state in observations:
                aid = state['agent_id']
                if aid == self.bait_id:
                    continue
                controller = self._controller(aid)
                if controller.released:
                    continue
                seen = [o for o in state['observations'] if o['type'] == 'Predator']
                if not seen:
                    continue
                obs = min(seen, key=lambda o:o['distance'])
                if obs['distance'] >= 70.:
                    continue
                pose = controller._localize(state) or controller.poses.get(aid)
                if pose is None:
                    continue
                predator = pose.point(obs)
                away = math.atan2(pose.p[1]-predator[1], pose.p[0]-predator[0])
                escape = any(controller._clear(pose.p,
                    (pose.p[0]+75*math.cos(away+offset), pose.p[1]+75*math.sin(away+offset)),5.01)
                    for offset in (-1.3,-1.,-.7,-.35,0.,.35,.7,1.,1.3))
                close_wall = obs['distance'] < 40. and not controller._free(pose.p,30.)
                if not escape or close_wall:
                    self.emergency_reserves = True
                    self.burst_used = 0
                    self.events.append(dict(time=round(sim_time,1), kind='emergency_reserve_budget_enabled',
                        observed_distance=obs['distance'], blocked_escape_cone=not escape,
                        close_wall=close_wall, agent_id=aid))
                    break
        return super().act(observations, sim_time)
