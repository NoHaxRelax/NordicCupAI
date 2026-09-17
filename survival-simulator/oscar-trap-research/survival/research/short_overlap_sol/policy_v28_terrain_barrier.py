"""V27 integration with an observation-only, all-action terrain barrier.

The earlier terrain gate covered route and ordinary-spacing candidates, but an
MPC/build-lead endpoint could still sit on a narrow biome boundary.  Odometry
error then put the native agent into slower terrain.  This wrapper erodes slow
terrain by a small uncertainty radius and gates every movement branch while a
fresh nearby predator DTO is present.
"""
import math

from integrated_guide.policy_v27 import Guide as V27Guide
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add


class Guide(V27Guide):
    _native_modifier = dict(forest=1., grassland=1., swamp=.5, desert=.8, river=.3)

    def _robust_not_slower(self, point, baseline, radius=8.):
        probes = [point]
        for k in range(16):
            a = math.tau*k/16
            probes.append(add(point, (radius*math.cos(a), radius*math.sin(a))))
        return all(self._terrain_modifier(q)+1e-9 >= baseline for q in probes)

    def _move(self, aid, state, pose, target, rule, turn=0., sprint=True):
        predators = [o for o in state['observations'] if o['type']=='Predator']
        nearest = min(predators,key=lambda o:o['distance']) if predators else None
        baseline = self._native_modifier[state['biome']]
        endpoint = self._project_endpoint(state,pose,target,sprint)
        guarded = bool(nearest and nearest['distance'] < 65.
                       and not self._robust_not_slower(endpoint,baseline))
        if guarded:
            observed = pose.point(nearest)
            cap = state['sprint_speed'] if sprint else state['speed']
            step = cap*baseline
            candidates=[]
            for k in range(32):
                a=math.tau*k/32
                q=add(pose.p,(step*math.cos(a),step*math.sin(a)))
                if (self._clear(pose.p,q,5.01)
                        and self._robust_not_slower(q,baseline)):
                    candidates.append((math.dist(q,observed),q))
            target=max(candidates,key=lambda row:row[0])[1] if candidates else pose.p
            rule='all_action_terrain_barrier'
        return super()._move(aid,state,pose,target,rule,turn,sprint)


# Alias used by the standard single-guide harness.
ReleaseGuide = Guide
