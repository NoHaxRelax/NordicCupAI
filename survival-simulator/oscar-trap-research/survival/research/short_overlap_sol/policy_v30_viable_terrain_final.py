"""V27 with viability-gated terrain avoidance and a moving final handoff."""
import math
from integrated_guide.policy_v27 import Guide as V27Guide
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add


class Guide(V27Guide):
    _native_modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)

    def act(self,observations,sim_time):
        for state in observations:
            aid=state['agent_id'];pose=self.poses.get(aid)
            if (pose is not None and self._local_edges(state)
                    and not self._edges_consistent(state,pose,tolerance=.00001)):
                self.poses.pop(aid,None)
        return super().act(observations,sim_time)

    def _robust_not_slower(self,q,baseline,radius=8.):
        return all(self._terrain_modifier((q[0]+radius*math.cos(math.tau*k/16),
                                           q[1]+radius*math.sin(math.tau*k/16)))+1e-9>=baseline
                   for k in range(16)) and self._terrain_modifier(q)+1e-9>=baseline

    def _future_viable(self,q,p,speed,baseline,depth=1):
        if depth>2:return True
        floor=50.+15.*depth
        step=speed*baseline
        away=math.atan2(q[1]-p[1],q[0]-p[0])
        for offset in (0.,.3,-.3,.6,-.6,.9,-.9,1.2,-1.2,1.55,-1.55,2.,-2.,math.pi):
            r=add(q,(step*math.cos(away+offset),step*math.sin(away+offset)))
            if (math.dist(r,p)>=floor and self._clear(q,r,5.01)
                    and self._robust_not_slower(r,baseline)
                    and self._future_viable(r,p,speed,baseline,depth+1)):
                return True
        return False

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        seen=[o for o in state['observations'] if o['type']=='Predator']
        obs=min(seen,key=lambda o:o['distance']) if seen else None
        # Once the follower is close in the final lane, keep translating through
        # the mouth. Standing at the hold point produced distant autonomous
        # captures; slow biomes require sprinting to maintain any lead.
        if (obs and obs['distance']<50. and rule in
                ('committed_look_away_final_lead','observe_final_follower')):
            target=tuple(self.site['goal']);sprint=True;rule='moving_final_handoff'
        baseline=self._native_modifier[state['biome']]
        endpoint=self._project_endpoint(state,pose,target,sprint)
        if (obs and obs['distance']<65.
                and not self._robust_not_slower(endpoint,baseline)):
            p=pose.point(obs);speed=float(state['sprint_speed']);options=[]
            for cap in (speed,speed*.75,speed*.5):
                step=cap*baseline
                for k in range(32):
                    a=math.tau*k/32;r=add(pose.p,(step*math.cos(a),step*math.sin(a)))
                    if (self._clear(pose.p,r,5.01) and self._robust_not_slower(r,baseline)
                            and math.dist(r,p)>=50.
                            and self._future_viable(r,p,speed,baseline)):
                        progress=math.dist(pose.p,target)-math.dist(r,target)
                        options.append((progress-.15*abs(math.dist(r,p)-65.),r))
            if options:
                target=max(options,key=lambda row:row[0])[1]
                sprint=True;rule='viable_terrain_barrier'
        return super()._move(aid,state,pose,target,rule,turn,sprint)

ReleaseGuide=Guide
