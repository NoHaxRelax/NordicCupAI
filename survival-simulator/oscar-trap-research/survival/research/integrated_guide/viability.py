"""Short conservative escape viability using observed positions and static map.

A DTO is one predator movement stale. For a normal15-unit predator, an agent
endpoint at least50 from the observed point leaves20 after both unseen moves.
Two static-map escape successors must reach65 and80 respectively. This is a
finite safety check, not a proof of indefinite escape. Near intake, sacrifice
remains allowed. Native reserve births are enabled only when no checked escape
exists, and capped at eight per encounter.
"""
import math
from integrated_guide.spatial_geometry import SpatialGeometry
from integrated_guide.policy_v27 import Guide as BaseGuide
from backup_guides.policy import BackupGuides


class Guide(SpatialGeometry,BaseGuide):
    OFFSETS=(0.,.3,-.3,.6,-.6,.9,-.9,1.2,-1.2,1.55,-1.55,2.,-2.,math.pi)

    def act(self,observations,sim_time):
        self.shield_exhausted=False
        return super().act(observations,sim_time)

    def _successors(self,q,p,speed,depth=1):
        if depth>2:return True
        step=speed*self._terrain_modifier(q)
        away=math.atan2(q[1]-p[1],q[0]-p[0])
        floor=50.+15.*depth
        for offset in self.OFFSETS:
            angle=away+offset
            r=(q[0]+step*math.cos(angle),q[1]+step*math.sin(angle))
            if (math.dist(r,p)>=floor and self._clear(q,r,5.01)
                    and self._successors(r,p,speed,depth+1)):
                return True
        return False

    def _safe(self,q,p,speed):
        return math.dist(q,p)>=50. and self._successors(q,p,speed)

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        seen=[o for o in state['observations'] if o['type']=='Predator']
        if aid==self.bait_id or not seen or self.released:
            return super()._move(aid,state,pose,target,rule,turn,sprint)
        obs=min(seen,key=lambda o:o['distance'])
        p=pose.point(obs)
        # The current intake relies on a permitted final guide sacrifice.
        if math.dist(p,self.site['goal'])<150. and math.dist(pose.p,self.site['goal'])<150.:
            return super()._move(aid,state,pose,target,rule,turn,sprint)
        speed=float(state['sprint_speed'])
        bounded={**state,'speed':min(float(state['speed']),speed)}
        q=self._project_endpoint(bounded,pose,target,sprint)
        if not self._clear(pose.p,q,5.01):
            q=self._local_progress_target(bounded,pose,target,sprint=sprint)
        if self._clear(pose.p,q,5.01) and self._safe(q,p,speed):
            return super()._move(aid,bounded,pose,q,rule,turn,sprint)
        options=[]
        fallback=[]
        for cap in (speed,speed*.75,speed*.5,0.):
            step=cap*self._terrain_modifier(pose.p)
            for k in range(32 if cap else 1):
                angle=math.tau*k/32
                r=(pose.p[0]+step*math.cos(angle),pose.p[1]+step*math.sin(angle))
                if not self._clear(pose.p,r,5.01):continue
                distance=math.dist(r,p)
                fallback.append((distance,r))
                if self._safe(r,p,speed):
                    progress=math.dist(pose.p,target)-math.dist(r,target)
                    options.append((progress-.45*abs(distance-65.),r))
        if options:
            q=max(options,key=lambda item:item[0])[1]
            label='conservative_three_step_escape'
        else:
            self.shield_exhausted=True
            q=max(fallback,key=lambda item:item[0])[1] if fallback else pose.p
            label='no_checked_escape_maximize_observed_separation'
        action=super()._move(aid,bounded,pose,q,label,turn,True)
        self.decisions[aid]['proposed_rule']=rule
        self.decisions[aid]['finite_escape_check_exhausted']=not bool(options)
        return action


class ReserveGuides(BackupGuides):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.births_requested=3
        self.reserve_enabled=False
        self.reserve_births=0

    def _controller(self,aid):
        if aid not in self.controllers:
            self.controllers[aid]=Guide(self.static_map,bait_id=self.bait_id,guide_id=aid)
        return self.controllers[aid]

    def act(self,observations,sim_time):
        actions=super().act(observations,sim_time)
        if not self.reserve_enabled and any(c.shield_exhausted for c in self.controllers.values()):
            self.reserve_enabled=True
            self.events.append(dict(time=round(sim_time,1),kind='reserve_budget_enabled_after_no_checked_escape'))
        if not self.reserve_enabled or self.reserve_births>=8:return actions
        candidates=[]
        for state in observations:
            aid=state['agent_id']
            if aid==self.bait_id or self.controllers[aid].released:continue
            distance=self._nearest_predator(state)
            if distance is not None and distance<75.:candidates.append((distance,aid))
        if candidates:
            distance,parent=max(candidates)
            next(a for a in actions if a['agent_id']==parent)['spawn_agent']=True
            self.reserve_births+=1
            self.events.append(dict(time=round(sim_time,1),kind='finite_escape_reserve_birth',
                parent_id=parent,number=self.reserve_births,observed_distance=distance))
        return actions
