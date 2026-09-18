"""Release a surviving guide using ordinary observation evidence only."""
import math
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import sub,wrap

class ReleaseMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.release_evidence=0
        self.released=False
        self.release_route=[]
        self.release_arrived=False

    def act(self,observations,sim_time):
        guide=next((s for s in observations if s['agent_id']==self.guide_id),None)
        if guide is not None:
            measured=self._localize(guide)
            if measured:self.poses[self.guide_id]=measured
            pose=self.poses.get(self.guide_id)
            if pose is not None:
                predators=[pose.point(o) for o in guide['observations'] if o['type']=='Predator']
                goal=tuple(self.site['goal'])
                # An observation near the bait and much farther from the guide
                # supports handoff without reading the predator's actual target.
                evidence=any(math.dist(p,goal)<35 and math.dist(p,pose.p)>math.dist(p,goal)+30 for p in predators)
                self.release_evidence=self.release_evidence+1 if evidence else 0
                if not self.released and self.release_evidence>=20:
                    self.released=True
                    self.events.append(dict(time=round(sim_time,1),kind='guide_release_observed',agent_id=self.guide_id))
                if self.released:
                    self.act_tick+=1;self.decisions={}
                    target=self.runup
                    if math.dist(pose.p,target)<8:
                        if not self.release_arrived:
                            self.events.append(dict(time=round(sim_time,1),kind='guide_returned_to_staging',agent_id=self.guide_id))
                            self.release_arrived=True
                        target=pose.p
                    if not self.release_route:self.release_route=self._path(pose.p,target,5.01)
                    self._advance_route(pose,self.release_route,5.01)
                    waypoint=self.release_route[0] if self.release_route else target
                    # Do not walk back toward a still-close predator on a detour.
                    endpoint=self._project_endpoint(guide,pose,waypoint,False)
                    if any(math.dist(endpoint,p)<max(60.,math.dist(p,goal)+20) for p in predators):waypoint=pose.p
                    nearest=min(predators,key=lambda p:math.dist(p,pose.p)) if predators else None
                    turn=0 if nearest is None else wrap(math.atan2(nearest[1]-pose.p[1],nearest[0]-pose.p[0])-pose.theta)
                    action=self._move(self.guide_id,guide,pose,waypoint,'return_to_staging_after_observed_handoff',turn=turn,sprint=False)
                    return [action if s['agent_id']==self.guide_id else dict(agent_id=s['agent_id'],move_distance=0.,move_direction=0.,turn_angle=0.,spawn_agent=False) for s in observations]
        return super().act(observations,sim_time)
