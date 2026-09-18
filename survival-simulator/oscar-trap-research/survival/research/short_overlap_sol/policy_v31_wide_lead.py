"""Wide terrain routes with a deliberately larger steady chase lead.

The predator is kept at a predicted60-unit moving separation (65 stationary),
inside the native direct-chase pivot threshold while leaving more safety margin.
"""
from integrated_guide.policy_v31_compact_fallback import Guide as Base
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import add,sub,mul,unit,wrap
import math

class Guide(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.site=dict(self.site)
        self.site['hold']=tuple(self.site['mouth'][i]-self.site['inward'][i]*95. for i in range(2))

    def act(self, observations, sim_time):
        self.act_tick += 1
        self.decisions = {}
        actions = []
        for state in observations:
            aid = state['agent_id']
            measured = self._localize(state)
            if measured:
                self.poses[aid] = measured
            pose = self.poses.get(aid)
            threats = [o for o in state['observations'] if o['type'] == 'Predator']
            nearest = min(threats, key=lambda o:o['distance']) if threats else None
            if aid == self.bait_id:
                actions.append(dict(agent_id=aid, move_distance=0., move_direction=0., turn_angle=0., spawn_agent=False))
                continue
            if pose is None:
                # Before global localization, control pursuit in the native
                # relative frame instead of outrunning the follower at full speed.
                if nearest:
                    modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state['biome']]
                    # A straight relative escape can carry an unlocalized guide
                    # all the way into the boundary before an edge permits global
                    # localization.  Sprint on a shallow, consistent arc: its
                    # outward component still exceeds the predator's 15-unit
                    # sprint, while curvature prevents a long boundary-normal run.
                    requested = state['sprint_speed']
                    offset = self._relative_escape_sign * .42
                    direction=wrap(nearest['angle']+math.pi+offset)
                    actions.append(dict(agent_id=aid,move_distance=requested,
                        move_direction=direction,turn_angle=nearest['angle'],spawn_agent=False))
                else:
                    actions.append(dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=.3,spawn_agent=False))
                self.decisions[aid]={'rule':'relative_spacing_before_localization'}
                continue
            if nearest:
                self._mpc_observation=dict(nearest)
                observed = pose.point(nearest)
                delta=sub(observed,self.last_predator_point) if self.last_predator_point is not None else (0.,0.)
                magnitude=math.hypot(*delta)
                self.pred_velocity=mul(unit(delta),min(15.,magnitude)) if sim_time-self.last_predator_seen<=.11 else (0.,0.)
                self.pred_motion=magnitude
                self.stationary_ticks = self.stationary_ticks+1 if self.pred_motion<1. else 0
                self.last_predator_point = observed
                self.last_predator_seen = sim_time
            pred = self.last_predator_point
            if pred is None:
                actions.append(self._move(aid,state,pose,pose.p,'scan',turn=.3,sprint=False)); continue
            turn = wrap(math.atan2(pred[1]-pose.p[1],pred[0]-pose.p[0])-pose.theta)
            if nearest is None:
                turn=wrap(turn + (-1.,0.,1.,0.)[self.act_tick % 4]*math.pi/3)
            distance = math.dist(pose.p,pred)
            far = self.runup if self.phase in ('localize','route') else tuple(self.site['far'])
            if self.phase == 'localize': self.phase='route'
            if self.phase=='route' and math.dist(pose.p,far)<6:
                self.phase='approach'; self.route=[]; far=tuple(self.site['far'])
            outside=-sum(a*b for a,b in zip(sub(pose.p,tuple(self.site['mouth'])),self.site['inward']))
            lane=abs(sum(a*b for a,b in zip(sub(pose.p,tuple(self.site['far'])),self.site['cross'])))
            if self.phase!='final' and 80<=outside<=150 and lane<15 and nearest and distance<70:
                axial=sum(a*b for a,b in zip(sub(pred,pose.p), self.site['inward']))
                cross=abs(sum(a*b for a,b in zip(sub(pred,pose.p),self.site['cross'])))
                if axial<0 and cross<20 and self.pred_motion>1.:
                    self.phase='final'; self.final_ticks=0
            if self.phase=='final':
                self.final_ticks+=1
                # A brief committed lead looks toward the trap, so increasing
                # separation cannot trigger the predator's watched-pivot rule.
                if self.final_ticks<=5:
                    away_turn=wrap(math.atan2(self.site['inward'][1],self.site['inward'][0])-pose.theta)
                    actions.append(self._move(aid,state,pose,tuple(self.site['hold']),'committed_look_away_final_lead',turn=away_turn,sprint=True)); continue
                if math.dist(pred,tuple(self.site['mouth']))<50:
                    actions.append(self._move(aid,state,pose,tuple(self.site['hold']),'hold_after_observed_intake',turn=turn,sprint=False)); continue
                if nearest is None or self.stationary_ticks>=3:
                    self.phase='approach'; self.route=[]; far=tuple(self.site['far'])
                else:
                    actions.append(self._move(aid,state,pose,tuple(self.site['hold']),'observe_final_follower',turn=turn,sprint=False)); continue
            if self.route and not self._clear(pose.p,self.route[0],11.): self.route=[]
            if not self.route:self.route=self._path(pose.p,far,11.)
            # Invoke the existing observation-only lookahead before ordinary
            # spacing becomes infeasible.  DTO distance is one movement stale;
            # below 47 the hidden native step can already put the guide near the
            # kill radius, leaving a one-tick emergency too late around walls.
            if nearest and nearest['distance'] < 47. and self.stationary_ticks < 3:
                actions.append(self._evade(aid,state,pose,pred,
                    'early_escape_lookahead',turn=turn,desired=(self.route[0] if self.route else pose.p)))
                continue
            self._advance_route(pose,self.route,11.)
            desired=self.route[0] if self.route else pose.p
            current_modifier=self._terrain_modifier(pose.p)
            nominal=self._project_endpoint(state,pose,desired,True)
            if (nearest and nearest['distance']<95.
                    and self._terrain_modifier(nominal)+1e-9<current_modifier):
                predicted=add(pred,mul(self.pred_velocity,2.))
                step=state['sprint_speed']*current_modifier
                lead=[]
                for k in range(32):
                    angle=math.tau*k/32
                    q=add(pose.p,(step*math.cos(angle),step*math.sin(angle)))
                    if (self._terrain_modifier(q)+1e-9>=current_modifier
                            and self._clear(pose.p,q,5.01)):
                        lead.append((math.dist(q,predicted),q))
                if lead:
                    _,q=max(lead,key=lambda row:row[0])
                    actions.append(self._move(aid,state,pose,q,
                        'build_lead_before_slow_terrain',turn=turn,sprint=True))
                    continue
            if nearest and getattr(self,'_mpc_until',0)>=self.act_tick and self.stationary_ticks<3:
                if not self._free(pose.p,30.) or distance<45:
                    actions.append(self._evade(aid,state,pose,pred,'escape_lookahead',turn=turn,desired=desired));continue
                self._mpc_until=0
            # If the ordinary Predator DTO has been absent for a full second,
            # stale distance must not authorize indefinite holding. Walk toward
            # the last observed point, then sweep a small free ring around it,
            # continuously scanning. A fresh DTO immediately returns control to
            # the v12 close-hearing spacing logic.
            lost_timeout = nearest is None and sim_time-self.last_predator_seen>1.
            if lost_timeout:
                if math.dist(pose.p,pred)>12:
                    lost_path=self._path(pose.p,pred,5.01)
                    lost_target=lost_path[0] if lost_path else pose.p
                else:
                    lost_target=pose.p
                    for offset in range(16):
                        angle=math.tau*((self.act_tick//4+offset)%16)/16
                        candidate=add(pred,(30*math.cos(angle),30*math.sin(angle)))
                        if self._free(candidate,5.01) and self._clear(pose.p,candidate,5.01):
                            lost_target=candidate;break
                scan_turn=wrap(turn+(-1.,0.,1.,0.)[self.act_tick%4]*math.pi/3)
                actions.append(self._move(aid,state,pose,lost_target,
                    'bounded_last_observation_reacquisition',turn=scan_turn,sprint=False))
                continue
            # Keep predator in hearing range across rest. DTO is one movement stale.
            if distance>95:
                # Hold for a visible follower; do not chase it away from the route.
                desired=pose.p
                if nearest is None and sim_time-self.last_predator_seen>1.:
                    reacquire=self._path(pose.p,pred,5.01)
                    if reacquire: desired=reacquire[0]
            # Native DTOs precede the previous predator movement. Predict that
            # unseen movement plus the upcoming one from observed velocity only.
            spacing_point=add(pred,mul(self.pred_velocity,2.))
            spacing_distance=math.dist(pose.p,spacing_point)
            target_spacing=65. if self.stationary_ticks>=3 else 60.
            # Select safe reachable translation; walk when its separation permits.
            candidates=[]
            for cap in (3.,6.,10.,15.,20.):
                sprint=cap>state["speed"]
                capped={**state,"speed":cap,"sprint_speed":cap}
                for k in range(24):
                    angle=math.tau*k/24
                    target=add(pose.p,(25*math.cos(angle),25*math.sin(angle)))
                    endpoint=self._project_endpoint(capped,pose,target,False)
                    if not self._clear(pose.p,endpoint,5.01):continue
                    if (nearest and nearest['distance']<95.
                            and self._terrain_modifier(endpoint)+1e-9<self._terrain_modifier(pose.p)):
                        continue
                    sep=math.dist(endpoint,spacing_point)
                    if sep<30:continue
                    # A sharp turn invalidates velocity extrapolation. Bound both
                    # unseen native movements using the last observed position.
                    if math.dist(endpoint,pred)<55.:continue
                    # Keep turns inside hearing range where feasible; direct
                    # pursuit can be lost outside60 when the predator faces away.
                    upper=75.
                    if sep>upper and distance<=80:continue
                    score=.7*(math.dist(pose.p,desired)-math.dist(endpoint,desired))-.8*abs(sep-target_spacing)-(.2 if sprint else 0)
                    candidates.append((score,endpoint,sprint))
            # Holding is valid when safely separated; avoids running off during rest.
            if 45<=spacing_distance<=75 and distance>=75.:candidates.append((-.8*abs(spacing_distance-target_spacing),pose.p,False))
            if candidates:
                _,target,sprint=max(candidates,key=lambda row:row[0])
                actions.append(self._move(aid,state,pose,target,'observed_velocity_spacing',turn=turn,sprint=sprint))
            else:
                actions.append(self._evade(aid,state,pose,spacing_point,'predicted_emergency_escape',turn=turn,desired=desired))
        return actions

ReleaseGuide=Guide
