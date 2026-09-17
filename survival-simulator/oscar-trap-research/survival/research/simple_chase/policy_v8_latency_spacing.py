"""Continuous native-observation gaze; distance-controlled guide, no birth.

Only static geometry and native DTOs enter this controller. Setup positions
and predator wake/energy/target are never supplied to it.
"""
import math
from real_map_guide_sol.policy_v15_hearing_reacquisition_frozen import RealMapGuidePolicy, add, sub, mul, unit, wrap


class SimpleChase(RealMapGuidePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_ticks = 0
        self.pred_velocity = (0.,0.)
        self.pred_motion = 0.
        self.stationary_ticks = 0
        self.runup = None
        for distance in (250.,225.,200.,180.,160.,145.):
            candidate = tuple(self.site['mouth'][i]-self.site['inward'][i]*distance for i in range(2))
            if self._free(candidate,11.) and self._clear(candidate,tuple(self.site['far']),11.):
                self.runup = candidate
                break
        if self.runup is None:
            raise ValueError('no eligible axial runup for simplified guide')

    from simple_chase.policy_v5_short_site import SimpleChase as _SiteSelector
    _select_site = _SiteSelector._select_site

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
                actions.append(dict(agent_id=aid, move_distance=state['sprint_speed'] if nearest else 0.,
                    move_direction=wrap(nearest['angle'] + math.pi) if nearest else 0.,
                    turn_angle=nearest['angle'] if nearest else .3, spawn_agent=False))
                self.decisions[aid]={'rule':'native_edge_localization_and_escape'}
                continue
            if nearest:
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
            if self.phase=='approach' and math.dist(pose.p,far)<8 and nearest and distance<80:
                axial=sum(a*b for a,b in zip(sub(pred,pose.p), self.site['inward']))
                cross=abs(sum(a*b for a,b in zip(sub(pred,pose.p),self.site['cross'])))
                if axial<0 and cross<12 and distance<=55 and abs(nearest.get('rel_dir',math.pi))<.4 and self.pred_motion>1.:
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
            self._advance_route(pose,self.route,11.)
            desired=self.route[0] if self.route else pose.p
            # Keep predator in hearing range across rest. DTO is one movement stale.
            if distance>75:
                # Hold for a visible follower; do not chase it away from the route.
                desired=pose.p
                if nearest is None and sim_time-self.last_predator_seen>1.:
                    reacquire=self._path(pose.p,pred,5.01)
                    if reacquire: desired=reacquire[0]
            # Native DTOs precede the previous predator movement. Predict that
            # unseen movement plus the upcoming one from observed velocity only.
            spacing_point=add(pred,mul(self.pred_velocity,2.))
            spacing_distance=math.dist(pose.p,spacing_point)
            target_spacing=58. if self.stationary_ticks>=3 else 50.
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
                    sep=math.dist(endpoint,spacing_point)
                    if sep<40:continue
                    # Keep turns inside hearing range where feasible; direct
                    # pursuit can be lost outside60 when the predator faces away.
                    upper=60. if nearest and abs(nearest.get('rel_dir',0.))>.5 else 65.
                    if sep>upper and distance<=80:continue
                    score=.7*(math.dist(pose.p,desired)-math.dist(endpoint,desired))-.8*abs(sep-target_spacing)-(.2 if sprint else 0)
                    candidates.append((score,endpoint,sprint))
            # Holding is valid when safely separated; avoids running off during rest.
            if 45<=spacing_distance<=65:candidates.append((-.8*abs(spacing_distance-target_spacing),pose.p,False))
            if candidates:
                _,target,sprint=max(candidates,key=lambda row:row[0])
                actions.append(self._move(aid,state,pose,target,'observed_velocity_spacing',turn=turn,sprint=sprint))
            else:
                actions.append(self._evade(aid,state,pose,spacing_point,'predicted_emergency_escape',turn=turn,desired=desired))
        return actions
