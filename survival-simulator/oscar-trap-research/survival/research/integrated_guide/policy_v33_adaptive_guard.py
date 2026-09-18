"""Wide routes with terrain-scaled conservative escape alternatives.

The bound considers every static terrain cell in a square containing all
possible normal-predator moves from its observed position. Empty lookahead
preserves the earlier same-terrain separation fallback. No dynamic truth enters.
"""
import math
from integrated_guide.policy_v31_compact_fallback import Guide as Base


class Guide(Base):
    def _motion_bound(self,p,moves):
        radius=15.*moves
        height=int(self.height)
        lo_x=max(0,int(p[0]-radius));hi_x=min(int(self.width)-1,int(p[0]+radius)+1)
        lo_y=max(0,int(p[1]-radius));hi_y=min(height-1,int(p[1]+radius)+1)
        labels=set()
        for x in range(lo_x,hi_x+1):
            labels.update(self._terrain_codes[x*height+lo_y:x*height+hi_y+1])
        return moves*15.*max(self._terrain_penalties[i] for i in labels)

    def _future_viable(self,q,p,speed,baseline,floors,depth=1):
        if depth>2:return True
        step=speed*baseline
        away=math.atan2(q[1]-p[1],q[0]-p[0])
        for offset in (0.,.3,-.3,.6,-.6,.9,-.9,1.2,-1.2,1.55,-1.55,2.,-2.,math.pi):
            a=away+offset;r=(q[0]+step*math.cos(a),q[1]+step*math.sin(a))
            if (math.dist(r,p)>=floors[depth] and self._clear(q,r,5.01)
                    and self._robust_not_slower(r,baseline)
                    and self._future_viable(r,p,speed,baseline,floors,depth+1)):
                return True
        return False

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        seen=[o for o in state['observations'] if o['type']=='Predator']
        obs=min(seen,key=lambda o:o['distance']) if seen else None
        if (obs and obs['distance']<50. and rule in
                ('committed_look_away_final_lead','observe_final_follower')):
            target=tuple(self.site['goal']);sprint=True;rule='moving_final_handoff'
        baseline=self._native_modifier[state['biome']]
        endpoint=self._project_endpoint(state,pose,target,sprint)
        if obs and obs['distance']<65. and not self._robust_not_slower(endpoint,baseline):
            p=pose.point(obs);speed=float(state['sprint_speed'])
            floors=[20.+self._motion_bound(p,depth+2) for depth in range(3)]
            options=[]
            for cap in (speed,speed*.75,speed*.5):
                step=cap*baseline
                for k in range(32):
                    a=math.tau*k/32;r=(pose.p[0]+step*math.cos(a),pose.p[1]+step*math.sin(a))
                    if (self._clear(pose.p,r,5.01) and self._robust_not_slower(r,baseline)
                            and math.dist(r,p)>=floors[0]
                            and self._future_viable(r,p,speed,baseline,floors)):
                        progress=math.dist(pose.p,target)-math.dist(r,target)
                        options.append((progress-.15*abs(math.dist(r,p)-max(55.,floors[0]+10.)),r))
            if options:
                target=max(options,key=lambda row:row[0])[1]
                sprint=True;rule='terrain_scaled_viable_escape'
        return super()._move(aid,state,pose,target,rule,turn,sprint)
