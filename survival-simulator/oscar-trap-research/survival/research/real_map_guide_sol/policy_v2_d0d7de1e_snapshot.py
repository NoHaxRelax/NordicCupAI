"""Observation-only predator guide with an immutable perfect static map.

The constructor may inspect arena dimensions and obstacle rectangles.  Runtime
actions accept only native agent DTOs and public simulation time.  Dynamic
agent/predator coordinates, predator IDs, targets, energy, and rest state are
never accepted.  A dedicated bait ID and guide ID are role assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math


Point = tuple[float,float]


def add(a,b):return (a[0]+b[0],a[1]+b[1])
def sub(a,b):return (a[0]-b[0],a[1]-b[1])
def mul(a,s):return (a[0]*s,a[1]*s)
def dot(a,b):return a[0]*b[0]+a[1]*b[1]
def norm(a):return math.hypot(*a)
def unit(a):return mul(a,1/max(norm(a),1e-12))
def rot(a,t):
    c,s=math.cos(t),math.sin(t);return (a[0]*c-a[1]*s,a[0]*s+a[1]*c)
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi


@dataclass
class Pose:
    p:Point
    theta:float
    def point(self,o):
        return add(self.p,rot((o["distance"]*math.cos(o["angle"]),
                              o["distance"]*math.sin(o["angle"])),self.theta))


class RealMapGuidePolicy:
    """Select a gap refuge, localize from edges, search, and deliver one predator."""

    def __init__(self,static_map,bait_id=0,guide_id=1):
        self.width=float(static_map["width"]);self.height=float(static_map["height"])
        self.rects=[(float(o["x"]),float(o["y"]),float(o["width"]),float(o["height"]))
                    for o in static_map["obstacles"]]
        self.edges=[]
        for i,(x,y,w,h) in enumerate(self.rects):
            self.edges += [((x,y),(x+w,y),i),((x+w,y),(x+w,y+h),i),
                           ((x+w,y+h),(x,y+h),i),((x,y+h),(x,y),i)]
        self.bait_id=bait_id;self.guide_id=guide_id
        self.site=self._select_site()
        # Neither role receives its fixture position. Both establish a world
        # pose from native Edge observations before using the static map.
        self.poses={};self.localize_ticks={};self.bait_route=[];self.bait_ready=False
        self.phase="localize";self.route=[];self.search=[];self.search_index=0
        self.last_predator_seen=-1e9;self.last_predator_point=None
        self.engaged=False;self.events=[];self.decisions={};self.act_tick=0

    def _free(self,p,radius,ignore=()):
        for i,(x,y,w,h) in enumerate(self.rects):
            if i in ignore:continue
            if x-radius < p[0] < x+w+radius and y-radius < p[1] < y+h+radius:return False
        return radius<=p[0]<=self.width-radius and radius<=p[1]<=self.height-radius

    @staticmethod
    def _segment_rect(a,b,r):
        x,y,w,h=r;lo=(x,y);hi=(x+w,y+h);dx=b[0]-a[0];dy=b[1]-a[1]
        t0,t1=0.,1.
        for p,q in ((-dx,a[0]-lo[0]),(dx,hi[0]-a[0]),(-dy,a[1]-lo[1]),(dy,hi[1]-a[1])):
            if abs(p)<1e-12:
                if q<0:return False
            else:
                t=q/p
                if p<0:t0=max(t0,t)
                else:t1=min(t1,t)
                if t0>t1:return False
        return True

    def _clear(self,a,b,radius,ignore=()):
        if not self._free(a,radius,ignore) or not self._free(b,radius,ignore):return False
        for i,(x,y,w,h) in enumerate(self.rects):
            if i in ignore:continue
            if self._segment_rect(a,b,(x-radius,y-radius,w+2*radius,h+2*radius)):return False
        return True

    def _select_site(self):
        sites=[]
        for i,a in enumerate(self.rects):
            for j,b in enumerate(self.rects):
                if i==j:continue
                for axis in (0,1):
                    ax,ay,aw,ah=(a if axis==0 else (a[1],a[0],a[3],a[2]))
                    bx,by,bw,bh=(b if axis==0 else (b[1],b[0],b[3],b[2]))
                    gap=bx-(ax+aw);low=max(ay,by);high=min(ay+ah,by+bh);overlap=high-low
                    if not (10.9<=gap<=19.1 and overlap>=54.9):continue
                    cross=ax+aw+gap/2
                    xy=lambda u,v:(u,v) if axis==0 else (v,u)
                    for sign in (1,-1):
                        mouth=xy(cross,low if sign==1 else high);inward=xy(0.,float(sign))
                        goal=add(mouth,mul(inward,30.));far=add(mouth,mul(inward,-125.));hold=add(mouth,mul(inward,-75.))
                        if not (self._free(goal,5.01) and self._free(far,11.01) and self._free(hold,11.01)):continue
                        if not self._clear(far,hold,11.01):continue
                        # Prefer long overlap, central mouths and uncluttered staging.
                        margin=min(far[0],far[1],self.width-far[0],self.height-far[1])
                        score=overlap+.05*margin
                        sites.append((score,dict(mouth=mouth,inward=inward,cross=(-inward[1],inward[0]),
                            goal=goal,far=far,hold=hold,gap=gap,overlap=overlap,obstacle_indices=[i,j],axis=axis)))
        if not sites:raise ValueError("static map has no eligible 11-19 unit gap with 55-unit overlap")
        site=max(sites,key=lambda row:row[0])[1]
        return {k:(list(v) if isinstance(v,tuple) else v) for k,v in site.items()}

    def _local_edges(self,state):
        return [(tuple(o["coords"][0]),tuple(o["coords"][1])) for o in state["observations"] if o["type"]=="Edge"]

    def _localize(self,state):
        local=self._local_edges(state)
        if not local:return None
        candidates=[]
        for la,lb in local:
            lv=sub(lb,la);length=norm(lv)
            for wa,wb,_ in self.edges:
                if abs(norm(sub(wb,wa))-length)>.05:continue
                for x,y in ((wa,wb),(wb,wa)):
                    theta=wrap(math.atan2(y[1]-x[1],y[0]-x[0])-math.atan2(lv[1],lv[0]))
                    origin=sub(x,rot(la,theta))
                    # A reversed lone edge often places the observer on its
                    # solid side. Creature clearance rejects that mirror.
                    if not self._free(origin,5.01):continue
                    score=0.;matched=0
                    for u,v in local:
                        tu,tv=add(origin,rot(u,theta)),add(origin,rot(v,theta));best=1e9
                        for c,d,_ in self.edges:
                            if abs(norm(sub(d,c))-norm(sub(v,u)))>.05:continue
                            best=min(best,math.dist(tu,c)+math.dist(tv,d),math.dist(tu,d)+math.dist(tv,c))
                        score+=best;matched+=best<.15
                    if matched==len(local):candidates.append((score,origin,theta))
        if not candidates:return None
        candidates.sort(key=lambda x:x[0]);best=candidates[0]
        # Repeated equal walls can remain ambiguous even with a corner in
        # view. Accept only one pose cluster at the best residual.
        def different(c):
            return (math.dist(best[1],c[1])>.15 or
                    abs(wrap(best[2]-c[2]))>.01)
        if any(different(c) and c[0]<best[0]+.05 for c in candidates[1:]):return None
        return Pose(best[1],best[2])

    def _expanded_corners(self,radius=11.2):
        nodes=[]
        for x,y,w,h in self.rects:
            for p in ((x-radius,y-radius),(x+w+radius,y-radius),(x+w+radius,y+h+radius),(x-radius,y+h+radius)):
                if self._free(p,radius-.15):nodes.append(p)
        return nodes

    def _path(self,start,target,radius=11.0):
        start_tight=not self._free(start,radius)
        if not start_tight and self._clear(start,target,radius):return [target]
        nodes=[start,target]+self._expanded_corners(radius+.2)
        # Bound work to corners relevant to the start-target envelope plus 250.
        lo=(min(start[0],target[0])-250,min(start[1],target[1])-250);hi=(max(start[0],target[0])+250,max(start[1],target[1])+250)
        nodes=nodes[:2]+[p for p in nodes[2:] if lo[0]<=p[0]<=hi[0] and lo[1]<=p[1]<=hi[1]]
        dist=[math.inf]*len(nodes);dist[0]=0.;prev={};heap=[(0.,0)]
        while heap:
            cost,i=heapq.heappop(heap)
            if cost!=dist[i]:continue
            if i==1:break
            for j in range(1,len(nodes)):
                if i==j:continue
                d=math.dist(nodes[i],nodes[j])
                if cost+d>=dist[j]:continue
                if i==0 and start_tight:
                    # A guide may first see a predator while physically valid
                    # for itself but too close to a wall for its larger
                    # follower. Escape to a nearby predator-clear corner; all
                    # subsequent edges use predator-radius clearance.
                    clear=(j!=1 and d<=150 and self._free(nodes[j],radius)
                           and self._clear(nodes[i],nodes[j],5.01))
                else:
                    clear=self._clear(nodes[i],nodes[j],radius)
                if not clear:continue
                dist[j]=cost+d;prev[j]=i;heapq.heappush(heap,(dist[j],j))
        if not math.isfinite(dist[1]):return []
        out=[];cur=1
        while cur:out.append(nodes[cur]);cur=prev[cur]
        return list(reversed(out))

    def _search_points(self):
        if self.search:return self.search
        points=[]
        for y in range(100,int(self.height)-99,160):
            row=[]
            for x in range(100,int(self.width)-99,160):
                p=(float(x),float(y))
                if self._free(p,12):row.append(p)
            if (y//160)%2:row.reverse()
            points.extend(row)
        self.search=points;return points

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        delta=sub(target,pose.p);modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
        cap=state["sprint_speed"] if sprint else state["speed"]
        distance=min(cap,norm(delta)/modifier);direction=wrap(math.atan2(delta[1],delta[0])-pose.theta) if norm(delta)>.01 else 0.
        pose.p=add(pose.p,rot((distance*modifier,0.),pose.theta+direction));pose.theta=wrap(pose.theta+turn)
        self.decisions[aid]={"rule":rule}
        return dict(agent_id=aid,move_distance=distance,move_direction=direction,turn_angle=turn,spawn_agent=False)

    def _unlocalized(self,aid,state,rule):
        """Travel in a broad arc until enough mapped edge geometry is visible."""
        tick=self.localize_ticks.get(aid,0)+1;self.localize_ticks[aid]=tick
        turn=.035 if tick%180 else .65
        self.decisions[aid]={"rule":rule}
        return dict(agent_id=aid,move_distance=state["sprint_speed"],
                    move_direction=0.,turn_angle=turn,spawn_agent=False)

    def act(self,observations,sim_time):
        self.act_tick+=1
        states={s["agent_id"]:s for s in observations};self.decisions={};actions=[]
        for aid,state in sorted(states.items()):
            if aid==self.bait_id:
                measured=(self._localize(state) if aid not in self.poses or
                          self.act_tick%10==0 else None)
                if measured:self.poses[aid]=measured
                pose=self.poses.get(aid)
                if pose is None:
                    actions.append(self._unlocalized(aid,state,"bait_localize_from_native_edges"));continue
                goal=tuple(self.site["goal"])
                if math.dist(pose.p,goal)<4:
                    self.bait_ready=True;self.bait_route=[]
                    actions.append(self._move(aid,state,pose,goal,"hold_deployed_bait_in_refuge",sprint=False));continue
                if not self.bait_route:self.bait_route=self._path(pose.p,goal,5.01)
                while self.bait_route and math.dist(pose.p,self.bait_route[0])<4:self.bait_route.pop(0)
                target=self.bait_route[0] if self.bait_route else pose.p
                actions.append(self._move(aid,state,pose,target,"deploy_random_start_bait_by_static_map_route",sprint=True));continue
            if aid!=self.guide_id:
                actions.append(dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=.25,spawn_agent=False));continue
            measured=(self._localize(state) if aid not in self.poses or
                      self.act_tick%10==0 else None)
            if measured:self.poses[aid]=measured
            pose=self.poses.get(aid)
            threats=[o for o in state["observations"] if o["type"]=="Predator"]
            if pose is None:
                actions.append(self._unlocalized(aid,state,"guide_localize_from_native_edges"));continue
            if not self.bait_ready:
                if threats:
                    nearest=min(threats,key=lambda o:o["distance"])
                    target=pose.point(dict(distance=100.,angle=wrap(nearest["angle"]+math.pi)))
                    target=(min(max(target[0],12.),self.width-12.),
                            min(max(target[1],12.),self.height-12.))
                    actions.append(self._move(aid,state,pose,target,"evade_until_bait_deployed",turn=nearest["angle"],sprint=True))
                else:
                    actions.append(self._move(aid,state,pose,pose.p,"wait_for_random_start_bait_deployment",turn=.35,sprint=False))
                continue
            if threats:
                self.last_predator_seen=sim_time
                nearest_now=min(threats,key=lambda o:o["distance"])
                self.last_predator_point=pose.point(nearest_now)
                if not self.engaged:
                    self.engaged=True;self.phase="route_to_refuge";self.route=[]
                    self.events.append(dict(time=round(sim_time,2),kind="engaged_predator_from_native_observation"))
            nearest=min(threats,key=lambda o:o["distance"]) if threats else None
            face_away=wrap(nearest["angle"]+math.pi) if nearest else .18

            if self.phase in ("localize","search") and not self.engaged:
                self.phase="search";points=self._search_points()
                if not self.route:
                    target=points[self.search_index%len(points)];self.search_index+=1
                    self.route=self._path(pose.p,target,11.)
                while self.route and math.dist(pose.p,self.route[0])<5:self.route.pop(0)
                target=self.route[0] if self.route else pose.p
                actions.append(self._move(aid,state,pose,target,"static_map_search_for_predator",turn=.22,sprint=True));continue

            if self.phase=="route_to_refuge":
                if nearest is None:
                    target=pose.p;rule="scan_for_follower_reobservation"
                    if (self.last_predator_point is not None and
                            sim_time-self.last_predator_seen<=8):
                        reacquire=self._path(pose.p,self.last_predator_point,5.01)
                        if reacquire:target=reacquire[0];rule="bounded_retrace_to_last_observed_predator"
                    actions.append(self._move(aid,state,pose,target,rule,turn=.35,sprint=False));continue
                if nearest["distance"]>48:
                    actions.append(self._move(aid,state,pose,pose.p,"wait_for_follower_to_close",turn=face_away,sprint=False));continue
                if not self.route:self.route=self._path(pose.p,tuple(self.site["far"]),11.)
                while self.route and math.dist(pose.p,self.route[0])<6:self.route.pop(0)
                if math.dist(pose.p,tuple(self.site["far"]))<6:
                    self.phase="wait_aligned_follower";self.route=[]
                elif not self.route:
                    actions.append(self._move(aid,state,pose,pose.p,"predator_clear_route_unavailable",turn=face_away,sprint=False));continue
                else:
                    actions.append(self._move(aid,state,pose,self.route[0],"obstacle_aware_follower_route",turn=face_away,sprint=False));continue
            if self.phase=="wait_aligned_follower":
                behind=[o for o in threats if abs(o["angle"])>math.pi/2]
                if behind and min(o["distance"] for o in behind)<=50:
                    self.phase="axial_lead";self.events.append(dict(time=round(sim_time,2),kind="aligned_follower_lead"))
                else:
                    actions.append(self._move(aid,state,pose,tuple(self.site["far"]),"wait_125_outside_refuge",turn=face_away,sprint=False));continue
            if self.phase=="axial_lead":
                if math.dist(pose.p,tuple(self.site["hold"]))<5:self.phase="hold_for_sacrifice"
                else:
                    actions.append(self._move(aid,state,pose,tuple(self.site["hold"]),"lead_50_along_refuge_axis",turn=face_away,sprint=True));continue
            actions.append(self._move(aid,state,pose,tuple(self.site["hold"]),"hold_75_outside_refuge",turn=face_away,sprint=False))
        return actions
