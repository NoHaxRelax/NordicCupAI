"""Observation-only encounter guide with an immutable perfect static map.

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
        self.active_guide_id=guide_id;self.guide_role_ids=[guide_id]
        self.site=self._select_site()
        # Neither role receives its fixture position. Both establish a world
        # pose from native Edge observations before using the static map.
        self.poses={};self.localize_ticks={};self.bait_route=[];self.bait_ready=False
        self.bait_goal_scan_ticks=0
        self.phase="localize";self.route=[];self.search=[];self.search_index=0
        self.last_predator_seen=-1e9;self.last_predator_point=None
        self.engaged=False;self.events=[];self.decisions={};self.act_tick=0
        self.birth_requested_at=None;self.child_born=False;self.child_ids=set()
        self.encounter_tick=0;self.follow_score=0;self.follow_confirmed=False
        self.last_follow_distance=None;self.follow_history=[]
        self.last_follow_predator_point=None;self.last_follow_guide_point=None
        self.missing_ticks=0;self.missed_gazes=0
        self.pending_guide_candidate=None;self.pending_guide_ticks=0

    def oracle_handoff(self):
        """End fixture approach without importing its hidden parent pose.

        The harness calls this at the first native Predator DTO. Only guide
        navigation memory is discarded; bait deployment remains intact. The
        parent must relocalize from the ordinary DTO's Edge observations.
        """
        self.poses.pop(self.guide_id,None);self.localize_ticks.pop(self.guide_id,None)
        self.route=[];self.engaged=False;self.phase="forage"
        self.last_predator_seen=-1e9;self.last_predator_point=None
        self.follow_score=0;self.follow_confirmed=False;self.last_follow_distance=None
        self.last_follow_predator_point=None;self.last_follow_guide_point=None
        self.missing_ticks=0;self.missed_gazes=0
        self.events.append(dict(kind="oracle_approach_cutoff_parent_pose_cleared"))

    def _switch_active_guide(self,agent_id,sim_time,reason):
        """Assign the lineage member supported by ordinary pursuit evidence."""
        if agent_id==self.active_guide_id:return
        self.active_guide_id=agent_id;self.guide_role_ids.append(agent_id)
        self.route=[];self.engaged=False;self.phase="forage"
        self.last_predator_seen=-1e9;self.last_predator_point=None
        self.follow_score=0;self.follow_confirmed=False;self.last_follow_distance=None
        self.last_follow_predator_point=None;self.last_follow_guide_point=None
        self.missing_ticks=0;self.missed_gazes=0
        self.events.append(dict(time=round(sim_time,2),kind="lineage_guide_role",
                                agent_id=agent_id,reason=reason))

    def _lineage_reserve(self,aid,state):
        predators=[o for o in state["observations"] if o["type"]=="Predator"]
        if not predators:return self._forage_unmapped(aid,state)
        q=min(predators,key=lambda o:o["distance"])
        pose=self.poses.get(aid)
        if pose is not None:
            return self._move(aid,state,pose,pose.p,
                "lineage_reserve_hold_for_target_reassignment",turn=q["angle"],sprint=False)
        self.decisions[aid]={"rule":"lineage_reserve_hold_for_target_reassignment"}
        return dict(agent_id=aid,move_distance=0.,move_direction=0.,
                    turn_angle=q["angle"],spawn_agent=False)

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
                        # The validated refuge frontier uses a five-unit bait
                        # depth from this mouth. Keep it fixed for all real-map
                        # runs so the delivery problem is evaluated directly.
                        goal=add(mouth,mul(inward,5.));far=add(mouth,mul(inward,-125.));hold=add(mouth,mul(inward,-75.))
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
        # A physical edge is repeated once per vision ray. Deduplication keeps
        # native geometry intact and makes continuous collision correction
        # cheap enough to perform whenever an edge is present.
        found={}
        for o in state["observations"]:
            if o["type"]!="Edge":continue
            a,b=tuple(o["coords"][0]),tuple(o["coords"][1])
            key=tuple(round(v,4) for p in sorted((a,b)) for v in p)
            found[key]=(a,b)
        return list(found.values())

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

    def _edges_consistent(self,state,pose,tolerance=.75):
        """Validate a prior observation-derived pose without global uniqueness."""
        local=self._local_edges(state)
        if not local:return False
        for a,b in local:
            wa,wb=add(pose.p,rot(a,pose.theta)),add(pose.p,rot(b,pose.theta))
            length=norm(sub(wb,wa));best=math.inf
            for c,d,_ in self.edges:
                if abs(norm(sub(d,c))-length)>.1:continue
                best=min(best,math.dist(wa,c)+math.dist(wb,d),
                         math.dist(wa,d)+math.dist(wb,c))
            if best>tolerance:return False
        return True

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
        if not points:points=[tuple(self.site["far"])]
        self.search=points;return points

    def _move(self,aid,state,pose,target,rule,turn=0.,sprint=True):
        delta=sub(target,pose.p);modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
        cap=state["sprint_speed"] if sprint else state["speed"]
        distance=min(cap,norm(delta)/modifier);direction=wrap(math.atan2(delta[1],delta[0])-pose.theta) if norm(delta)>.01 else 0.
        pose.p=add(pose.p,rot((distance*modifier,0.),pose.theta+direction));pose.theta=wrap(pose.theta+turn)
        self.decisions[aid]={"rule":rule}
        return dict(agent_id=aid,move_distance=distance,move_direction=direction,turn_angle=turn,spawn_agent=False)

    def _project_endpoint(self,state,pose,target,sprint=False):
        delta=sub(target,pose.p);distance=norm(delta)
        modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
        cap=state["sprint_speed"] if sprint else state["speed"]
        return add(pose.p,mul(unit(delta),min(distance,cap*modifier)))

    def _local_progress_target(self,state,pose,target,sprint=False):
        """Best collision-clear local step toward a remembered world point."""
        modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
        cap=state["sprint_speed"] if sprint else state["speed"]
        step=cap*modifier;candidates=[]
        for k in range(16):
            heading=2*math.pi*k/16
            q=add(pose.p,mul((math.cos(heading),math.sin(heading)),step))
            if self._clear(pose.p,q,5.01):candidates.append(q)
        return min(candidates,key=lambda q:math.dist(q,target),default=pose.p)

    def _unlocalized(self,aid,state,rule):
        """Travel in a broad arc until enough mapped edge geometry is visible."""
        tick=self.localize_ticks.get(aid,0)+1;self.localize_ticks[aid]=tick
        turn=.035 if tick%180 else .65
        self.decisions[aid]={"rule":rule}
        return dict(agent_id=aid,move_distance=state["sprint_speed"],
                    move_direction=0.,turn_angle=turn,spawn_agent=False)

    def _evade(self,aid,state,pose,predator,rule,turn=None,desired=None,uncertainty=0.):
        """Choose a safe, locally clear sprint with progress-biased tangent."""
        modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state["biome"]]
        step=state["sprint_speed"]*modifier
        candidates=[]
        for k in range(16):
            heading=2*math.pi*k/16
            q=add(pose.p,mul((math.cos(heading),math.sin(heading)),step))
            if self._clear(pose.p,q,5.01):candidates.append(q)
        # Native DTO positions precede the predator step that just occurred.
        # Cover that 15-unit latency plus the next possible 15-unit sprint.
        current=math.dist(pose.p,predator);threshold=max(75.+uncertainty,current+25.+uncertainty)
        safe=[q for q in candidates if math.dist(q,predator)>=threshold]
        if safe:
            desired=desired or tuple(self.site["far"])
            # Once spacing is safe, prefer tangential progress over fleeing to
            # the arena boundary. A small separation term avoids ties.
            target=max(safe,key=lambda q:(math.dist(pose.p,desired)-math.dist(q,desired)
                                          +.02*math.dist(q,predator)))
        else:
            target=max(candidates,key=lambda q:math.dist(q,predator),default=pose.p)
        if turn is None:
            turn=wrap(math.atan2(pose.p[1]-predator[1],pose.p[0]-predator[0])-pose.theta)
        return self._move(aid,state,pose,target,rule,turn=turn,sprint=True)

    def _observe_following(self,nearest,sim_time,pose):
        """Infer pursuit only from repeated ordinary Predator DTOs."""
        distance=float(nearest["distance"]);facing=abs(float(nearest.get("rel_dir",math.pi)))<math.pi/2
        predator_point=pose.point(nearest);pursuit=False
        if self.last_follow_predator_point is not None:
            motion=sub(predator_point,self.last_follow_predator_point)
            toward=unit(sub(self.last_follow_guide_point,self.last_follow_predator_point))
            pursuit=dot(motion,toward)>.2
        if facing and pursuit:self.follow_score=min(4,self.follow_score+1)
        else:self.follow_score=max(0,self.follow_score-1)
        self.last_follow_distance=distance
        self.last_follow_predator_point=predator_point;self.last_follow_guide_point=pose.p
        self.follow_history.append(dict(time=round(sim_time,2),distance=round(distance,2),
                                        facing=facing,pursuit_motion=pursuit,score=self.follow_score))
        self.follow_history=self.follow_history[-24:]
        if self.follow_score>=2 and not self.follow_confirmed:
            self.follow_confirmed=True
            self.events.append(dict(time=round(sim_time,2),kind="following_verified_from_native_history"))
        elif self.follow_score==0 and self.follow_confirmed:
            self.follow_confirmed=False
            self.events.append(dict(time=round(sim_time,2),kind="following_verification_revoked"))

    def _gaze_turn(self,nearest):
        """Face the observed follower every fourth tick, drift away otherwise."""
        if self.encounter_tick%4==0:return float(nearest["angle"])
        # Turn just beyond perpendicular so the intervening frames are a real
        # look-away, including the predator's >=90 distance pivot regime.
        return wrap(float(nearest["angle"])+math.pi/2+.12)

    def _forage_unmapped(self,aid,state):
        """Give native-born reserve agents useful observation-only behavior."""
        predators=[o for o in state["observations"] if o["type"]=="Predator"]
        fruits=[o for o in state["observations"] if o["type"]=="Fruit"]
        if predators:
            q=min(predators,key=lambda o:o["distance"]);distance=state["sprint_speed"]
            direction=wrap(q["angle"]+math.pi);turn=direction;rule="reserve_evade_observed_predator"
        elif fruits:
            q=min(fruits,key=lambda o:o["distance"]);distance=min(state["speed"],max(0.,q["distance"]-4.))
            direction=q["angle"];turn=direction;rule="reserve_gather_observed_fruit"
        else:
            distance=state["speed"];direction=0.;turn=.17;rule="reserve_wander_for_fruit"
        self.decisions[aid]={"rule":rule}
        return dict(agent_id=aid,move_distance=distance,move_direction=direction,
                    turn_angle=turn,spawn_agent=False)

    def _forage_known(self,aid,state,pose):
        fruits=[o for o in state["observations"] if o["type"]=="Fruit"]
        if fruits:
            q=min(fruits,key=lambda o:o["distance"]);target=pose.point(q)
            return self._move(aid,state,pose,target,"parent_gather_observed_fruit",
                              turn=q["angle"],sprint=False)
        target=add(pose.p,rot((100.,0.),pose.theta))
        return self._move(aid,state,pose,target,"parent_wander_for_fruit",turn=.17,sprint=False)

    def act(self,observations,sim_time):
        self.act_tick+=1
        states={s["agent_id"]:s for s in observations};self.decisions={};actions=[]
        newborns=set(states)-{self.bait_id,self.guide_id}
        switched_for_birth=False
        if newborns and not self.child_born:
            self.child_born=True;self.child_ids.update(newborns)
            self.events.append(dict(time=round(sim_time,2),kind="native_child_observed",ids=sorted(newborns)))
            self._switch_active_guide(min(newborns),sim_time,"native_newborn_near_encounter")
            switched_for_birth=True
        # Predator IDs/targets remain unavailable. Reassign only after two
        # consecutive native DTO frames point to a closer lineage member that
        # the predator is facing.
        lineage=(set(states)&({self.guide_id}|self.child_ids))
        candidates=[]
        for lineage_id in lineage:
            pp=[o for o in states[lineage_id]["observations"] if o["type"]=="Predator"
                and abs(float(o.get("rel_dir",math.pi)))<math.pi/2]
            if pp:candidates.append((min(float(o["distance"]) for o in pp),lineage_id))
        candidate=min(candidates)[1] if candidates else None
        if not switched_for_birth and candidate is not None and candidate!=self.active_guide_id:
            if candidate==self.pending_guide_candidate:self.pending_guide_ticks+=1
            else:self.pending_guide_candidate=candidate;self.pending_guide_ticks=1
            if self.pending_guide_ticks>=2:
                self._switch_active_guide(candidate,sim_time,"two_native_facing_closest_frames")
                self.pending_guide_candidate=None;self.pending_guide_ticks=0
        elif candidate==self.active_guide_id or candidate is None:
            self.pending_guide_candidate=None;self.pending_guide_ticks=0
        request_birth=False
        for aid,state in sorted(states.items()):
            if aid==self.bait_id:
                measured=(self._localize(state) if aid not in self.poses or
                          self._local_edges(state) else None)
                if measured:self.poses[aid]=measured
                pose=self.poses.get(aid)
                if pose is None:
                    actions.append(self._unlocalized(aid,state,"bait_localize_from_native_edges"));continue
                goal=tuple(self.site["goal"])
                if measured is not None and math.dist(pose.p,goal)<4:
                    self.bait_ready=True;self.bait_route=[];self.bait_goal_scan_ticks=0
                    actions.append(self._move(aid,state,pose,goal,"hold_deployed_bait_in_refuge",sprint=False));continue
                if measured is not None and math.dist(pose.p,goal)>6:
                    self.bait_ready=False;self.bait_goal_scan_ticks=0
                if self.bait_ready:
                    actions.append(self._move(aid,state,pose,goal,"hold_deployed_bait_in_refuge",sprint=False));continue
                if math.dist(pose.p,goal)<4:
                    # At depth five the open mouth can leave a forward-facing
                    # bait with no edge DTO. Rotate in place until a fresh
                    # mapped edge confirms its dead-reckoned arrival.
                    self.bait_goal_scan_ticks+=1
                    if self._edges_consistent(state,pose):
                        self.bait_ready=True;self.bait_route=[]
                        self.events.append(dict(time=round(sim_time,2),
                            kind="bait_ready_from_prior_pose_edge_consistency"))
                        actions.append(self._move(aid,state,pose,goal,
                            "hold_edge_validated_depth5_bait",sprint=False));continue
                    actions.append(self._move(aid,state,pose,pose.p,
                        "scan_edges_to_confirm_depth5_bait",turn=.35,sprint=False));continue
                if not self.bait_route:self.bait_route=self._path(pose.p,goal,5.01)
                while self.bait_route and math.dist(pose.p,self.bait_route[0])<4:self.bait_route.pop(0)
                target=self.bait_route[0] if self.bait_route else pose.p
                actions.append(self._move(aid,state,pose,target,"deploy_random_start_bait_by_static_map_route",sprint=True));continue
            if aid!=self.active_guide_id:
                if aid in lineage:actions.append(self._lineage_reserve(aid,state))
                else:actions.append(self._forage_unmapped(aid,state))
                continue
            measured=(self._localize(state) if aid not in self.poses or
                      self._local_edges(state) else None)
            if measured:self.poses[aid]=measured
            pose=self.poses.get(aid)
            threats=[o for o in state["observations"] if o["type"]=="Predator"]
            if threats:
                self.missing_ticks=0;self.missed_gazes=0
                self.last_predator_seen=sim_time
                if pose is not None:
                    nearest_now=min(threats,key=lambda o:o["distance"])
                    self.last_predator_point=pose.point(nearest_now)
                if not self.engaged:
                    self.engaged=True;self.phase="route_to_refuge";self.route=[]
                    self.encounter_tick=0;self.follow_score=0;self.follow_confirmed=False
                    self.last_follow_distance=None;self.follow_history=[]
                    self.last_follow_predator_point=None;self.last_follow_guide_point=None
                    self.events.append(dict(time=round(sim_time,2),kind="parent_encountered_predator"))
                if not self.child_born:
                    request_birth=True
                    if self.birth_requested_at is None:
                        self.birth_requested_at=sim_time
                        self.events.append(dict(time=round(sim_time,2),kind="native_child_birth_requested"))
            if pose is None:
                if threats:
                    q=min(threats,key=lambda o:o["distance"])
                    self.decisions[aid]={"rule":"unlocalized_parent_evade_after_encounter"}
                    actions.append(dict(agent_id=aid,move_distance=state["sprint_speed"],
                        move_direction=wrap(q["angle"]+math.pi),turn_angle=q["angle"],spawn_agent=False))
                else:
                    actions.append(self._forage_unmapped(aid,state))
                continue
            if self.engaged:self.encounter_tick+=1
            if self.engaged and not threats:self.missing_ticks+=1
            if not self.bait_ready:
                if threats:
                    nearest=min(threats,key=lambda o:o["distance"])
                    target=pose.point(dict(distance=100.,angle=wrap(nearest["angle"]+math.pi)))
                    target=(min(max(target[0],12.),self.width-12.),
                            min(max(target[1],12.),self.height-12.))
                    actions.append(self._move(aid,state,pose,target,"evade_until_bait_deployed",turn=nearest["angle"],sprint=True))
                elif self.engaged:
                    actions.append(self._move(aid,state,pose,pose.p,"hold_encounter_until_bait_deployed",turn=.35,sprint=False))
                else:
                    actions.append(self._forage_known(aid,state,pose))
                continue
            nearest=min(threats,key=lambda o:o["distance"]) if threats else None
            if nearest:self._observe_following(nearest,sim_time,pose)
            gaze_turn=self._gaze_turn(nearest) if nearest else .18

            if not self.engaged:
                self.phase="forage";actions.append(self._forage_known(aid,state,pose));continue

            if self.phase=="route_to_refuge":
                if nearest is None:
                    target=pose.p;rule="scan_for_follower_reobservation"
                    scheduled_gaze=self.encounter_tick%4==0
                    if scheduled_gaze and self.last_predator_point is not None:
                        self.missed_gazes+=1
                        turn=wrap(math.atan2(self.last_predator_point[1]-pose.p[1],
                                             self.last_predator_point[0]-pose.p[0])-pose.theta)
                        target=(self._local_progress_target(state,pose,self.last_predator_point)
                                if math.dist(pose.p,self.last_predator_point)>85 else pose.p)
                        actions.append(self._move(aid,state,pose,target,
                            "scheduled_gaze_to_last_observed_predator",turn=turn,sprint=False));continue
                    if self.missed_gazes==0 and self.last_predator_point is not None:
                        if not self.route:self.route=self._path(pose.p,tuple(self.site["far"]),11.)
                        while self.route and math.dist(pose.p,self.route[0])<6:self.route.pop(0)
                        if self.route:
                            bound=max(0.,math.dist(pose.p,self.last_predator_point)
                                      -15*(self.missing_ticks+1))
                            travel=unit(sub(self.route[0],pose.p));away=unit(sub(pose.p,self.last_predator_point))
                            endpoint=self._project_endpoint(state,pose,self.route[0],sprint=False)
                            post_move_bound=(math.dist(endpoint,self.last_predator_point)
                                             -15*(self.missing_ticks+2))
                            if bound<30 or post_move_bound<30 or dot(travel,away)<-.2:
                                actions.append(self._evade(aid,state,pose,self.last_predator_point,
                                    "cadence_blind_safe_tangent",turn=.12,desired=self.route[0],
                                    uncertainty=15*self.missing_ticks));continue
                            actions.append(self._move(aid,state,pose,self.route[0],
                                "cadence_blind_continue_route",turn=.12,sprint=False));continue
                    if (self.last_predator_point is not None and
                            sim_time-self.last_predator_seen<=8):
                        reacquire=self._path(pose.p,self.last_predator_point,5.01)
                        if reacquire:target=reacquire[0]
                        else:target=self._local_progress_target(state,pose,self.last_predator_point)
                        rule="bounded_retrace_to_last_observed_predator"
                    if sim_time-self.last_predator_seen>8:
                        self.engaged=False;self.phase="forage";self.route=[]
                        self.last_predator_point=None
                        self.follow_score=0;self.follow_confirmed=False;self.last_follow_distance=None
                        self.last_follow_predator_point=None;self.last_follow_guide_point=None
                        actions.append(self._move(aid,state,pose,pose.p,"acquisition_timed_out_resume_forage",turn=.35,sprint=False));continue
                    actions.append(self._move(aid,state,pose,target,rule,turn=.35,sprint=False));continue
                predator_point=pose.point(nearest)
                if not self.follow_confirmed:
                    if nearest["distance"]<65:
                        actions.append(self._evade(aid,state,pose,predator_point,
                            "safe_spacing_while_verifying_following",turn=gaze_turn));continue
                    if nearest["distance"]>90:
                        actions.append(self._move(aid,state,pose,predator_point,
                            "approach_observed_predator_until_following_verified",
                            turn=gaze_turn,sprint=False));continue
                    actions.append(self._move(aid,state,pose,pose.p,
                        "cadenced_gaze_verify_following",turn=gaze_turn,sprint=False));continue
                if nearest["distance"]>90:
                    actions.append(self._move(aid,state,pose,predator_point,
                        "close_excess_follower_separation",turn=gaze_turn,sprint=False));continue
                if nearest["distance"]>72:
                    actions.append(self._move(aid,state,pose,pose.p,"wait_for_follower_to_close",turn=gaze_turn,sprint=False));continue
                if nearest["distance"]<65:
                    desired=self.route[0] if self.route else tuple(self.site["far"])
                    actions.append(self._evade(aid,state,pose,predator_point,
                        "restore_safe_follower_spacing",turn=gaze_turn,desired=desired));continue
                if not self.route:self.route=self._path(pose.p,tuple(self.site["far"]),11.)
                while self.route and math.dist(pose.p,self.route[0])<6:self.route.pop(0)
                if math.dist(pose.p,tuple(self.site["far"]))<6:
                    self.phase="wait_aligned_follower";self.route=[]
                elif not self.route:
                    actions.append(self._move(aid,state,pose,pose.p,"predator_clear_route_unavailable",turn=gaze_turn,sprint=False));continue
                else:
                    travel=unit(sub(self.route[0],pose.p));away=unit(sub(pose.p,predator_point))
                    endpoint=self._project_endpoint(state,pose,self.route[0],sprint=False)
                    post_move_bound=math.dist(endpoint,predator_point)-30
                    if post_move_bound<35 or dot(travel,away)<-.2:
                        actions.append(self._evade(aid,state,pose,predator_point,
                            "goal_biased_tangent_around_follower",turn=gaze_turn,desired=self.route[0]));continue
                    actions.append(self._move(aid,state,pose,self.route[0],"obstacle_aware_follower_route",turn=gaze_turn,sprint=False));continue
            if self.phase=="wait_aligned_follower":
                aligned=(nearest is not None and nearest["distance"]<=50 and
                         dot(sub(pose.point(nearest),pose.p),tuple(self.site["inward"]))<0)
                if aligned:
                    self.phase="axial_lead";self.events.append(dict(time=round(sim_time,2),kind="aligned_follower_lead"))
                else:
                    actions.append(self._move(aid,state,pose,tuple(self.site["far"]),"wait_125_outside_refuge",turn=gaze_turn,sprint=False));continue
            if self.phase=="axial_lead":
                if math.dist(pose.p,tuple(self.site["hold"]))<5:self.phase="hold_for_sacrifice"
                else:
                    actions.append(self._move(aid,state,pose,tuple(self.site["hold"]),"lead_50_along_refuge_axis",turn=gaze_turn,sprint=True));continue
            actions.append(self._move(aid,state,pose,tuple(self.site["hold"]),"hold_75_outside_refuge",turn=gaze_turn,sprint=False))
        if request_birth:
            parent=next((a for a in actions if a["agent_id"]==self.guide_id),None)
            if parent is not None:parent["spawn_agent"]=True
        return actions
