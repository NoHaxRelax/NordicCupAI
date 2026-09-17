"""Cooperative orchard colony and optional temporary predator displacement.

Policy inputs are ordinary cached observation DTOs and simulation time. There is
no engine import except ActionRequest. Oracle poses are an explicitly separate
diagnostic mode, never supplied in observation-only runs. No predator rest,
energy, identity, tree age or fruit age is available to this controller.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import statistics
import json
from src.utils.DTOs import ActionRequest


def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def bearing(a, b): return math.atan2(b[1]-a[1], b[0]-a[0])
def project(p, angle, distance):
    return (p[0]+distance*math.cos(angle), p[1]+distance*math.sin(angle))
def rotate(p, a):
    return (p[0]*math.cos(a)-p[1]*math.sin(a), p[0]*math.sin(a)+p[1]*math.cos(a))
def canonical(value):
    """Observation order is not a signal; suppress irrelevant float noise."""
    if isinstance(value,float):return round(value,6)
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    return value
def edge_point(edge, point=(0., 0.)):
    a,b=edge; dx,dy=b[0]-a[0],b[1]-a[1]
    k=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy or 1)))
    return a[0]+k*dx,a[1]+k*dy


@dataclass
class Pose:
    x: float = 0.
    y: float = 0.
    theta: float = 0.
    group: int = 0
    def xy(self): return self.x,self.y


class LocalMap:
    """Dead reckoning corrected by observed static objects and agent links.

    Separate components have arbitrary coordinate frames, never invented global
    positions. Relative agent direction aligns them when they meet. Edge/tree
    matching is local and ambiguous; errors are measured, not assumed absent.
    """
    def __init__(self):
        self.poses={}; self.features={}; self.trees={}; self.homes={};self.edges={}
        self.merges=0; self.corrections=0

    def point(self, aid, o):
        p=self.poses[aid]
        return project(p.xy(),p.theta+o['angle'],o['distance'])

    def merge(self, old, new, angle, translation):
        def tf(q):
            r=rotate(q,angle); return r[0]+translation[0],r[1]+translation[1]
        for p in self.poses.values():
            if p.group==old:
                p.x,p.y=tf(p.xy());p.theta=wrap(p.theta+angle);p.group=new
        for aid, fs in self.features.items():
            if self.poses[aid].group==new and aid in self._moving:
                self.features[aid]=[(kind,tf(q)) for kind,q in fs]
        self.trees.setdefault(new,[]).extend([(tf(q),t) for q,t in self.trees.pop(old,[])])
        self.edges.setdefault(new,[]).extend([((tf(a),tf(b)),t) for (a,b),t in self.edges.pop(old,[])])
        self.homes.pop(old,None); self.merges+=1

    def update(self, states, t, oracle=None):
        for s in states:
            aid=s['agent_id']
            if aid not in self.poses:
                self.poses[aid]=Pose(group=aid)
        if oracle is not None:
            for aid,(x,y,theta) in oracle.items(): self.poses[aid]=Pose(x,y,theta,-1)
        else:
            # Translation correction against last tick's static observations.
            for s in states:
                aid=s['agent_id'];p=self.poses[aid];prev=self.features.get(aid,[])
                residual=[]
                for o in s['observations']:
                    fs=[]
                    if o['type']=='Tree': fs=[self.point(aid,o)]
                    elif o['type']=='Edge':
                        fs=[project(p.xy(),p.theta+math.atan2(y,x),math.hypot(x,y)) for x,y in o['coords']]
                    for q in fs:
                        match=min((r for k,r in prev if k==o['type']),key=lambda r:dist(q,r),default=None)
                        if match is not None and dist(q,match)<32:
                            residual.append((match[0]-q[0],match[1]-q[1]))
                if residual:
                    # Nearest-object mismatches must not drag the coordinate
                    # frame toward a newly appearing neighboring tree.
                    clusters=[[v for v in residual if dist(v,r)<1.] for r in residual]
                    cluster=max(clusters,key=lambda c:(len(c),-math.hypot(*c[0])))
                    if len(cluster)<2 and math.hypot(*cluster[0])>8:continue
                    dx,dy=(statistics.median(v[i] for v in cluster) for i in (0,1))
                    p.x+=dx;p.y+=dy
                    if math.hypot(dx,dy)>.1:self.corrections+=1
            # Merge relative frames only with a real, identified agent observation.
            for s in states:
                aid=s['agent_id'];p=self.poses[aid]
                for o in s['observations']:
                    if o['type']!='Agent' or o['id'] not in self.poses:continue
                    other=self.poses[o['id']]
                    target=self.point(aid,o)
                    angle=wrap(p.theta+o['angle']+math.pi-o['rel_dir']-other.theta)
                    if p.group==other.group:
                        # A lower-ID neighbor provides a current relative anchor.
                        # This limits between-agent drift without world position.
                        if aid<o['id']:
                            if dist(other.xy(),target)>2:self.features[o['id']]=[]
                            other.x,other.y=target;other.theta=wrap(other.theta+angle)
                        continue
                    r=rotate(other.xy(),angle)
                    self._moving={a for a,v in self.poses.items() if v.group==other.group}
                    self.merge(other.group,p.group,angle,(target[0]-r[0],target[1]-r[1]))
        for s in states:
            aid=s['agent_id'];p=self.poses[aid];fs=[]
            tree_list=self.trees.setdefault(p.group,[])
            for o in s['observations']:
                if o['type']=='Tree':
                    q=self.point(aid,o);fs.append(('Tree',q))
                    j=min(range(len(tree_list)),key=lambda j:dist(tree_list[j][0],q),default=None)
                    if j is not None and dist(tree_list[j][0],q)<18:tree_list[j]=(q,t)
                    else:tree_list.append((q,t))
                elif o['type']=='Edge':
                    points=[project(p.xy(),p.theta+math.atan2(y,x),math.hypot(x,y)) for x,y in o['coords']]
                    fs.extend([('Edge',q) for q in points])
                    es=self.edges.setdefault(p.group,[])
                    if not any(dist(a,points[0])<3 and dist(b,points[1])<3 for (a,b),tt in es):es.append((tuple(points),t))
            self.features[aid]=fs
        for group in self.trees:
            self.edges[group]=[(edge,tt) for edge,tt in self.edges.get(group,[]) if t-tt<35]
            self.trees[group]=[(q,tt) for q,tt in self.trees[group] if t-tt<55]
            candidates=[q for q,tt in self.trees[group] if t-tt<15]
            if not candidates:continue
            old=self.homes.get(group)
            def richness(q):return sum(max(0,1-dist(q,r)/160) for r,tt in self.trees[group] if t-tt<30)
            best=max(candidates,key=richness)
            if old is None or richness(best)>richness(old)*1.45+.4:self.homes[group]=best

    def advance(self, s, action):
        p=self.poses[s['agent_id']]
        movement=min(action.move_distance,s['sprint_speed'])
        if s['energy']<s['max_energy']/5:movement=min(movement,s['speed'])
        movement*= {'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        p.x,p.y=project(p.xy(),p.theta+action.move_direction,movement)
        p.theta=wrap(p.theta+action.turn_angle)


class ColonyPolicy:
    def __init__(self, banish=False, *, variant='rest', min_fuel=300., population=4,
                 release_radius=450., guide_limit=18., risk_gate=True):
        self.banish=banish;self.variant=variant;self.min_fuel=min_fuel
        self.population=population;self.release_radius=release_radius
        self.guide_limit=guide_limit
        self.risk_gate=risk_gate
        self.map=LocalMap();self.guide=None;self.phase='forage';self.start=0.
        self.home=None;self.group=None;self.last_pred=None;self.still=0
        self.last_seen=0.;self.release=None;self.exit_angle=None;self.return_point=None
        self.exclusions=[];self.events=[];self.decisions={};self.roles={}
        self.cooldown={};self.guide_ids=set();self.ever_guides=set();self.pred_speeds={}
        self.ticks=0;self.scan_ticks={};self.time=0.
        self.acquired=False

    def event(self,t,kind,**kw):self.events.append(dict(t=round(t,2),kind=kind,**kw))

    def predators(self,s):return sorted([o for o in s['observations'] if o['type']=='Predator'],key=lambda o:o['distance'])

    def safe_heading(self,s,desired,distance,pred=None):
        """Use visible edge segments for a short collision-free heading search.

        Tests entire short segment, not endpoint tunnelling. It does not know
        unseen obstacles or terrain. Predicted predator separation is conservative.
        """
        if distance<=0:return desired
        edges=[o['coords'] for o in s['observations'] if o['type']=='Edge']
        pose=self.map.poses[s['agent_id']]
        for edge,tt in self.map.edges.get(pose.group,[]):
            if dist(edge_point(edge,pose.xy()),pose.xy())>100:continue
            edges.append(tuple(rotate((x-pose.x,y-pose.y),-pose.theta) for x,y in edge))
        if not edges:return desired
        candidates=[wrap(desired+k*math.pi/12) for k in (0,1,-1,2,-2,3,-3,4,-4,6,-6,9,-9,12)]
        def score(a):
            pts=[(r*math.cos(a),r*math.sin(a)) for r in (distance/2,distance,distance+24)]
            clearance=min((dist(edge_point(e,q),q) for e in edges for q in pts),default=100)
            value=30*math.cos(wrap(a-desired))+min(clearance,25)
            if clearance<9:value-=200
            if pred:
                q=project((0,0),pred['angle'],pred['distance'])
                value+=min(70,dist(pts[1],q))*.4
            return value
        return max(candidates,key=score)

    def __call__(self, states, t, *, oracle_poses=None):
        states=canonical(states)
        for s in states:s['observations'].sort(key=lambda o:json.dumps(o,sort_keys=True,separators=(',',':')))
        self.ticks+=1;self.time=t
        self.map.update(states,t,oracle_poses)
        byid={s['agent_id']:s for s in states};m=self.map
        self.exclusions=[e for e in self.exclusions if t<e['until']]
        self.roles={s['agent_id']:'worker' for s in states};self.decisions={}
        if self.guide is not None and self.guide not in byid:
            self.event(t,'guide_lost',aid=self.guide,phase=self.phase)
            self.guide=None;self.phase='forage'
        if self.guide is None and self.banish and len(states)>=2:
            choices=[]
            for s in states:
                aid=s['agent_id'];p=m.poses[aid];home=m.homes.get(p.group)
                pp=self.predators(s)
                if not pp or home is None or s['energy']<self.min_fuel or t<self.cooldown.get(aid,0):continue
                if min(s['sprint_speed'],20)<15.5 or s['biome'] in ('river','swamp'):continue
                predator=m.point(aid,pp[0]);d=pp[0]['distance']
                if d<42 or d>200 or dist(predator,home)>400:continue
                if self.risk_gate:
                    # An observation-supported guard can refuse a poor job. No
                    # hidden terrain or predator energy is used in this gate.
                    if s['age']>70 or d>130 or len(pp)>1:continue
                    edges=[edge for edge,tt in m.edges.get(p.group,[]) if t-tt<20]
                    if any(dist(edge_point(edge,p.xy()),p.xy())<100 for edge in edges):continue
                    away=bearing(predator,p.xy())
                    if any(dist(edge_point(edge,project(p.xy(),away,140)),project(p.xy(),away,140))<65 for edge in edges):continue
                    if s['energy']<max(self.min_fuel,s['max_energy']/5+250):continue
                workers=[m.poses[o['agent_id']] for o in states if o['agent_id']!=aid and m.poses[o['agent_id']].group==p.group]
                if not workers:continue
                # Do not remove a distant healthy forager if it cannot be the target.
                nearest=min(dist(w.xy(),predator) for w in workers)
                if d>nearest+35:continue
                choices.append((s['energy']-1.5*d,aid,p.group,home))
            if choices:
                _,self.guide,self.group,self.home=max(choices)
                self.phase='lead';self.start=t;self.last_pred=None;self.still=0;self.last_seen=t
                self.release=None;self.return_point=None;self.exit_angle=None;self.acquired=False
                self.guide_ids.add(self.guide);self.ever_guides.add(self.guide)
                self.event(t,'guide_selected',aid=self.guide,energy=round(byid[self.guide]['energy'],2),home=list(self.home))
        fruit_claims=[];actions=[]
        # Hungry workers get first claim. A guide's fruit is not reserved remotely.
        ordered=sorted(states,key=lambda s:(s['energy']/s['max_energy'],s['agent_id']))
        slots=max(0,self.population-len(states))
        for s in ordered:
            aid=s['agent_id'];p=m.poses[aid];pos=p.xy();pp=self.predators(s);pred=pp[0] if pp else None
            home=m.homes.get(p.group);walk=min(s['speed'],s['sprint_speed'])
            direction=0.;move=0.;turn=.14;spawn=False;why='scan orchard';role='worker'
            danger=pred is not None and pred['distance']<110
            if aid==self.guide:
                role='guide';self.roles[aid]=role
                # Frame merges can move the colony frame; use its live observed home.
                if p.group!=self.group:
                    self.group=p.group;self.home=home or pos;self.last_pred=None
                pr=m.point(aid,pred) if pred else self.last_pred
                speed=15.
                if pred:
                    if self.last_pred is not None and t-self.last_seen<.15:
                        speed=min(20.,dist(pr,self.last_pred));self.still=self.still+1 if speed<.8 else 0
                    else:self.still=0
                    self.last_seen=t;self.last_pred=pr
                    if not self.acquired and speed>3 and pred['distance']<100 and abs(pred['rel_dir'])<.65:
                        self.acquired=True;self.event(t,'pursuit_inferred',aid=aid,distance=round(pred['distance'],2))
                # Only release once remote, or when energy/time requires an abort.
                remote=pr is not None and dist(pr,self.home)>self.release_radius
                if self.risk_gate and self.phase=='lead' and not self.acquired and t-self.start>4:
                    self.phase='return';self.return_point=None
                    self.event(t,'acquisition_timeout',aid=aid)
                if self.phase=='lead' and (t-self.start>self.guide_limit or s['energy']<s['max_energy']/5+45):
                    self.phase='escape';self.release=pr or pos;self.exit_angle=None
                    self.event(t,'abort_to_escape',aid=aid,energy=round(s['energy'],2),remote=remote)
                if self.phase=='lead' and self.acquired and remote and (self.variant=='distance' or self.still>=2):
                    self.phase='escape';self.release=pr;self.exit_angle=None
                    self.exclusions.append(dict(group=p.group,point=pr,until=t+60))
                    self.event(t,'release_attempt',aid=aid,point=list(pr),home=list(self.home),inferred_still=self.still,energy=round(s['energy'],2))
                if self.phase=='lead':
                    outward=bearing(self.home,pos)
                    if pred:
                        away=p.theta+pred['angle']+math.pi
                        # A guide on the colony side first arcs round the predator.
                        offset=wrap(outward-away)
                        d=pred['distance']
                        max_angle=.18 if d<70 else (.45 if d<100 else .8)
                        heading=away+max(-max_angle,min(max_angle,offset))
                        request=max(walk,min(20.,speed+.8))
                        if d<48:request=20.
                        move=min(s['sprint_speed'],request) if d<82 else (walk if d<120 else 0.)
                        turn=pred['angle'];direction=wrap(heading-p.theta)
                        if not self.acquired and d>76:
                            direction=pred['angle'];move=min(walk,d-76);why='approach to acquire predator'
                        near_edge=min((dist(edge_point(edge,pos),pos) for edge,tt in m.edges.get(p.group,[])),default=999)
                        if near_edge<100 and d<100:move=min(s['sprint_speed'],20.)
                        if self.ticks%12==0 and d>62 and self.acquired:
                            turn=direction;self.scan_ticks[aid]=self.ticks
                        if self.acquired:why='lead acquired predator away from orchard'
                    else:
                        away=bearing(pr,pos) if pr else outward
                        direction=wrap(away-p.theta);move=min(s['sprint_speed'],16.) if t-self.last_seen<.25 else (walk if t-self.last_seen<.6 else 0.)
                        turn=wrap(bearing(pos,pr)-p.theta) if pr else .22
                        why='reacquire guide contact'
                        if t-self.last_seen>1.5:
                            self.phase='return';self.return_point=None
                            self.event(t,'lost_contact',aid=aid,remote=remote)
                            if remote and self.acquired:
                                self.release=pr
                                self.exclusions.append(dict(group=p.group,point=pr,until=t+60))
                                self.event(t,'remote_disengagement',aid=aid,point=list(pr),energy=round(s['energy'],2))
                if self.phase=='escape':
                    # Travel perpendicular to the home line or outward, with local
                    # wall avoidance. Never deliberately run straight through prey.
                    if self.exit_angle is None:
                        away=bearing(self.release,pos)
                        outward=bearing(self.home,self.release)
                        side=1 if math.sin(away-outward)>=0 else -1
                        self.exit_angle=outward+side*.8
                    direction=wrap(self.exit_angle-p.theta)
                    gap=dist(pos,self.release)
                    move=walk
                    if pred and pred['distance']<70:
                        direction=wrap(pred['angle']+math.pi);move=min(s['sprint_speed'],max(walk,speed+.8))
                    turn=pred['angle'] if pred else 0.;why='break contact away from workers'
                    if self.ticks%8==0 and (not pred or pred['distance']>80):turn=direction
                    if gap>320 and (not pred or pred['distance']>180):
                        self.phase='return';self.return_point=None
                        self.event(t,'separated',aid=aid,energy=round(s['energy'],2))
                if self.phase=='return':
                    target=home or self.home
                    if self.release is not None and self.return_point is None:
                        # Tangential dogleg avoids drawing a released predator home.
                        a=bearing(self.release,target)
                        side=1 if math.sin(bearing(self.release,pos)-a)>=0 else -1
                        self.return_point=project(self.release,a+side*math.pi/2,350)
                    waypoint=self.return_point or target
                    if self.return_point is not None and dist(pos,waypoint)<50:
                        self.return_point=target;waypoint=target
                    direction=wrap(bearing(pos,waypoint)-p.theta);move=min(walk,dist(pos,waypoint));turn=max(-.3,min(.3,direction))
                    why='return via release bypass'
                    if pred and pred['distance']<150:
                        direction=wrap(pred['angle']+math.pi);move=walk if pred['distance']>65 else min(s['sprint_speed'],16.)
                        turn=pred['angle'];why='do not escort predator home'
                    if dist(pos,target)<160 or t-self.start>45:
                        self.event(t,'guide_finished',aid=aid,energy=round(s['energy'],2))
                        self.cooldown[aid]=t+8;self.guide=None;self.phase='forage'
            else:
                fruits=[]
                for o in s['observations']:
                    if o['type']!='Fruit':continue
                    q=m.point(aid,o)
                    if any(g==p.group and dist(q,c)<20 for g,c in fruit_claims):continue
                    if any(e['group']==p.group and dist(q,e['point'])<260 for e in self.exclusions):continue
                    if home is not None and dist(q,home)>260 and s['energy']>140:continue
                    fruits.append(o)
                fruits.sort(key=lambda o:o['distance'])
                if danger:
                    direction=wrap(pred['angle']+math.pi);move=walk
                    if pred['distance']<68 and s['energy']>s['max_energy']/5+7:move=min(s['sprint_speed'],16.)
                    turn=pred['angle'];why='worker escape'
                elif fruits and s['energy']<s['max_energy']-45:
                    f=fruits[0];direction=f['angle'];move=min(walk,max(0.,f['distance']-7))
                    turn=max(-.4,min(.4,direction));why='assigned visible fruit'
                    fruit_claims.append((p.group,m.point(aid,f)))
                elif home is not None:
                    # Distinct nearby trees distribute harvesting across the patch.
                    trees=[q for q,tt in m.trees.get(p.group,[]) if t-tt<25 and dist(q,home)<190]
                    trees.sort(key=lambda q:(round(q[0],1),round(q[1],1)))
                    peers=sorted(o['agent_id'] for o in states if m.poses[o['agent_id']].group==p.group and o['agent_id']!=self.guide)
                    rank=peers.index(aid) if aid in peers else 0
                    target=trees[rank%len(trees)] if trees else home
                    d=dist(pos,target)
                    if d>35:direction=wrap(bearing(pos,target)-p.theta);move=min(walk,d-35);turn=max(-.35,min(.35,direction))
                    why='work orchard sector'
                    if s['energy']<85 and not fruits:
                        direction=0.;move=walk;turn=.08;why='hungry scout seeks new fruit'
                else:
                    move=walk;direction=0.;turn=.024 if aid%2 else -.024;why='scout for orchard'
                # Keep workers away from all recently released locations.
                for exclusion in self.exclusions:
                    if exclusion['group']==p.group and dist(pos,exclusion['point'])<280:
                        direction=wrap(bearing(exclusion['point'],pos)-p.theta);move=walk;why='worker avoids release zone'
                food=bool(fruits or any(o['type']=='Tree' for o in s['observations']))
                reserve=150 if s['age']>65 or len(states)<3 else 245
                # Both arms reserve one healthy individual; optional bait cost is
                # therefore measured against an equally funded colony.
                richest=max(states,key=lambda q:q['energy'])['agent_id']
                if aid==richest and len(states)>=3 and s['age']<65:reserve=max(reserve,self.min_fuel+100)
                spawn=slots>0 and food and s['energy']>reserve and not danger
                if spawn:slots-=1
            direction=self.safe_heading(s,direction,move,pred if danger else None)
            a=ActionRequest(agent_id=aid,move_distance=round(float(move),6),move_direction=round(float(wrap(direction)),6),turn_angle=round(float(wrap(turn)),6),spawn_agent=spawn)
            actions.append((aid,a));self.decisions[aid]=why
        for aid,a in actions:m.advance(byid[aid],a)
        return actions
