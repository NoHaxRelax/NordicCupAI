"""Wall funnel controller accepting ONLY native observation DTOs and public time.

No engine import, environment reference, world coordinates, geometry argument,
predator identity/energy/rest flag, scenario seed, or evaluator feedback.
Reuses the handoff's observation-derived registration and landmark odometry.
"""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wall_deployment'))
from controller import WallPolicy, add, sub, mul, dot, unit, norm, rot, wrap


class ObservedFunnel(WallPolicy):
    def __init__(self, capacity=33, gate=True, replenish=False, gather=False):
        super().__init__()
        self.capacity = capacity
        self.gate = gate
        self.replenish = replenish
        self.gather = gather
        self.stations = {}
        self.workers = {}

    def identify(self, group, states):
        """Infer two parallel faces from observed complete edge endpoints."""
        live = sorted(group.members & states.keys())
        if len(live)<2:
            return None
        holder = live[0]
        hp = self.poses[holder].p
        for edge in group.edges:
            length = math.dist(*edge)
            if not 69.9 <= length <= 100.1:
                continue
            tangent = unit(sub(edge[1],edge[0]))
            midpoint = mul(add(*edge),.5)
            normal = (-tangent[1],tangent[0])
            if dot(sub(hp,midpoint),normal)<0:
                normal=mul(normal,-1)
            if not 4.9 <= dot(sub(hp,midpoint),normal) <= 12:
                continue
            for other in group.edges:
                if abs(math.dist(*other)-length)>.1:
                    continue
                om = mul(add(*other),.5)
                delta = sub(midpoint,om)
                width = dot(delta,normal)
                if not 29.9 <= width <= 35.1 or abs(dot(delta,tangent))>.1:
                    continue
                return dict(holder=holder,anchor=hp,front=add(om,mul(normal,-5.1)),
                    normal=normal,tangent=tangent,length=length,width=width,
                    seen=0,previous=[],still_since=None,gate_until=-1.,mapped=self.time)
        return None

    def act(self, observations, sim_time):
        self.time=sim_time
        states={s['agent_id']:s for s in observations}
        if not self.poses and all(not s['observations'] for s in observations):
            return [dict(agent_id=aid,move_distance=0.,move_direction=0.,turn_angle=0.,
                         spawn_agent=False) for aid in sorted(states)]
        for aid in sorted(states.keys() & self.poses.keys()):
            self._map(aid,states[aid])
        self._register(states)
        for aid in sorted(states):
            self._map(aid,states[aid])
        actions=[]
        self.decisions={}
        for group in self.groups:
            live=sorted(group.members & states.keys())
            if not live:
                continue
            key=id(group)
            station=self.stations.get(key)
            if station is None:
                station=self.identify(group,states)
                if station:
                    self.stations[key]=station
                    self._event('mapped_wall_from_edges',width=station['width'],
                                length=station['length'],holder=station['holder'])
            if station and station['holder'] in states:
                holder=station['holder']
                hp=self.poses[holder]
                points=[hp.polar(o) for o in states[holder]['observations']
                    if o['type']=='Predator' and
                    dot(sub(hp.polar(o),station['anchor']),station['normal']) < -20]
                station['seen']=max(station['seen'],len(points))
                # Do not pretend occluded/missing predators are resting. Require
                # all previously counted targets to be present and stationary.
                old=station['previous']
                stationary=bool(points) and len(points)==len(old)==station['seen']
                unmatched=list(old)
                for p in points:
                    if not unmatched:
                        stationary=False;break
                    q=min(unmatched,key=lambda q:math.dist(p,q))
                    if math.dist(p,q)>1e-5:
                        stationary=False
                    unmatched.remove(q)
                if stationary:
                    if station['still_since'] is None:
                        station['still_since']=sim_time
                    if .2 <= sim_time-station['still_since'] < .4:
                        station['gate_until']=sim_time+1.8
                else:
                    station['still_since']=None
                station['previous']=points
            for aid in live:
                state=states[aid];pose=self.poses[aid]
                target=pose.p;turn=0.;speed=state['speed'];spawn=False
                rule='scan_for_wall'
                if not station:
                    turn=.35
                elif aid==station['holder']:
                    rule='hold_observed_wall'
                    target=station['anchor']
                    # Birth only when there is no living collector. Native birth
                    # placement/traits/energy are never chosen by the policy.
                    spawn=self.replenish and len(live)==1 and station['seen']<self.capacity and state['energy']>200
                else:
                    worker=self.workers.setdefault(aid,{'phase':'depart','path':None})
                    n,t=station['normal'],station['tangent']
                    front=station['front']
                    standby=add(front,mul(n,-118))
                    if worker['path'] is None:
                        if dot(sub(pose.p,front),n)>10:
                            outer=station['length']/2+100
                            # Stay well outside a held predator's 60-unit hearing
                            # circle while routing newborns around the wall end.
                            worker['path']=[add(station['anchor'],mul(n,85)),
                                add(add(station['anchor'],mul(n,85)),mul(t,outer)),
                                add(standby,mul(t,outer)),standby]
                        else:
                            worker['path']=[standby]
                    if worker['phase']=='depart':
                        nearby=[o for o in state['observations'] if o['type']=='Predator']
                        if (nearby and min(o['distance'] for o in nearby)<55 and
                                dot(sub(pose.p,front),n)<0):
                            worker['phase']='collect'
                            worker['path']=[]
                        while worker['path'] and math.dist(pose.p,worker['path'][0])<2:
                            worker['path'].pop(0)
                        if worker['path']:
                            target=worker['path'][0];speed=state['sprint_speed']
                            rule='route_to_collection_side'
                        else:
                            worker['phase']='collect'
                    if worker['phase']=='collect':
                        threats=[o for o in state['observations'] if o['type']=='Predator']
                        if threats:
                            nearest=min(threats,key=lambda o:o['distance'])
                            p=pose.polar(nearest)
                            can_deliver=(not self.gate or station['seen']==0 or
                                         sim_time<station['gate_until'])
                            if self.gather and station['seen']==0:
                                if nearest['distance']<60:
                                    points=[pose.polar(o) for o in threats]
                                    center=mul(tuple(map(sum,zip(*points))),1/len(points))
                                    v=sub(pose.p,center)
                                    worker.update(phase='gather',center=center,
                                        orbit_angle=math.atan2(v[1],v[0]),gather_started=sim_time)
                                target=pose.p;rule='wait_for_pack_to_approach'
                            elif can_deliver:
                                worker['phase']='deliver'
                            else:
                                # Orbit a staging point outside the old trap's
                                # target-switch zone using only local bearings.
                                away=unit(sub(pose.p,p))
                                tangent=(-away[1],away[0])
                                target=add(pose.p,mul(unit(add(tangent,mul(away,.6))),20))
                                speed=state['sprint_speed'];rule='wait_for_observed_stillness'
                            turn=wrap(math.atan2(p[1]-pose.p[1],p[0]-pose.p[0])-pose.theta+.02)
                        else:
                            target=standby;turn=.4;rule='scan_collection_area'
                    if worker['phase']=='gather':
                        center=worker['center']
                        angle=worker['orbit_angle']
                        goal=add(center,(80*math.cos(angle),80*math.sin(angle)))
                        if math.dist(pose.p,goal)<12:
                            worker['orbit_angle']=angle+.23
                            goal=add(center,(80*math.cos(angle+.23),80*math.sin(angle+.23)))
                        target=goal;speed=state['sprint_speed'];rule='gather_visible_pack_on_circle'
                        threats=[o for o in state['observations'] if o['type']=='Predator']
                        points=[pose.polar(o) for o in threats]
                        if points:
                            mean=mul(tuple(map(sum,zip(*points))),1/len(points))
                            v=sub(mean,pose.p)
                            turn=wrap(math.atan2(v[1],v[0])-pose.theta)
                            compact=max(math.dist(p,mean) for p in points)<6
                            behind=all(dot(sub(pose.p,p),n)>25 for p in points)
                            aligned=abs(dot(sub(pose.p,front),t))<20
                            if (len(points)>=self.capacity and compact and behind and aligned
                                    and sim_time-worker['gather_started']>5):
                                worker['phase']='deliver'
                                self._event('pack_gathered_from_observations',count=len(points),
                                    radius=max(math.dist(p,mean) for p in points),guide=aid)
                        else:
                            v=sub(center,pose.p)
                            turn=wrap(math.atan2(v[1],v[0])-pose.theta)
                    if worker['phase']=='deliver':
                        target=front;rule='deliver_to_observed_front_face'
                        threats=[o for o in state['observations'] if o['type']=='Predator']
                        if threats:
                            nearest=min(threats,key=lambda o:o['distance'])
                            if nearest['distance']>75:
                                target=pose.p
                            if nearest['distance']<45:
                                speed=state['sprint_speed']
                            turn=wrap(nearest['angle']+math.pi)
                delta=sub(target,pose.p)
                modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state['biome']]
                distance=min(speed,norm(delta)/modifier)
                direction=wrap(math.atan2(delta[1],delta[0])-pose.theta) if norm(delta)>.01 else 0.
                action=dict(agent_id=aid,move_distance=distance,move_direction=direction,
                            turn_angle=turn,spawn_agent=spawn)
                actions.append(action)
                self.decisions[aid]={'rule':rule}
                pose.p=add(pose.p,rot((distance*modifier,0),pose.theta+direction))
                pose.theta=wrap(pose.theta+turn)
        return actions
