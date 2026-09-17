"""Scout/holder/bodyguard prototype using only public observation DTOs.

Each connected map component prepares at most one trap and reserves it for one
delivery. Disconnected maps are aligned on observed agent encounters. No global
coordinates, hidden predator identities/rest state, or setup hints are inputs.
"""
import math
from controller import WallPolicy as ColonyPolicy, Trap, add, sub, mul, dot, norm, unit, rot, wrap, point_segment


class WallPolicy(ColonyPolicy):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.active_cap=4
        self.catalog={};self.guards={};self.guides={};self.escorts={};self.scout_targets={}
        self.discovery_sizes={}
        self.metrics.update(discovered_sites=0,map_merges=0,prepared=0,ready=0,
                            guide_started=0,delivery_unverified=0,guard_responses=0)

    def _register(self,states):
        super()._register(states)
        # A public Agent observation gives both relative position and heading.
        # Transform the entire source map, including remembered food and roles.
        for aid in sorted(states):
            for o in states[aid]['observations']:
                other=o.get('id')
                if o['type']!='Agent' or other not in states or o['distance']<1e-6:continue
                dst,src=self.group_for[aid],self.group_for[other]
                if dst is src:continue
                if min(dst.members)>min(src.members):continue
                if dst.trap and src.trap:continue  # preserve two occupied stations
                p=self.poses[aid].polar(o)
                theta=wrap(self.poses[aid].theta+o['angle']+math.pi-o['rel_dir'])
                angle=wrap(theta-self.poses[other].theta)
                offset=sub(p,rot(self.poses[other].p,angle))
                transform=lambda q:add(offset,rot(q,angle))
                for member in src.members:
                    self.poses[member].p=transform(self.poses[member].p)
                    self.poses[member].theta=wrap(self.poses[member].theta+angle)
                    self.group_for[member]=dst
                dst.edges.extend((transform(a),transform(b)) for a,b in src.edges)
                for f in src.food:f.p=transform(f.p)
                dst.food.extend(src.food);dst.members|=src.members;dst.retired|=src.retired
                for site in self.catalog.pop(id(src),[]):
                    for key in ['anchor','front','center']:site[key]=transform(site[key])
                    site['normal']=rot(site['normal'],angle)
                    site['edge']=tuple(transform(p) for p in site['edge'])
                    self.catalog.setdefault(id(dst),[]).append(site)
                if src.trap:
                    t=src.trap;t.anchor=transform(t.anchor);t.normal=rot(t.normal,angle)
                    t.edge=tuple(transform(p) for p in t.edge);dst.trap=t
                    for mapping in [self.guards,self.guides]:
                        if id(src) in mapping:mapping[id(dst)]=mapping.pop(id(src))
                dst.predator_tracks=[];dst.waypoints={};dst.targets={}
                self.groups.remove(src);self.metrics['map_merges']+=1
                self._event('map_merge',observer=aid,other=other,members=sorted(dst.members))

    def _discover(self,group):
        if self.discovery_sizes.get(id(group))==len(group.edges):return
        self.discovery_sizes[id(group)]=len(group.edges)
        sites=self.catalog.setdefault(id(group),[])
        for a,b in list(group.edges):
            length=math.dist(a,b)
            if not 69.9<=length<=100.1:continue
            u=unit(sub(b,a))
            for c,d in list(group.edges):
                for corner,far in [(c,d),(d,c)]:
                    if min(math.dist(corner,a),math.dist(corner,b))>.15:continue
                    short=sub(far,corner);thickness=norm(short)
                    if not 29.9<=thickness<=35.05 or abs(dot(unit(short),u))>.001:continue
                    center=add(mul(add(a,b),.5),mul(short,.5))
                    if any(math.dist(center,s['center'])<1 for s in sites):continue
                    inward=unit(short)
                    # Full rectangle inferred from two perpendicular full edges.
                    opposite=(add(a,short),add(b,short))
                    for edge in [opposite,(a,add(a,short)),(b,add(b,short))]:
                        if not any(min(math.dist(edge[0],x)+math.dist(edge[1],y),
                            math.dist(edge[0],y)+math.dist(edge[1],x))<.1 for x,y in group.edges):group.edges.append(edge)
                    for side in [-1,1]:
                        normal=mul(inward,side)
                        anchor=add(center,mul(normal,thickness/2+7))
                        front=add(center,mul(normal,-(thickness/2+7)))
                        guard=add(anchor,mul(normal,45))
                        if any(point_segment(p,*edge)<5.1 for p in [anchor,front,guard] for edge in group.edges):continue
                        face=(a,b) if side==-1 else opposite
                        sites.append(dict(center=center,anchor=anchor,front=front,normal=normal,
                            edge=face,thickness=thickness,length=length,found=self.time))
                        self.metrics['discovered_sites']+=1
                        self._event('site_discovered',anchor=anchor,thickness=round(thickness,2),length=round(length,2))

    def _prepare(self,group,states):
        if group.trap or not self.wall or self.time<group.wall_cooldown:return
        live=group.members&states.keys()-group.retired
        if len(live)<3:return
        ranked=[]
        for site in self.catalog.get(id(group),[]):
            near=sorted((a for a in live if states[a]['energy']>95),key=lambda a:math.dist(self.poses[a].p,site['anchor']))
            if len(near)<2:continue
            food=sum(f.kind=='Tree' and math.dist(f.p,site['anchor'])<280 and
                dot(sub(f.p,site['anchor']),site['normal'])>20 for f in group.food)
            distance=sum(math.dist(self.poses[a].p,site['anchor']) for a in near[:2])
            if not food or distance>650:continue
            ranked.append((distance-60*food,site,near))
        if not ranked:return
        _,site,near=min(ranked,key=lambda r:r[0])
        group.trap=Trap(site['anchor'],site['normal'],site['edge'],near[0],self.time,stage='prepare')
        group.trap.front=site['front'];group.trap.thickness=site['thickness']
        self.guards[id(group)]=near[1];self.metrics['prepared']+=1
        self._event('prepare_trap',holder=near[0],guard=near[1],anchor=site['anchor'])

    def _scout(self,group,aid,state):
        p=self.poses[aid].p
        if state['energy']<115:return super()._forage(group,aid,state)
        key=(id(group),aid);target,at=self.scout_targets.get(key,(p,-100))
        if math.dist(p,target)<12 or self.time-at>7:
            angle=aid*2.399963+int(self.time/7)*1.13
            target=add(p,mul((math.cos(angle),math.sin(angle)),230));at=self.time
            self.scout_targets[key]=(target,at)
        return target,'scout_walls'

    def act(self,observations,sim_time):
        if not self.wall:return super().act(observations,sim_time)
        self.time=sim_time;states={s['agent_id']:s for s in observations}
        if sim_time==0 and all(not s['observations'] for s in states.values()):
            return [dict(agent_id=a,move_distance=0.,move_direction=0.,turn_angle=0.,spawn_agent=False) for a in sorted(states)]
        for a in sorted(states.keys()&self.poses.keys()):self._map(a,states[a])
        self._register(states)
        for a in sorted(states):self._map(a,states[a])
        self.decisions={};actions=[];global_births=0
        for group in self.groups:
            live=group.members&states.keys()
            if not live:continue
            predators=self._predators(group,states);self._discover(group);self._prepare(group,states)
            trap=group.trap;gid=id(group);guard=self.guards.get(gid);guide=self.guides.get(gid)
            if trap:
                if trap.holder not in live:
                    self._event('holder_lost',holder=trap.holder);group.trap=None;trap=None;group.wall_cooldown=self.time+30
                else:
                    if guard not in live:
                        choices=list(live-{trap.holder,guide}-group.retired)
                        guard=min(choices,key=lambda a:math.dist(self.poses[a].p,trap.anchor)) if choices else None
                        self.guards[gid]=guard
                    station=add(trap.anchor,mul(trap.normal,45))
                    if trap.stage=='prepare' and guard is not None and math.dist(self.poses[trap.holder].p,trap.anchor)<3 and math.dist(self.poses[guard].p,station)<5:
                        trap.stage='ready';self.metrics['ready']+=1;self._event('trap_ready',holder=trap.holder,guard=guard)
                    if trap.stage=='prepare' and self.time-trap.created>20:
                        self._event('preparation_timeout');group.trap=None;trap=None;group.wall_cooldown=self.time+15
            if trap and trap.stage=='ready' and guard in live:
                # Reserve one guide only. Side/route gating prevents an agent
                # from bringing a predator through the protected camp.
                for aid in sorted(live-{trap.holder,guard}-group.retired):
                    p=self.poses[aid].p
                    seen=[self.poses[aid].polar(o) for o in states[aid]['observations'] if o['type']=='Predator']
                    if dot(sub(p,trap.front),trap.normal)>-10 or math.dist(p,trap.front)>200:continue
                    if self._blocked(group,p,trap.front):continue
                    choices=[q for q in seen if math.dist(p,q)<145 and dot(sub(trap.front,p),sub(p,q))>0]
                    if choices:
                        guide=aid;self.guides[gid]=aid;trap.stage='reserved';self.metrics['guide_started']+=1
                        self._event('guide_started',agent=aid,holder=trap.holder);break
            if trap and guide is not None and guide not in live and trap.stage=='reserved':
                trap.stage='occupied_unverified';self.metrics['delivery_unverified']+=1
                self._event('delivery_unverified',guide=guide)
            group.fruit_claims=set();births=0
            for aid in sorted(live):
                state=states[aid];pose=self.poses[aid];p=pose.p;look=None;sprint=False
                target,rule=super()._forage(group,aid,state)
                if aid==min(live) and not trap:target,rule=self._scout(group,aid,state)
                if trap and aid==trap.holder:target,rule=trap.anchor,'hold' if trap.stage!='prepare' else 'position_holder'
                if trap and aid==guard:
                    target,rule=add(trap.anchor,mul(trap.normal,45)),'bodyguard_station'
                    threats=[q for q in predators if dot(sub(q,trap.anchor),trap.normal)>-10 and math.dist(q,trap.anchor)<240]
                    if threats or aid in self.escorts:
                        if aid not in self.escorts:
                            self.escorts[aid]=self.time;self.metrics['guard_responses']+=1;self._event('bodyguard_escort',agent=aid)
                        tangent=(-trap.normal[1],trap.normal[0]);target=add(p,mul(tangent,30));rule='bodyguard_escort'
                        if threats:
                            q=min(threats,key=lambda q:math.dist(p,q));look=q
                            sprint=math.dist(p,q)<65 and state['energy']>state['max_energy']/5+8
                        if self.time-self.escorts[aid]>20 and not threats:
                            self.escorts.pop(aid);target=add(trap.anchor,mul(trap.normal,45))
                if trap and aid==guide and trap.stage=='reserved':
                    target,rule=trap.front,'guide_to_trap'
                    seen=[pose.polar(o) for o in state['observations'] if o['type']=='Predator']
                    if seen:
                        q=min(seen,key=lambda q:math.dist(p,q));look=q;gap=math.dist(p,q)
                        if gap>75 and math.dist(p,trap.front)>10:target=p
                        sprint=gap<45 and state['energy']>state['max_energy']/5+8
                    if math.dist(p,trap.front)<3:rule='sacrifice_at_front'
                threats=[q for q in predators if math.dist(p,q)<90 and not
                    (trap and dot(sub(q,trap.anchor),trap.normal)<-18)]
                special=rule in ['guide_to_trap','sacrifice_at_front','bodyguard_escort']
                if threats and not special:
                    q=min(threats,key=lambda q:math.dist(p,q));look=q
                    target=add(p,mul(unit(sub(p,q)),30));rule='emergency_escape'
                    sprint=math.dist(p,q)<65 and state['energy']>state['max_energy']/5+8
                waypoint=self._waypoint(group,aid,target);delta=sub(waypoint,p)
                modifier=dict(forest=1.,grassland=1.,swamp=.5,desert=.8,river=.3)[state['biome']]
                speed=state['sprint_speed'] if sprint else min(state['speed'],state['sprint_speed'])
                distance=min(speed,norm(delta)/modifier)
                direction=wrap(math.atan2(delta[1],delta[0])-pose.theta) if norm(delta)>.01 else 0
                turn=direction if distance>.1 else (.7 if int(round(sim_time*10))%10==0 else 0.)
                if look is not None:turn=wrap(math.atan2(look[1]-p[1],look[0]-p[0])+.02-pose.theta)
                spawn=(len(live)+births<self.active_cap and len(states)+global_births<18 and state['age']>=25
                    and state['energy']>175 and not threats and aid not in [guide,guard])
                if spawn:births+=1;global_births+=1
                raw=dict(agent_id=aid,move_distance=distance,move_direction=direction,turn_angle=turn,spawn_agent=spawn)
                actions.append(raw);self.decisions[aid]=dict(rule=rule,detail='Scout v1; public DTOs only, one reserved predator per station.')
                pose.p=add(p,rot((distance*modifier,0),pose.theta+direction));pose.theta=wrap(pose.theta+turn)
        return sorted(actions,key=lambda a:a['agent_id'])
