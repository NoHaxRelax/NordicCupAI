"""Oscar's Orchard plus shared-map corner steering, with no bait or trapping."""
import json
import math
from pathlib import Path
import numpy as np
from models.exploration.world_estimator import WorldEstimator, rotate
from models.exploration.global_planner import load_planner_config
from models.survival.oscar_orchard import OrchardPolicy
from models.entrapment.bystander_avoidance import avoid_predators
from models.avoidance.corner_steering import steer


def local(pose, point): return tuple(map(float,rotate(np.asarray(point)-pose.position,-pose.heading)))


class OrchardAvoidancePolicy:
    def __init__(self, seed=0, *, corner_mode='nearest', survival_settings=None):
        self.strict=corner_mode.endswith('-strict')
        self.mode=corner_mode.removesuffix('-strict')
        settings=json.loads((Path(__file__).with_name('oscar_latest.json')).read_text())
        settings.update(survival_settings or {})
        self.orchard=OrchardPolicy(seed=seed,**settings)
        self.estimator=WorldEstimator(load_planner_config().estimator)
        self.memories={}; self.corners={}; self.roles={}; self.events=[]; self.debug={}
        self.metrics=dict(steering_attempts=0,steering_releases=0,aligned_exits=0,misaligned_exits=0,aborted_attempts=0)
        # Recorder compatibility. None of the trap machinery is instantiated.
        self.site=self.bait=self.incoming=None; self.retired_baits=set(); self.tracks={}

    def _arrival(self, aid): return False

    def choose_corner(self,pose,predator):
        group=self.estimator.groups[pose.group_id]
        size=group.world_size or (group.known_width,group.known_height)
        if not group.anchored or not all(size): return None
        w,h=size
        corners=[(40.,40.),(w-40.,40.),(40.,h-40.),(w-40.,h-40.)]
        if self.mode=='nearest': return min(corners,key=lambda c:math.dist(c,predator))
        key=(pose.group_id,group.frame_revision)
        if key not in self.corners:
            # Observed tree density is an output proxy, not knowledge of unseen fruit.
            counts=[sum(math.dist(t.position,c)<min(w,h)*.4 for t in group.trees) for c in corners]
            self.corners[key]=corners[min(range(4),key=lambda i:counts[i])]
        return self.corners[key]

    def __call__(self,states_list,sim_time):
        self.estimator.update(states_list,sim_time)
        heir={a:m.heir_done for a,m in self.orchard.minds.items()}
        base=dict(self.orchard(states_list,sim_time))
        observations={}
        for s in states_list:
            pose=self.estimator.poses[s['agent_id']]
            if pose.uncertainty>8.: continue
            pool=observations.setdefault(pose.group_id,[])
            for o in s['observations']:
                if o['type']!='Predator': continue
                p=pose.position+rotate((o['distance']*math.cos(o['angle']),o['distance']*math.sin(o['angle'])),pose.heading)
                h=pose.heading+o['angle']+math.pi-o.get('rel_dir',0.)
                if not any(math.dist(p,old[0])<18. for old in pool): pool.append((p,h))
        # Only the nearest currently detectable observer steers each sighting.
        # Other agents escape normally rather than pulling in opposite directions.
        owners=set()
        for gid,pool in observations.items():
            for p,h in pool:
                candidates=[]
                for state in states_list:
                    aid=state['agent_id'];pose=self.estimator.poses[aid]
                    if pose.group_id!=gid or pose.uncertainty>8.:continue
                    d=math.dist(p,pose.position)
                    bearing=math.atan2(pose.position[1]-p[1],pose.position[0]-p[0])-h
                    if d<=60. or (d<=250. and abs(math.atan2(math.sin(bearing),math.cos(bearing)))<math.pi/6):
                        candidates.append((d,aid))
                if candidates:owners.add(min(candidates)[1])
        actions=[];self.roles={};self.debug={}
        self.memories={a:m for a,m in self.memories.items() if a in base}
        for s in states_list:
            aid=s['agent_id'];pose=self.estimator.poses[aid];a=base[aid]
            sightings=[o for o in s['observations'] if o['type']=='Predator'];shared=[]
            if pose.uncertainty<=8.:
                for p,h in observations.get(pose.group_id,[]):
                    q=local(pose,p);d=math.hypot(*q)
                    if d>300. or any(abs(d-o['distance'])<18. and abs(math.atan2(math.sin(math.atan2(q[1],q[0])-o['angle']),math.cos(math.atan2(q[1],q[0])-o['angle'])))<.15 for o in sightings):continue
                    shared.append(dict(type='Predator',distance=d,angle=math.atan2(q[1],q[0]),rel_dir=math.atan2(pose.position[1]-p[1],pose.position[0]-p[0])-h))
            a,avoiding=avoid_predators(a,s,shared_predators=shared)
            role='avoiding_predator' if avoiding else 'gatherer'
            memory=self.memories.get(aid)
            if memory and memory.get('phase')=='done':
                if sim_time-memory['started']>8.: self.memories.pop(aid);memory=None
            if self.mode!='off' and sightings and (aid in owners or memory is not None) and (memory is None or memory.get('phase')!='done'):
                o=min(sightings,key=lambda o:o['distance'])
                p=pose.position+rotate((o['distance']*math.cos(o['angle']),o['distance']*math.sin(o['angle'])),pose.heading)
                c=self.choose_corner(pose,p) if pose.uncertainty<=8. else None
                if c is not None:
                    desired=math.atan2(c[1]-p[1],c[0]-p[0])-pose.heading
                    memory=self.memories.setdefault(aid,{})
                    old_phase=memory.get('phase')
                    changed,debug=steer(base[aid],s,sightings,desired,memory,sim_time,strict=self.strict,tolerance=math.radians(10 if self.strict else 20))
                    if debug.get('phase')=='aborted':
                        self.metrics['aborted_attempts']+=1;self.events.append(dict(time=sim_time,kind='steering_aborted',agent=aid))
                    if changed is not None:
                        a=changed;role='steering_predator';self.debug[aid]=dict(debug,corner=list(c))
                        if old_phase is None:
                            self.metrics['steering_attempts']+=1;self.events.append(dict(time=sim_time,kind='steering_started',agent=aid))
                        if memory.get('phase')=='done':
                            self.metrics['steering_releases']+=1
                            self.metrics['aligned_exits' if debug['aligned_exit'] else 'misaligned_exits']+=1
                            self.events.append(dict(time=sim_time,kind='aligned_exit' if debug['aligned_exit'] else 'misaligned_exit',agent=aid,heading_error_degrees=math.degrees(debug['heading_error'])))
            elif not sightings and memory: memory['phase']='done'
            self.roles[aid]=role
            m=self.orchard.minds[aid]
            if base[aid].spawn_agent and not a.spawn_agent:m.heir_done=heir.get(aid,False)
            m.last_action=(a.move_distance,a.move_direction,a.turn_angle,s['biome'],s['energy'],s['speed'],s['sprint_speed'],s['max_energy'])
            cost=.05*min(a.move_distance,s['speed'])+.5*max(0.,a.move_distance-s['speed'])+abs(a.turn_angle)/math.tau
            m.spawned_ok=a.spawn_agent and s['energy']-cost>100.
            actions.append((aid,a))
        self.orchard.last_spawners=[aid for aid,a in actions if self.orchard.minds[aid].spawned_ok]
        self.estimator.remember_actions([a for _,a in actions])
        return actions

    def snapshot(self):
        return dict(phase='orchard_corner_'+self.mode+('_strict10' if self.strict else ''),site=None,bait=None,incoming=None,
            roles=self.roles,metrics=self.metrics,map={gid:dict(anchored=g.anchored,size=g.world_size or (g.known_width,g.known_height),trees=len(g.trees)) for gid,g in self.estimator.groups.items()},steering=self.debug,
            guides={aid:dict(debug=dict(forecast=d)) for aid,d in self.debug.items()},
            estimated_agents={aid:dict(position=p.position.tolist(),heading=p.heading,uncertainty=p.uncertainty,group=p.group_id) for aid,p in self.estimator.poses.items()})
