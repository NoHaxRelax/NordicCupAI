"""Fruit timing economics and controlled observation-based patch policies.

No generated-map claims: localization is integrated from actions on the arranged
flat clear patch and checked against the engine diagnostically. The oracle is
explicitly separate. Fruit tracking itself consumes the public observation data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from sprinter_decoys import fixture, add_agent, observe, wrap, ROOT
from src.elements.environment import Environment
from src.utils.DTOs import ActionRequest


def key(x,y):return (round(x,2),round(y,2))


def visible(point,region):
    x,y,theta,hear,vision,cone=region
    dx,dy=point[0]-x,point[1]-y;d=math.hypot(dx,dy)
    return d<=hear or (d<=vision and abs(wrap(math.atan2(dy,dx)-theta))<=cone/2)


class FruitPolicy:
    def __init__(self,mode,*,reproduce=False,scan=False,patch_radius=250):
        self.mode=mode;self.reproduce=reproduce;self.scan=scan
        self.patch_radius=patch_radius
        self.pose={0:(0.,0.,0.)};self.tracks={};self.previous_regions=[]
        self.known_new=0;self.unknown_new=0;self.wait_ticks=0;self.emergency_picks=0
        self.unlocalized_ticks=0

    def localize_children(self,states):
        # A nearby Agent observation supplies its ID, relative pose and heading.
        for _ in range(3):
            changed=False
            for s in states:
                if s['agent_id'] not in self.pose:continue
                x,y,theta=self.pose[s['agent_id']]
                for o in s['observations']:
                    if o['type']=='Agent' and o['id'] not in self.pose:
                        bearing=theta+o['angle']
                        self.pose[o['id']]=(x+o['distance']*math.cos(bearing),
                            y+o['distance']*math.sin(bearing),bearing+math.pi-o['rel_dir'])
                        changed=True
            if not changed:break

    def __call__(self,states,t,oracle_ages=None):
        self.localize_children(states)
        seen={};regions=[]
        for s in states:
            aid=s['agent_id']
            if aid not in self.pose:continue
            x,y,theta=self.pose[aid]
            regions.append((x,y,theta,s['hearing_radius'],s['vision_range'],s['vision_angle']))
            for o in s['observations']:
                if o['type']=='Fruit':
                    fx=x+o['distance']*math.cos(theta+o['angle'])
                    fy=y+o['distance']*math.sin(theta+o['angle'])
                    seen[key(fx,fy)]=(fx,fy)
        # Clear disappeared fruit only when its location is currently observable.
        for k in list(self.tracks):
            if k not in seen and any(visible(self.tracks[k]['point'],r) for r in regions):del self.tracks[k]
        for k,point in seen.items():
            if k not in self.tracks:
                known=any(visible(point,r) for r in self.previous_regions)
                self.tracks[k]=dict(point=point,first_seen=t,known_new=known)
                self.known_new+=known;self.unknown_new+=not known
        self.previous_regions=regions
        actions=[];reserved=set();slots=max(0,3-len(states))
        for s in states:
            aid=s['agent_id'];move=direction=turn=0.;spawn=False
            if aid not in self.pose:
                self.unlocalized_ticks+=1
                actions.append((aid,ActionRequest(agent_id=aid,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)))
                continue
            x,y,theta=self.pose[aid]
            choices=[]
            for k,point in seen.items():
                if k in reserved:continue
                # A local operating radius uses only integrated relative pose.
                # In this arranged clear patch it avoids boundary collisions.
                if math.hypot(*point)>self.patch_radius:continue
                d=math.dist((x,y),point)
                track=self.tracks[k]
                if self.mode=='immediate':wait=0.
                elif self.mode=='oracle':wait=max(0.,20-(oracle_ages or {}).get(k,20))
                elif self.mode=='blind':wait=max(0.,20-(t-track['first_seen']))
                else:wait=max(0.,20-(t-track['first_seen'])) if track['known_new'] else 0.
                # Only wait if a conservative survival budget covers it.
                # Age threshold is hidden, so assume surcharge after age60.
                living_rate=1+(s['age']*.1 if s['age']>60 else 0)
                emergency=s['energy']<living_rate*wait+d*.05+25
                if emergency and wait:
                    wait=0.;self.emergency_picks+=1
                choices.append((wait+d/100,k,point,wait,d))
            if choices:
                _,k,point,wait,d=min(choices)
                reserved.add(k)
                direction=wrap(math.atan2(point[1]-y,point[0]-x)-theta)
                target_gap=15.5 if wait>.001 else 8.
                move=min(s['speed'],s['sprint_speed'],max(0.,d-target_gap))
                turn=direction
                if wait>.001:
                    self.wait_ticks+=1
                    if move==0 and self.scan:turn=.1
            else:
                # Hold the local patch and inspect it; no unobserved world targets.
                turn=.1 if self.scan else 0.
            if self.reproduce and slots and seen:
                reserve=125 if s['age']>65 and len(states)<2 else 225
                spawn=s['energy']>reserve
                if spawn:slots-=1
            self.pose[aid]=(x+move*math.cos(theta+direction),y+move*math.sin(theta+direction),theta+turn)
            actions.append((aid,ActionRequest(agent_id=aid,move_distance=move,
                move_direction=direction,turn_angle=turn,spawn_agent=spawn)))
        return actions


def run(*,mode='observed',seed=0,patch=False,reproduce=False,energy=150,
        initial_fruit_age=0,appearance='initial',agent_age=0,seconds=60,tree_births=True):
    e=fixture(seed)
    a=add_agent(e,800,600,energy);a.direction=0;a.age=agent_age
    anchor=(800,600)
    if patch:
        # Source-native trees age, die, reproduce fruit, and optionally new trees
        # spawn globally. Flat forest acceptance is intentionally favourable.
        e.spawn_tree=Environment.spawn_tree.__get__(e)
        for dx,dy in ((-45,0),(45,0),(0,70)):
            tree=e.spawn_tree(x=800+dx,y=600+dy)
            if tree:tree.grow(20)
        if not tree_births:e.spawn_tree=lambda *args,**kwargs:None
    def create_fruit():
        f=e.spawn_fruit(x=830,y=600)
        for _ in range(round(initial_fruit_age*10)):f.grow(.2)
    if not patch and appearance=='initial':create_fruit()
    policy=FruitPolicy(mode,reproduce=reproduce,scan=patch)
    states=observe(e)
    movement_energy=turn_energy=0.;eaten=[];rotted=0;received=0.;max_pose_error=0.
    deaths=[];born=1;peak=1;trace=[];wait_state=None
    for tick in range(round(seconds*10)):
        if not patch and appearance=='new' and tick==10:create_fruit()
        before={obj.agent_id:obj for obj in e.agents}
        # Hidden ages are supplied ONLY to the explicitly labelled oracle.
        oracle={key(f.x-anchor[0],f.y-anchor[1]):f.age/2 for f in e.fruits} if mode=='oracle' else None
        actions=policy(states,e.time,oracle)
        for aid,action in actions:
            old=before[aid].energy
            e.agent_step(aid,action.move_distance,action.move_direction,action.turn_angle,action.spawn_agent)
            tc=min(math.pi,abs(action.turn_angle))/(2*math.pi)
            birth_charge=100 if action.spawn_agent and e._next_agent_id>born else 0
            movement_energy+=old-before[aid].energy-tc-birth_charge;turn_energy+=tc
            born=e._next_agent_id
        fruits_before=list(e.fruits)
        energy_before={obj.agent_id:obj.energy for obj in e.agents}
        age_before={obj.agent_id:obj.age for obj in e.agents}
        expected_passive={obj.agent_id:.1+(.01*(obj.age+.1) if obj.age+.1>obj.max_age and obj.energy>.1 else 0) for obj in e.agents}
        e.non_agent_step(.1)
        for obj in list(before.values())+ [obj for obj in e.agents if obj.agent_id not in before]:
            if obj.agent_id in energy_before:
                # Source removal from the agent list can skip the next member;
                # do not misclassify its uncharged passive cost as food credit.
                passive=expected_passive[obj.agent_id] if obj.age>age_before[obj.agent_id] else 0.
                received+=max(0,obj.energy-energy_before[obj.agent_id]+passive)
        for f in fruits_before:
            if f not in e.fruits:
                # Agents eat before the rot pass, even on the last over-age
                # tick. Diagnose a pickup from contact with a processed agent.
                picked=any(obj.agent_id in age_before and obj.age>age_before[obj.agent_id]
                    and energy_before[obj.agent_id]>.1
                    and math.hypot(obj.x-f.x,obj.y-f.y)<obj.size+f.radius
                    for obj in list(before.values())+[obj for obj in e.agents if obj.agent_id not in before])
                if picked:eaten.append(dict(t=round(e.time,2),age_seconds=round(f.age/2,2),energy=round(f.energy,3)))
                else:rotted+=1
        for aid in before:
            if aid not in e.agents_dict:deaths.append(dict(id=aid,t=round(e.time,2)))
        states=[e.get_agent_state(obj.agent_id) for obj in e.agents]
        # Diagnosis only; never correct controller coordinates from true state.
        for obj in e.agents:
            if obj.agent_id in policy.pose:
                px,py,_=policy.pose[obj.agent_id]
                max_pose_error=max(max_pose_error,math.hypot(obj.x-anchor[0]-px,obj.y-anchor[1]-py))
        peak=max(peak,len(e.agents))
        if tick%100==0:trace.append(dict(t=round(e.time,2),alive=len(e.agents),energy=round(sum(obj.energy for obj in e.agents),3),fruits=len(e.fruits),trees=len(e.trees)))
        if not e.agents:break
    return dict(mode=mode,seed=seed,patch=patch,reproduce=reproduce,initial_energy=energy,
        initial_fruit_age=initial_fruit_age,appearance=appearance,initial_agent_age=agent_age,
        horizon=seconds,duration=round(e.time,2),alive=len(e.agents),total_created=e._next_agent_id,
        peak_agents=peak,remaining_energy=round(sum(obj.energy for obj in e.agents),3),
        score=round(e.score,5),fruits_eaten=len(eaten),fruit_nutrition=round(sum(f['energy'] for f in eaten),3),
        actual_energy_received=round(received,3),mean_harvest_age=round(statistics.mean(f['age_seconds'] for f in eaten),3) if eaten else None,
        rotten_fruits=rotted,movement_energy=round(movement_energy,3),turn_energy=round(turn_energy,3),
        known_new_tracks=policy.known_new,unknown_age_tracks=policy.unknown_new,
        waiting_agent_ticks=policy.wait_ticks,emergency_pick_decisions=policy.emergency_picks,
        unlocalized_agent_ticks=policy.unlocalized_ticks,max_pose_error=round(max_pose_error,8),
        deaths=deaths,harvests=eaten,trace=trace,tree_births=tree_births if patch else False,
        patch_radius=policy.patch_radius)


def cases(phase):
    if phase=='finite':
        for appearance in ('initial','new'):
            for fruit_age in (0,10,25,40,49):
                # A fruit deliberately inserted old into a previously empty area
                # cannot establish a real newborn event; reserve 'new' for fresh.
                if appearance=='new' and fruit_age:continue
                for mode in ('immediate','observed','oracle','blind'):
                    yield dict(mode=mode,appearance=appearance,initial_fruit_age=fruit_age)
        for energy in (15,30,45,60):
            for mode in ('immediate','observed','oracle','blind'):
                yield dict(mode=mode,appearance='new',energy=energy)
        for agent_age in (70,100):
            for mode in ('immediate','observed','oracle'):
                yield dict(mode=mode,appearance='new',agent_age=agent_age,seconds=40)
    elif phase=='patch':
        for seed in (3,11,29):
            for reproduce in (False,True):
                for mode in ('immediate','observed','oracle'):
                    yield dict(mode=mode,patch=True,seed=seed,reproduce=reproduce,seconds=300)


def run_case(case):return run(**case)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('finite','patch','all'),default='finite')
    p.add_argument('--workers',type=int,default=1);args=p.parse_args()
    for phase in ('finite','patch') if args.phase=='all' else (args.phase,):
        rows=[]
        pool=None
        if args.workers>1:
            from concurrent.futures import ProcessPoolExecutor
            pool=ProcessPoolExecutor(max_workers=args.workers)
            results=pool.map(run_case,cases(phase))
        else:results=map(run_case,cases(phase))
        for row in results:
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('trace','harvests')}),flush=True)
        if pool:pool.shutdown()
        output=ROOT/'results'/f'fruit-waiting-{phase}.json'
        output.write_text(json.dumps(dict(source_commit='acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Controlled flat patches; observed policy tracks fruit in a verified local coordinate frame; oracle alone receives true fruit ages. No predators, internal obstacles, or remote validation.',runs=rows),indent=2)+'\n')
        print(output,flush=True)


if __name__=='__main__':main()
