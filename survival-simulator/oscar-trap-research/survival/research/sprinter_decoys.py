"""Dedicated decoy and separate banishment probes using actual upstream ticks.

Synthetic arranged forest arenas, no new predator/tree spawns. Decoy decisions
use cached observations and reported energy. Banishment uses assigned world
waypoints and true bait pose in this controlled fixture, an unvalidated
localization/acquisition assumption. Hidden predator energy/rest are measurement
fields only. Adaptive sprint estimates use odometry, inaccurate at collisions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/survival-simulator'))
sys.path.insert(0,str(ROOT/'debugger'))
import numpy as np
from src.elements.environment import Environment
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.elements.obstacle import Obstacle
from src.elements.biome import Forest_biome
from src.utils.DTOs import ActionRequest


def wrap(x): return (x+math.pi)%(2*math.pi)-math.pi


def fixture(seed=0,width=1600,height=1200):
    e=Environment.__new__(Environment)
    e.width,e.height,e.chunk_size=width,height,400
    e.rng=random.Random(seed)
    e.agents=[];e.predators=[];e.fruits=[];e.trees=[]
    e.obstacles=[Obstacle(0,0,width,30),Obstacle(0,height-30,width,30),
                 Obstacle(0,0,30,height),Obstacle(width-30,0,30,height)]
    e.edges=set(edge for o in e.obstacles for edge in o.edges)
    e.agents_dict={};e.fruits_dict={};e.agent_observations={}
    e._next_agent_id=e._next_fruit_id=0
    e.score=e.time=0.
    e.biome_map=np.full((width,height),Forest_biome(),dtype=object)
    e.spawn_tree=lambda *a,**kw:None
    e.spawn_predator=lambda *a,**kw:None
    e._update_spatial_grid()
    return e


def add_agent(e,x,y,energy=150,capacity=500):
    a=Agent(x,y,energy=energy,max_energy=capacity,max_age=90,rng=e.rng)
    a.agent_id=e._next_agent_id;e._next_agent_id+=1
    a.direction=math.pi
    e.agents.append(a);e.agents_dict[a.agent_id]=a;e._update_agent_grid()
    return a


def add_pred(e,x=700,y=600,energy=102,heading=0):
    p=Predator(x,y,energy=energy,rng=e.rng)
    p.direction=heading;p.resting=False
    e.predators.append(p);e._update_predator_grid()
    return p


def observe(e):
    for a in e.agents:
        aa,ff,tt,oo,pp,ee=e._get_local_objects(a)
        e.agent_observations[a.agent_id]=a.observe(agents=aa,fruits=ff,trees=tt,
                                                 obstacles=oo,predators=pp,edges=ee)
    return [e.get_agent_state(a.agent_id) for a in e.agents]


def ripe_food(e,points):
    for x,y in points:
        f=e.spawn_fruit(x=x,y=y)
        if f: f.grow(40) # source-equivalent freshly mature food, 30s left before rot


class DecoyController:
    def __init__(self,mode='pulse',request=16,trigger=60,release_x=None,initial=None):
        self.mode=mode;self.request=request;self.trigger=trigger
        self.release_x=release_x
        self.dead=dict(initial or {}) # arranged initial frame, only used for banishment
        self.last_pred={};self.stationary={};self.phase='lead'
        self.release_time=None;self.active=0;self.switches=0;self.retired=set()
        self.escape_origin=None;self.escape_angle=None

    def __call__(self,states,t,relay=False):
        actions=[];decisions={}
        eligible=[s for s in states if s['agent_id'] in self.dead]
        if relay and eligible:
            current=next((s for s in eligible if s['agent_id']==self.active),None)
            if current is None or current['energy']<200:
                candidates=[s for s in eligible if s['agent_id']!=self.active and s['energy']>135
                            and (current is None or s['energy']>current['energy']+50)]
                if candidates:
                    self.retired.add(self.active)
                    self.active=max(candidates,key=lambda s:s['energy'])['agent_id'];self.switches+=1
        for s in states:
            aid=s['agent_id'];move=0.;direction=0.;turn=0.;why='colony stationary'
            if aid not in self.dead:
                actions.append((aid,ActionRequest(agent_id=aid,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)))
                continue
            x,y,theta=self.dead[aid]
            pp=[p for p in s['observations'] if p['type']=='Predator']
            p=min(pp,key=lambda q:q['distance']) if pp else None
            walk=min(s['speed'],s['sprint_speed'])
            if p:
                d,q=p['distance'],p['angle']
                point=(x+d*math.cos(theta+q),y+d*math.sin(theta+q))
                old=self.last_pred.get(aid)
                still=old is not None and math.dist(point,old)<.05
                observed_speed=math.dist(point,old) if old is not None else 15
                self.stationary[aid]=self.stationary.get(aid,0)+1 if still else 0
                self.last_pred[aid]=point
                direction=wrap(q+math.pi);turn=q
                if self.release_x is not None:
                    # Assigned banishment route is east; odometry works only in
                    # this clear fixture. No hidden predator pose/rest is read.
                    if self.phase=='lead' and x>=self.release_x:
                        self.phase='wait_for_rest'
                    if self.phase=='wait_for_rest' and self.stationary.get(aid,0)>=2:
                        self.phase='leave';self.release_time=t
                        self.escape_origin=point
                        # Controlled pose-aware clearance: pick an available
                        # 320-unit straight exit, maximizing predator separation.
                        candidates=[k*math.pi/4 for k in range(8)]
                        candidates=[z for z in candidates if 80<x+320*math.cos(z)<1520
                                    and 80<y+320*math.sin(z)<1120]
                        self.escape_angle=max(candidates,key=lambda z:math.dist(
                            (x+320*math.cos(z),y+320*math.sin(z)),point)) if candidates else math.pi
                    if self.phase=='leave':
                        # Walk beyond the vision radius during the observed rest.
                        direction=wrap(self.escape_angle-theta)
                        move=walk if math.dist((x,y),self.escape_origin)<310 else 0
                        turn=0
                        why='leave during inferred rest'
                    elif self.phase in ('lead','wait_for_rest'):
                        direction=wrap(0-theta)
                        move=min(self.request,s['sprint_speed']) if d<self.trigger else 0
                        why=self.phase
                elif relay and aid!=self.active:
                    if aid in self.retired:
                        move=min(20,s['sprint_speed']) if d<110 else 0
                        why='retired decoy builds separation'
                    else:
                        # Reserve cannot stand in the advancing predator's path.
                        if d<85: move=min(self.request,s['sprint_speed'])
                        elif d>115: direction=q;move=min(walk,d-115)
                        why='reserve keeps safer distance'
                else:
                    requested=self.request
                    if self.mode=='adaptive':requested=max(walk,min(20,observed_speed+.1))
                    move=min(requested,s['sprint_speed']) if self.mode=='continuous' or d<self.trigger else 0
                    why='pulse sprint' if move else 'wait for predator approach'
            elif self.release_x is not None and self.phase=='leave':
                direction=wrap(self.escape_angle-theta)
                move=walk if math.dist((x,y),self.escape_origin)<310 else 0
                why='continue leaving' if move else 'remote patch reached'
            else:
                turn=.2;why='scan after lost contact'
            # Source clamps low-energy movement before biome reduction.
            actual=min(move,s['sprint_speed'])
            if s['energy']<s['max_energy']/5 and actual>s['speed']: actual=s['speed']
            self.dead[aid]=(x+actual*math.cos(theta+direction),y+actual*math.sin(theta+direction),theta+turn)
            action=ActionRequest(agent_id=aid,move_distance=move,move_direction=direction,turn_angle=turn,spawn_agent=False)
            actions.append((aid,action));decisions[aid]=why
        return actions,decisions


def run_case(*,kind='decoy',request=16,trigger=60,energy=150,capacity=500,
             food=False,relay=False,mode='pulse',heading=0,gap=60,pred_energy=102,
             distractor=False,seed=0,seconds=45,release_x=1200,record=None):
    e=fixture(seed)
    p=add_pred(e,heading=heading,energy=pred_energy)
    a=add_agent(e,700+gap,600,energy,capacity)
    decoys=[a]
    if relay: decoys.append(add_agent(e,700+gap+60,625,energy,capacity))
    colony=[]
    if kind in ('banishment','baseline'):
        colony=[add_agent(e,360,600,500),add_agent(e,400,660,500)]
    if distractor: colony.append(add_agent(e,1050,650,500))
    if food: ripe_food(e,[(x,600) for x in (900,1060,1220,1380)])
    if kind=='baseline':
        e.kill_agent(a);decoys=[]
    initial={a.agent_id:(a.x,a.y,a.direction) for a in decoys}
    ctrl=DecoyController(mode,request,trigger,release_x if kind=='banishment' else None,initial)
    states=observe(e)
    recorder=None
    if record:
        from recorder import ReplayRecorder
        recorder=ReplayRecorder(e,title=record.stem,policy='dedicated sprinter '+kind,seed=seed,
            scenario='controlled '+kind,notes='Arranged forest arena; finite ripe fruit if shown. Decoy uses cached observations and energy. Banishment has assigned waypoint and privileged true bait pose. No remote evaluation.',every=2,
            policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        recorder.capture(force=True)
    deaths={};rest_cycles=0;attention=0;active_attention=0;active_ticks=0
    min_gap=math.inf;max_pred_x=p.x;fruit_eaten=0;fruit_energy=0;food_received=0.
    colony_approach=None;first_colony_entry=None;capture=None;cutoff=None;travel_energy=0.;turn_energy=0.
    boundary_ticks=0;trace=[];release_pred_x=None;living_energy=0.;captured_energy=0.
    for tick in range(int(seconds*10)):
        before={a.agent_id:a for a in e.agents}
        inputs=states
        if kind=='banishment':
            # Explicit diagnostic privilege: localization/acquisition is not
            # being claimed solved. It prevents wall sliding corrupting odometry.
            for obj in decoys:
                if obj.agent_id in e.agents_dict:ctrl.dead[obj.agent_id]=(obj.x,obj.y,obj.direction)
        actions,decisions=ctrl(states,e.time,relay)
        action_time=e.time
        for aid,act in actions:
            old=before[aid].energy
            e.agent_step(aid,act.move_distance,act.move_direction,act.turn_angle,False)
            if aid in initial:
                turn_cost=min(math.pi,abs(act.turn_angle))/(2*math.pi)
                travel_energy+=old-before[aid].energy-turn_cost;turn_energy+=turn_cost
        # Measure the actual target selected by predator's real sensing before tick.
        obs=p.observe(agents=e._get_local_agents(p),edges=e._get_local_edges(p))
        targets=[o for o in obs if o['type']=='Agent']
        chosen=min(targets,key=lambda o:o['distance'])['id'] if targets else None
        attention+=chosen in initial
        active_attention+=(chosen in initial) and not p.resting
        active_ticks+=not p.resting
        previous_rest=p.resting
        fruits_before=set(e.fruits)
        pre_interaction_energy={obj.agent_id:obj.energy for obj in decoys if obj.agent_id in e.agents_dict}
        passive_costs={}
        for obj in decoys:
            if obj.agent_id in e.agents_dict:
                cost=.1
                if obj.energy>.1 and obj.age+.1>obj.max_age:cost+=.01*(obj.age+.1)
                living_energy+=cost;passive_costs[obj.agent_id]=cost
        e.non_agent_step(.1)
        for obj in decoys:
            if obj.agent_id in pre_interaction_energy:
                food_received+=max(0.,obj.energy-pre_interaction_energy[obj.agent_id]+passive_costs[obj.agent_id])
        # Removed unrotted food was eaten (fixed initially-ripe fruit expires30s).
        eaten=[f for f in fruits_before if f not in e.fruits and f.age<=100]
        fruit_eaten+=len(eaten);fruit_energy+=sum(f.energy for f in eaten)
        rest_cycles+=(not previous_rest and p.resting)
        max_pred_x=max(max_pred_x,p.x)
        for aid,obj in before.items():
            if aid not in e.agents_dict and aid not in deaths:
                deaths[aid]=round(e.time,2)
                if aid in initial and obj.energy>0:
                    capture=capture or e.time;captured_energy+=obj.energy
        alive_decoys=[a for a in decoys if a.agent_id in e.agents_dict]
        # Include the just-captured bait's final position. Considering only
        # survivors would report a falsely safe minimum gap on a capture tick.
        present_this_tick=[obj for aid,obj in before.items() if aid in initial]
        if present_this_tick:
            min_gap=min(min_gap,*(math.hypot(obj.x-p.x,obj.y-p.y) for obj in present_this_tick))
        if alive_decoys:
            if cutoff is None and any(a.energy<a.max_energy/5 for a in alive_decoys): cutoff=e.time
            boundary_ticks+=any(a.x<50 or a.x>1550 or a.y<50 or a.y>1150 for a in alive_decoys)
        if ctrl.release_time is not None:
            if release_pred_x is None:release_pred_x=p.x
            if colony_approach is None and math.dist((p.x,p.y),(380,630))<250:colony_approach=e.time
        if first_colony_entry is None and math.dist((p.x,p.y),(380,630))<250:first_colony_entry=e.time
        states=[e.get_agent_state(a.agent_id) for a in e.agents]
        if recorder:recorder.capture(actions,decisions,inputs,action_time)
        if tick%10==0:trace.append(dict(t=round(e.time,2),pred=[round(p.x,2),round(p.y,2)],rest=p.resting,
            predator_energy=round(p.energy,2),decoys=[[a.agent_id,round(a.x,2),round(a.y,2),round(a.energy,2)] for a in alive_decoys],phase=ctrl.phase))
        if kind=='decoy' and not alive_decoys:break
    duration=e.time
    result=dict(kind=kind,request=request,trigger=trigger,initial_energy=energy,capacity=capacity,
        food=food,relay=relay,mode=mode,heading=heading,initial_gap=gap,predator_initial_energy=pred_energy,
        distractor=distractor,seed=seed,horizon=seconds,duration=round(duration,2),release_x=release_x,
        policy_privilege='true bait pose and assigned route' if kind=='banishment' else 'cached observations and clear-ground odometry for adaptive speed estimate',
        decoys_alive=sum(a.agent_id in e.agents_dict for a in decoys),
        colony_alive=sum(a.agent_id in e.agents_dict for a in colony),
        deaths=deaths,capture_time=None if capture is None else round(capture,2),
        first_sprint_cutoff=None if cutoff is None else round(cutoff,2),
        decoy_energy_remaining=round(sum(a.energy for a in decoys if a.agent_id in e.agents_dict),3),
        movement_energy=round(travel_energy,3),turn_energy=round(turn_energy,3),
        living_energy=round(living_energy,3),captured_decoy_remaining_energy=round(captured_energy,3),
        ripe_fruits_eaten=fruit_eaten,fruit_energy=round(fruit_energy,3),fruit_energy_received_by_decoys=round(food_received,3),
        min_gap=None if not math.isfinite(min_gap) else round(min_gap,3),
        predator_attention_fraction=round(attention/(tick+1),4),
        predator_active_attention_fraction=round(active_attention/max(1,active_ticks),4),
        predator_rest_cycles=rest_cycles,max_predator_x=round(max_pred_x,2),
        predator_displacement_east=round(max_pred_x-700,2),boundary_ticks=boundary_ticks,
        relay_switches=ctrl.switches,release_time=None if ctrl.release_time is None else round(ctrl.release_time,2),
        release_predator_x=None if release_pred_x is None else round(release_pred_x,2),
        predator_reentered_colony_radius=colony_approach,
        first_colony_entry=first_colony_entry,
        trace=trace)
    if recorder:
        recorder.save(record,reason='controlled experiment end',overwrite=True)
        result['replay']=str(record)
    return result


PHASES=('screen','adaptive','capacity','heldout','banishment','banishment-heldout','replays')


def cases_for(phase):
    if phase=='screen':
        for energy in (150,500):
            for request in (10,15.1,16,20):
                for mode in ('continuous','pulse'):
                    yield dict(request=request,energy=energy,mode=mode)
        for trigger in (30,40,50,75,90):
            yield dict(request=16,trigger=trigger,energy=500)
        for food in (False,True):
            for relay in (False,True):
                for energy in (150,500):
                    yield dict(request=16,energy=energy,food=food,relay=relay)
        for capacity in (300,350,500):
            yield dict(request=16,energy=75,capacity=capacity)
    elif phase=='adaptive':
        for energy in (150,500):
            for food in (False,True):
                for mode in ('pulse','adaptive'):
                    for trigger in (60,90):yield dict(mode=mode,trigger=trigger,energy=energy,food=food,seconds=60)
    elif phase=='capacity':
        for energy in (75,150):
            for cap in (150,200,300,500):
                for mode in ('pulse','adaptive'):
                    yield dict(energy=energy,capacity=cap,mode=mode,request=20 if mode=='adaptive' else 16,seconds=45)
    elif phase=='heldout':
        for seed,heading,gap,pe in ((1,-.5,45,42),(2,.5,75,102),(3,1.2,100,200),(4,-1.2,60,150),(5,3.14,80,102)):
            for request,mode in ((15.1,'pulse'),(16,'pulse'),(20,'pulse'),(20,'adaptive')):
                for food in (False,True):
                    yield dict(request=request,mode=mode,energy=500,heading=heading,gap=gap,pred_energy=pe,food=food,seconds=60,seed=seed)
    elif phase=='banishment':
        for energy in (150,500):
            for release_x in (1000,1200,1350):
                for food in (False,True):
                    for distractor in (False,True):
                        yield dict(kind='banishment',energy=energy,release_x=release_x,food=food,distractor=distractor,seconds=100)
    elif phase=='banishment-heldout':
        for seed,heading,pe in ((0,0,102),(11,-.5,42),(12,.5,102),(13,1.2,200),(14,-1.2,150),(15,3.14,102)):
            for kind in ('baseline','banishment'):
                yield dict(kind=kind,energy=500,heading=heading,pred_energy=pe,seed=seed,seconds=100,release_x=1000)
    elif phase=='replays':
        yield dict(mode='adaptive',energy=500,food=True,seconds=60,record=ROOT/'results/sprinter-decoys-adaptive-replay.json.gz')
        yield dict(kind='banishment',energy=500,seconds=60,release_x=1000,record=ROOT/'results/sprinter-decoys-banishment-replay.json.gz')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=[*PHASES,'all'],default='screen')
    args=parser.parse_args()
    for phase in PHASES if args.phase=='all' else (args.phase,):
        results=[run_case(**case) for case in cases_for(phase)]
        output=ROOT/'results'/f'sprinter-decoys-{phase}.json'
        output.write_text(json.dumps({'scope':'Controlled arranged forest arenas; no remote validation. Decoy cached-observation policies; banishment has true bait pose and assigned geometry.',
            'source_commit':'acfc31a4003a5f91bf11032a02cd98c178ddbd7e',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runs':results},indent=2)+'\n')
        print(output,flush=True)
        for row in results:print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)


if __name__=='__main__':main()
