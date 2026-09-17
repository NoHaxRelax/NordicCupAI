"""Bounded native-map, replacement and acquisition evaluations."""
from common import *
import argparse
import random
import numpy as np
from controller import PairPolicy,TwinPolicy
from sites import world,local
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.elements.tree import Tree
from src.elements.obstacle import Obstacle
from predator_control import controlled_env
from recording import start_recording, finish_recording

def initialize(case,*,policy_factory=None):
    seed=case.get('seed',173)
    native=case.get('native',False)
    if native:
        env=SimulationCore(seed=seed).env;site=case['site']
    else:
        width=case.get('width',50)
        env=controlled_env(width);env.rng=random.Random(seed)
        env.obstacles=[Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),
                       Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
        env.edges=set()
        for o in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in o.edges)
        site=case.get('site',dict(center=[800,600],angle=0,width=width))
    env.agents=[];env.agents_dict={};env.agent_observations={};env.predators=[];env._next_agent_id=0
    B=site['width']/2+2;outside=case.get('outside',0)
    places=[[-B,0],[B,0],[-B-120,0],[B+120,0]]
    if outside:places[:2]=[[B+8,0],[B,case.get('secondary_offset',60)]]
    poses={};crew=case.get('crew',2)
    for i,uv in enumerate(places[:crew]):
        xy=world(site,uv)
        if env._in_obstacle(tuple(xy),5,env.obstacles):return None,'blocked_start'
        a=Agent(*xy,energy=150,rng=env.rng)
        if case.get('initial_age'):a.age=case['initial_age']
        a.agent_id=i;env.agents.append(a);env.agents_dict[i]=a
        face=math.pi if outside and i<2 else math.pi/2
        a.direction=face+site['angle'];poses[i]=[800+uv[0],600+uv[1],face]
    env._next_agent_id=crew
    for j in range(case.get('predators',1)):
        uv=[site['width']/2+outside if outside else 0,j*20]
        xy=world(site,uv)
        if env._in_obstacle(tuple(xy),10,env.obstacles):return None,'blocked_predator_start'
        p=Predator(*xy,energy=102,rng=env.rng)
        p.direction=site['angle']+case.get('heading',0)+(math.pi if outside else 0)
        p.resting=False;env.predators.append(p)
    if not case.get('natural_predator_spawns',False):env.spawn_predator=lambda *a,**kw:None
    if case.get('second_station'):
        second=case['second_station'];B2=second['width']/2+2
        poses2={}
        for aid,side in [(2,-1),(3,1)]:
            a=env.agents_dict[aid];uv=[side*B2,0]
            a.x,a.y=world(second,uv);a.direction=second['angle']+math.pi/2
            poses2[aid]=[800+uv[0],600.,math.pi/2]
        env.predators[1].x,env.predators[1].y=second['center']
        env.predators[1].direction=second['angle']+case.get('heading',0)
        for a in env.agents[2:]+env.predators[1:]:
            if env._in_obstacle((a.x,a.y),a.size,env.obstacles):return None,'second_station_blocked'
    if case.get('orchard',False):
        # Mechanism fixture only: placed trees, no placed or replenished fruit.
        for side in [-1,1]:
            for dy in [-35,0,35]:
                t=Tree(800+side*(B+120),600+dy);t.grow(35);env.trees.append(t)
    env._update_spatial_grid()
    for a in env.agents:
        ag,f,tr,o,pr,edges=env._get_local_objects(a)
        env.agent_observations[a.agent_id]=a.observe(agents=ag,fruits=f,trees=tr,predators=pr,edges=edges)
    obstacles=[(o.x,o.y,o.width,o.height) for o in env.obstacles] if case.get('collision_model',False) else []
    policy=(policy_factory or PairPolicy)(site,poses,outside=outside,relay=case.get('relay','none'),
        mapped_obstacles=obstacles,food=case.get('food',True),safety_launch=case.get('safety_launch',False),
        anchor_window=case.get('anchor_window'),association_gate=case.get('association_gate',False))
    if case.get('second_station'):
        policy=TwinPolicy([site,case['second_station']],[{k:v for k,v in poses.items() if k<2},poses2],obstacles,case.get('anchor_window'),case.get('association_gate',False))
    if case.get('comparison'):
        from comparison import ComparisonPolicy
        policy=ComparisonPolicy(policy,baseline=case['comparison']=='nursery')
    return (env,policy,site),None

def run(case,record=None,*,reproduction=None,policy_factory=None):
    init,skip=initialize(case,policy_factory=policy_factory)
    if skip:return dict(case=case,skipped=skip)
    env,policy,site=init
    recorder,replay_path=start_recording(env,case,policy='Mapped water deployment',
        demonstration=record,reproduction=reproduction,
        notes='Arranged initial creatures. Cached public predator observations; known initial shared poses and mapped river geometry. '+
        ('Mapped static obstacles used for collision dead reckoning.' if case.get('collision_model') else 'Simple dead reckoning.'))
    ticks=joint=water_n=attention_n=active_joint=active_ticks=0
    after=after_joint=after_water=after_att=0
    first_capture=first_depletion=first_loss=None
    food={};move_energy={};birth_energy=0;deaths=[];trace=[]
    maximum_pose_error=0;worker_ticks=0;worker_captures=0;births=[]
    max_predators=len(env.predators)
    for tick in range(round(case.get('seconds',90)*10)):
        if not env.agents:break
        before=list(env.agents);states=[env.get_agent_state(a.agent_id) for a in before]
        energy0={a.agent_id:a.energy for a in before};t0=env.time
        actions=policy(states,env.time)
        assert {aid for aid,_ in actions}=={a.agent_id for a in before}
        for aid,a in actions:
            assert all(math.isfinite(float(getattr(a,k))) for k in ['move_distance','move_direction','turn_angle'])
            assert a.move_distance>=0
        policy.integrate(actions,states)
        for aid,a in actions:
            n=len(env.agents)
            env.agent_step(aid,a.move_distance,a.move_direction,a.turn_angle,a.spawn_agent)
            if len(env.agents)>n:
                births.append(dict(t=t0,parent=aid,child=env.agents[-1].agent_id,energy=env.agents[-1].energy))
                birth_energy+=100
            cost=energy0[aid]-env.agents_dict[aid].energy-(100 if len(env.agents)>n else 0)
            move_energy[aid]=move_energy.get(aid,0)+cost
        e_after={a.agent_id:a.energy for a in env.agents}
        ages_after_action={a.agent_id:a.age for a in env.agents}
        drains={a.agent_id:.1*env.biome_map[int(a.x),int(a.y)].energy_drain_rate+
            (.01*(a.age+.1) if a.age+.1>a.max_age else 0) for a in env.agents}
        pairs=[]
        for p in env.predators:
            seen=p.observe(agents=env._get_local_agents(p),edges=env._get_local_edges(p))
            detects=[o for o in seen if o['type']=='Agent']
            selected=min(detects,key=lambda o:o['distance'])['id'] if detects else None
            pairs.append((p,selected,not p.resting))
        env.non_agent_step(.1)
        env.agents_dict={a.agent_id:a for a in env.agents}
        for a in env.agents:
            # Native list-removal can skip a surviving agent's pass. Its age
            # then remains unchanged, so neither passive nor old-age cost ran.
            drain=drains[a.agent_id] if a.age>ages_after_action[a.agent_id] else 0
            gain=a.energy-e_after[a.agent_id]+drain
            if gain>1e-7:food[a.agent_id]=food.get(a.agent_id,0)+gain
            if a.agent_id in policy.positions:
                true=local(site,[a.x,a.y])+[800,600]
                maximum_pose_error=max(maximum_pose_error,float(np.linalg.norm(true-policy.positions[a.agent_id])))
        water=[env.biome_map[max(0,min(1599,int(p.x))),max(0,min(1199,int(p.y)))].type=='river' for p,_,_ in pairs]
        attention=[aid in policy.active for _,aid,_ in pairs]
        allw=all(water);alla=all(attention);both=allw and alla
        ticks+=1;water_n+=allw;attention_n+=alla;joint+=both
        isactive=any(active for _,_,active in pairs)
        active_ticks+=isactive;active_joint+=isactive and both
        if policy.base.acquisition_complete is not None:
            after+=1;after_joint+=both;after_water+=allw;after_att+=alla
        for a in before:
            if a not in env.agents:
                captured=a.energy>0
                deaths.append(dict(t=env.time,id=a.agent_id,cause='capture' if captured else 'depletion',
                    energy=a.energy,age=a.age,max_age=a.max_age))
                if first_loss is None:first_loss=env.time
                if captured and first_capture is None:first_capture=env.time
                if not captured and first_depletion is None:first_depletion=env.time
                if captured and a.agent_id not in policy.active:worker_captures+=1
        worker_ticks+=sum(a.agent_id not in policy.active for a in env.agents)
        max_predators=max(max_predators,len(env.predators))
        if tick%10==0:trace.append(dict(t=round(env.time,2),active=policy.active.copy(),
            agents=[dict(id=a.agent_id,uv=local(site,[a.x,a.y]).tolist(),energy=a.energy,age=a.age) for a in env.agents],
            predators=[dict(uv=local(site,[p.x,p.y]).tolist(),water=w,target=aid) for (p,aid,_),w in zip(pairs,water)],
            pose_error=maximum_pose_error,score=env.score))
        recorder.capture(actions,policy.decisions,states,t0)
        if case.get('stop_on_loss',True) and (first_capture is not None or any(a not in env.agents_dict for a in policy.active)):break
    row=dict(case=case,elapsed_s=env.time,alive=len(env.agents),score=env.score,
        acquired_s=policy.base.acquisition_complete,water_fraction=water_n/max(1,ticks),
        bait_attention_fraction=attention_n/max(1,ticks),joint_fraction=joint/max(1,ticks),
        active_joint_fraction=active_joint/max(1,active_ticks),after_acquisition_s=after/10,
        after_acquisition_joint_fraction=after_joint/max(1,after),
        after_acquisition_water_fraction=after_water/max(1,after),after_acquisition_attention_fraction=after_att/max(1,after),
        first_capture_s=first_capture,first_depletion_s=first_depletion,deaths=deaths,
        handoffs=policy.events,births=births,newborn_localization=policy.births,
        movement_turn_energy=move_energy,absorbed_native_fruit_energy=food,birth_energy_cost=birth_energy,
        energy_remaining={a.agent_id:a.energy for a in env.agents},
        maximum_pose_error=maximum_pose_error,worker_living_seconds=worker_ticks/10,
        worker_captures=worker_captures,maximum_predators=max_predators,trace=trace)
    row['fallback_s']=getattr(policy,'fallback_t',None)
    reason='requested horizon reached' if ticks>=round(case.get('seconds',90)*10) else 'loss or extinction stopped the run'
    row['recording']=finish_recording(recorder,replay_path,reason=reason)
    return row

def suite(mode):
    cases=[]
    if mode=='native':
        maps=json.loads((OUT/'sites-1-40.json').read_text())['data']
        for m in maps:
            for i,s in enumerate(m['sites']):
                if s['obstructed_samples']>30:continue
                for collision in [False,True]:
                    cases.append(dict(native=True,seed=m['seed'],site=s,site_index=i,
                        collision_model=collision,seconds=70))
    elif mode=='relay':
        for seed in [173,177,181]:
            for relay in ['none','reserve','newborn']:
                cases.append(dict(seed=seed,width=50,seconds=120,crew=4,relay=relay))
    elif mode=='orchard':
        for seed in [173,177,181]:
            for relay in ['none','reserve','newborn']:
                cases.append(dict(seed=seed,width=50,seconds=180,crew=4,relay=relay,orchard=True))
    results=[]
    for case in cases:
        r=run(case);results.append(r)
        print({k:v for k,v in r.items() if k not in ['trace','case','movement_turn_energy','newborn_localization']},flush=True)
        save(mode,results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','relay','orchard'])
    a=p.parse_args();suite(a.mode)
