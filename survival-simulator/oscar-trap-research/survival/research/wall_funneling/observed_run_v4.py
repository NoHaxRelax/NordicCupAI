"""Evaluate an observation-only controller; hidden state is used ONLY by the scorer.

Arranged native-sized walls and initial agents/approaches, unlimited agent food,
native sensing/physics/captures/births. No global map or fixture data enters policy.
This does not establish wall discovery from an unarranged generated-game start.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from uuid import uuid4
from run import ROOT, OUT, controlled_env, add_predator, Obstacle, ReplayRecorder
from predator_control import add_agent
from observed_policy import ObservedFunnel
from src.utils.DTOs import ActionRequest, ObservationResponse
POLICY_HASH=hashlib.sha256(Path(__file__).with_name('observed_policy.py').read_bytes()).hexdigest()


def run(width=30,length=100,predators=3,spread=15.,seconds=180.,seed=41,
        gate=True,horizontal=False,awake=False,native=False,stations=1,
        replenish=False,depth_spread=40.):
    assert 30<=width<=35 and 70<=length<=100
    assert 1<=stations<=9 and predators>=stations
    env=controlled_env();env.rng.seed(seed)
    rng=random.Random(seed)
    env.obstacles=[Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),
                   Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
    fixtures=[]
    for i in range(stations):
        cx,cy=(800,600) if stations==1 else (430+(i%3)*430,300+(i//3)*350)
        w,h=(length,width) if horizontal else (width,length)
        wall=Obstacle(cx-w/2,cy-h/2,w,h);env.obstacles.append(wall)
        n=(0,1) if horizontal else (1,0)
        t=(1,0) if horizontal else (0,1)
        def point(normal,tangent=0):
            return (cx+n[0]*normal+t[0]*tangent,cy+n[1]*normal+t[1]*tangent)
        anchor=point(width/2+5.1)
        holder=add_agent(env,x=anchor[0],y=anchor[1],energy=150);holder.direction=math.atan2(-n[1],-n[0])
        gx,gy=point(-width/2-7)
        guide=add_agent(env,x=gx,y=gy,energy=150);guide.direction=math.atan2(n[1],n[0])
        env._next_agent_id=len(env.agents)
        count=predators//stations+int(i<predators%stations)
        ps=[]
        for j in range(count):
            px,py=point(-width/2-rng.uniform(190,190+depth_spread),rng.uniform(-spread,spread))
            p=add_predator(env,px,py,heading=math.atan2(n[1],n[0])+rng.uniform(-.3,.3),
                           energy=102 if awake else 0)
            p.resting=not awake;ps.append(p)
        fixtures.append(dict(wall=wall,holder=holder,anchor=anchor,n=n,t=t,predators=ps))
    env.edges={tuple(sorted((a,b))) for o in env.obstacles for a,b in o.edges}
    env._update_spatial_grid()
    policy=ObservedFunnel(capacity=math.ceil(predators/stations),gate=gate,replenish=replenish)
    tag=f'observed-w{width}-l{length}-n{predators}-sites{stations}-s{seed}-spread{spread}-gate{int(gate)}-h{int(horizontal)}-a{int(awake)}-{uuid4().hex[:8]}'
    path=OUT/'replays'/f'{tag}.json.gz'
    rec=ReplayRecorder(env,title=tag,policy='observation-only-funnel-v4',seed=seed,every=10,
        scenario='arranged walls and predator approaches; observation-only policy',
        notes=__doc__,native_render=native,native_width=800,
        policy_sha256=POLICY_HASH)
    rec.capture()
    states=[ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]
    acquired={};losses={};streaks={p:0 for p in env.predators}
    trace=[];deaths=[];births=0;tail=[];max_held=0;max_maps=0
    for tick in range(round(seconds*10)):
        if not env.agents:
            break
        # Food assumption is a disclosed fixture intervention. It does not
        # refill predators, prevent contact deaths or override native births.
        for a in env.agents:
            a.energy=a.max_energy
        # Serialization boundary: only the native DTO schema plus public time.
        inputs=json.loads(json.dumps(states))
        actions=policy.act(inputs,env.time)
        assert len(actions)==len(env.agents)==len({a['agent_id'] for a in actions})
        before=list(env.agents);before_ids={a.agent_id for a in before}
        pairs=[];action_t=env.time
        for action in actions:
            a=env.agents_dict[action['agent_id']]
            assert 0<=action['move_distance']<=a.sprint_speed+.001
            assert all(math.isfinite(action[k]) for k in ['move_distance','move_direction','turn_angle'])
            pairs.append((a.agent_id,ActionRequest(**action)))
            env.agent_step(**action)
        births+=len({a.agent_id for a in env.agents}-before_ids)
        env.non_agent_step(.1)
        deaths.extend(dict(id=a.agent_id,time=round(env.time,1),holder=any(a is f['holder'] for f in fixtures))
                      for a in before if a.agent_id not in env.agents_dict)
        # Scoring uses hidden coordinates after actions. Never pass these values
        # to the controller, its gate, the next DTOs or arrival timing.
        held=0
        for p in env.predators:
            valid=False
            for f in fixtures:
                holder=f['holder'];anchor=f['anchor'];n=f['n'];t=f['t']
                delta=(p.x-anchor[0],p.y-anchor[1])
                normal=delta[0]*n[0]+delta[1]*n[1]
                tangent=delta[0]*t[0]+delta[1]*t[1]
                seen=[o for o in p.observe(agents=list(env.agents),edges=list(env.edges)) if o['type']=='Agent']
                chosen=min(seen,key=lambda o:o['distance'])['id'] if seen else None
                valid=(holder.agent_id in env.agents_dict and math.hypot(*delta)<=60
                    and normal<=-(width+15.1)+.01 and abs(tangent)<length/2-5
                    and (p.resting or chosen==holder.agent_id))
                if valid:break
            streaks[p]=streaks[p]+1 if valid else 0
            if streaks[p]>=20 and p not in acquired:
                acquired[p]=round(env.time-1.9,1)
            if p in acquired and not valid and p not in losses:
                losses[p]=round(env.time,1)
            held+=int(valid)
        tail.append(held);max_held=max(max_held,held);max_maps=max(max_maps,len(policy.stations))
        if tick%10==0:
            trace.append(dict(time=round(env.time,1),held=held,alive=len(env.agents),
                mapped=len(policy.stations),observed_counts=[s['seen'] for s in policy.stations.values()]))
        rec.capture(pairs,policy.decisions,inputs=states,action_t=action_t)
        states=[ObservationResponse(**env.get_agent_state(a.agent_id)).model_dump() for a in env.agents]
    rec.save(path,reason='horizon' if env.time>=seconds-.01 else 'all agents died')
    row=dict(width=width,length=length,predators=predators,spread=spread,seconds=round(env.time,1),
        seed=seed,gate=gate,horizontal=horizontal,awake=awake,stations=stations,births=births,
        replenish=replenish,depth_spread=depth_spread,policy_hash=POLICY_HASH,
        deaths=deaths,mapped=max_maps,acquired=len(acquired),lost_after_acquisition=len(losses),
        max_held=max_held,final_held=tail[-1],tail_min=min(tail[-300:]),
        all_held=env.time>=seconds-.01 and min(tail[-300:])==predators,
        continuous_all=len(acquired)==predators and not losses and env.time>=seconds-.01,
        holders_alive=sum(f['holder'].agent_id in env.agents_dict for f in fixtures),
        replay=str(path.relative_to(ROOT)),events=policy.events,trace=trace)
    (OUT/f'{tag}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width',type=float,default=30)
    p.add_argument('--length',type=float,default=100)
    p.add_argument('--predators',type=int,default=3)
    p.add_argument('--spread',type=float,default=15)
    p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--stations',type=int,default=1)
    p.add_argument('--ungated',dest='gate',action='store_false')
    p.add_argument('--renew-guides',dest='replenish',action='store_true')
    p.add_argument('--depth-spread',type=float,default=40)
    p.add_argument('--horizontal',action='store_true')
    p.add_argument('--awake',action='store_true')
    p.add_argument('--native',action='store_true')
    r=run(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in r.items() if k not in ['trace','events']},indent=2))
