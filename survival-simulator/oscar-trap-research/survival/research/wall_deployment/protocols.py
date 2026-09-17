"""Bounded native-physics trap protocols with explicitly privileged fixture control.

Controllers know exact geometry and creature positions in this experiment.
This tests feasibility, not observation-only deployment or native game scores.
All agents start with 150 energy; flat forest, fixed walls, no food or new spawns.
Every living agent receives exactly one legal action per tick; every run records.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from predator_control import controlled_env, add_predator, wrap
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ActionRequest
from experiments import OUT, ROOT, verify_source
from recorder import ReplayRecorder


def run(kind='first',width=30,height=100,gap=60,side=1,heading=0.,seconds=35,
        intruder_distance=160,resting=False,native=False):
    env=controlled_env();wall=Obstacle(800-width/2,600-height/2,width,height)
    env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),
        Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
    for w in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
    anchor=(wall.x+width+7,600);front=wall.x-7
    holder=env.spawn_agent(x=anchor[0],y=anchor[1]);holder.direction=math.pi
    guard=env.spawn_agent(x=anchor[0]+45,y=600);guard.direction=0
    original=None;incoming=None;guide=None
    if kind!='first':original=add_predator(env,wall.x-12,600,energy=102)
    if kind.startswith('guard'):
        incoming=add_predator(env,anchor[0]+intruder_distance,600,heading=math.pi,energy=0 if resting else 102)
        incoming.resting=resting
    else:
        vertical=kind.startswith('corner')
        gx,gy=(front,600+side*(height/2+90)) if vertical else (wall.x-125,600)
        guide=env.spawn_agent(x=gx,y=gy)
        px,py=(gx,gy+side*gap) if vertical else (gx-gap,gy)
        incoming=add_predator(env,px,py,heading=(-side*math.pi/2 if vertical else 0)+heading)
        guide.direction=math.atan2(py-gy,px-gx)
    env._update_spatial_grid()
    tag=f'{kind}-w{width}-h{height}-gap{gap}-side{side}-d{intruder_distance}-rest{int(resting)}'
    path=OUT/'replays'/f'protocol-v2-{tag}-{uuid4().hex[:10]}.json.gz'
    rec=ReplayRecorder(env,title=f'Trap protocol v2: {tag}',policy='privileged protocol feasibility v2',seed=1729,
        every=10,scenario='arranged flat forest',notes=__doc__,native_render=native,native_width=640)
    rec.capture();trace=[];deaths=[];kept=[];incoming_held=[];switches=0;guide_stage='lead';guard_active=False
    def contained(p):
        return wall.x-55<p.x<wall.x and abs(p.y-600)<height/2-5
    def chosen(p):
        seen=[o for o in p.observe(agents=list(env.agents),edges=list(env.edges)) if o['type']=='Agent']
        return min(seen,key=lambda o:o['distance'])['id'] if seen else None
    for tick in range(round(seconds*10)):
        if holder.agent_id not in env.agents_dict:break
        before=list(env.agents);pairs=[];decisions={};action_t=env.time
        for a in sorted(before,key=lambda a:a.agent_id):
            target=(a.x,a.y);look=None;speed=a.speed;rule='hold'
            if a is holder:
                yy=600
                if kind=='corner_track' and original:
                    yy=max(wall.y+15,min(wall.y+height-15,original.y))
                target=(anchor[0],yy)
            elif a is guard:
                rule='bodyguard_station'
                if kind=='guard_active':
                    # Exact position is a diagnostic privilege. Keep the guard
                    # nearer the threat than the holder, then retreat outward.
                    guard_active=True;rule='draw_intruder_away'
                    # Leave sideways, keeping the predator between neither
                    # guard and destination nor guard and holder.
                    target=(a.x,min(1140,a.y+30))
                    d=math.dist((a.x,a.y),(incoming.x,incoming.y))
                    if d<65 and a.energy>a.max_energy/5+8:speed=a.sprint_speed
                    if incoming.resting or d>150:target=(a.x,a.y)
                    look=math.atan2(incoming.y-a.y,incoming.x-a.x)+.02
            elif a is guide:
                d=math.dist((a.x,a.y),(incoming.x,incoming.y))
                target=(front,600)
                if kind.startswith('corner'):target=(front,600+side*min(20,height/2-20))
                # Deliberately keep the incoming predator close enough for a
                # transfer. Waiting and sprinting use real native energy.
                if d>75 and math.dist((a.x,a.y),target)>10:target=(a.x,a.y)
                if d<45 and a.energy>a.max_energy/5+8:speed=a.sprint_speed
                rule='sacrifice_at_front' if math.dist((a.x,a.y),target)<2 else 'guide_to_front'
                look=math.atan2(incoming.y-a.y,incoming.x-a.x)+.02
            dx,dy=target[0]-a.x,target[1]-a.y
            distance=min(speed,math.hypot(dx,dy));direction=wrap(math.atan2(dy,dx)-a.direction) if distance else 0
            turn=wrap(look-a.direction) if look is not None else (direction if distance else 0)
            action=ActionRequest(agent_id=a.agent_id,move_distance=distance,move_direction=direction,turn_angle=turn,spawn_agent=False)
            pairs.append((a.agent_id,action));decisions[a.agent_id]=dict(rule=rule)
            env.agent_step(a.agent_id,distance,direction,turn,False)
        target_before=chosen(original) if original else None
        if guide and original and target_before==guide.agent_id:switches+=1
        env.non_agent_step(.1)
        for a in before:
            if a.agent_id not in env.agents_dict:deaths.append(dict(agent=a.agent_id,time=round(env.time,1),cause='depletion' if a.energy<=0 else 'capture'))
        kept.append(contained(original) if original else True)
        incoming_held.append(contained(incoming))
        if tick%10==0:trace.append(dict(time=round(env.time,1),old_contained=kept[-1],incoming_contained=incoming_held[-1],
            old_target=target_before,old_position=[original.x,original.y] if original else None,
            new_position=[incoming.x,incoming.y],holder_alive=holder.agent_id in env.agents_dict))
        rec.capture(pairs,decisions,action_t=action_t)
    rec.save(path,reason='horizon' if env.time>=seconds-.01 else 'holder lost')
    alive=holder.agent_id in env.agents_dict;tail=min(150,len(kept))
    row=dict(kind=kind,width=width,height=height,gap=gap,side=side,heading=heading,
        intruder_distance=intruder_distance,initial_rest=resting,seconds=round(env.time,1),
        holder_alive=alive,guard_alive=guard.agent_id in env.agents_dict,
        guide_alive=guide.agent_id in env.agents_dict if guide else None,deaths=deaths,
        old_first_exit=next((round((i+1)*.1,1) for i,k in enumerate(kept) if not k),None),
        old_tail=sum(kept[-tail:])/tail,new_tail=sum(incoming_held[-tail:])/tail,
        old_targeted_guide_ticks=switches,
        success=bool(alive and env.time>=seconds-.01 and sum(kept[-tail:])/tail>.95 and
            (kind.startswith('guard') or sum(incoming_held[-tail:])/tail>.95)),
        replay=str(path),trace=trace)
    return row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['delivery','guard'],default='delivery')
    ap.add_argument('--native',action='store_true');args=ap.parse_args();verify_source();rows=[]
    configs=[]
    if args.suite=='delivery':
        for width,height in [(30,70),(35,100)]:
            for kind in ['first','second_straight','corner','corner_track']:
                for side in [-1,1]:configs.append(dict(kind=kind,width=width,height=height,side=side,heading=side*.15))
    else:
        for distance in [20,70,160,240]:
            for resting in [False,True]:
                for kind in ['guard_stationary','guard_active']:
                    configs.append(dict(kind=kind,intruder_distance=distance,resting=resting))
    for c in configs:
        r=run(**c,native=args.native);rows.append(r)
        print({k:r[k] for k in ['kind','width','side','intruder_distance','initial_rest','seconds','success','old_first_exit','old_tail','new_tail','old_targeted_guide_ticks']},flush=True)
        (OUT/f'protocol-{args.suite}-v2.json').write_text(json.dumps(dict(scope=__doc__,runs=rows),indent=2,default=lambda v:v.item())+'\n')


if __name__=='__main__':main()
