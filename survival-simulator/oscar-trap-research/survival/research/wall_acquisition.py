"""Local wall-trap acquisition experiments using exact upstream physics.

The planner has privileged obstacle and current creature coordinates. These trials
measure whether acquisition is physically useful, not an observation-only policy.
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import argparse
import heapq
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/survival-simulator'))
from predator_control import controlled_env,add_agent,add_predator,wrap
from src.elements.obstacle import Obstacle
from shapely.geometry import LineString,box


def path_to(start, goal, wall, clearance=7):
    x0,y0=wall.x-clearance,wall.y-clearance
    x1,y1=wall.x+wall.width+clearance,wall.y+wall.height+clearance
    solid=box(x0+1e-5,y0+1e-5,x1-1e-5,y1-1e-5)
    if not LineString([start,goal]).intersects(solid):return [goal]
    nodes=[start,goal,(x0,y0),(x0,y1),(x1,y0),(x1,y1)]
    queue=[(0,0,[])];seen=set()
    while queue:
        cost,i,path=heapq.heappop(queue)
        if i in seen:continue
        seen.add(i)
        if i==1:return [nodes[n] for n in path]
        for j in range(1,len(nodes)):
            if j==i or j in seen:continue
            if not LineString([nodes[i],nodes[j]]).intersects(solid):
                heapq.heappush(queue,(cost+math.dist(nodes[i],nodes[j]),j,path+[j]))
    return [goal]


def fixture(family,offset,width,height,energy,gap,heading_delta=0):
    env=controlled_env();wall=Obstacle(800-width/2,600-height/2,width=width,height=height)
    env.obstacles=[wall,Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
    for w in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in w.edges)
    if family=='front':ax,ay=740,600+offset;px,py=ax-gap,ay
    elif family=='above':ax,ay=800+offset,600-height/2-35;px,py=ax-gap,ay
    elif family=='behind':ax,ay=860,600+offset;px,py=800-width/2-gap/2,ay
    else:ax,ay=760,600-height/2-25+offset;px,py=ax-gap/math.sqrt(2),ay-gap/math.sqrt(2)
    agent=add_agent(env,ax,ay,energy=energy)
    p=add_predator(env,px,py,heading=math.atan2(ay-py,ax-px)+heading_delta)
    agent.direction=math.atan2(py-ay,px-ax)
    env._update_spatial_grid()
    return env,wall,agent,p


def trial(case, seconds=60, record=None):
    env,wall,a,p=fixture(**{k:v for k,v in case.items() if k!='mode'})
    mode=case['mode'];side=1 if p.x<800 else -1
    goal=(800+side*(wall.width/2+7),600)
    waypoints=path_to((a.x,a.y),goal,wall)
    acquisition=None;first_death=None;path_length=0;rows=[];holding=[];sensed=[]
    recorder=None
    if record:
        sys.path.insert(0,str(ROOT/'debugger'))
        from recorder import ReplayRecorder
        recorder=ReplayRecorder(env,title=f'Wall acquisition · {mode}',policy='geometry-aware wall route',seed=1729,every=1,scenario='controlled',notes='Arranged approach; exact coordinates supplied to planner. Original engine physics. Tests acquisition with finite energy, not ordinary-observation deployment.')
        recorder.capture()
    from src.utils.DTOs import ActionRequest
    for tick in range(round(seconds*10)):
        if not env.agents:break
        old=(a.x,a.y);old_time=env.time
        while waypoints and math.dist((a.x,a.y),waypoints[0])<1:
            waypoints.pop(0)
        distance=0;direction=0;turn=0
        gap=math.dist((a.x,a.y),(p.x,p.y))
        if waypoints:
            gx,gy=waypoints[0];distance=min(a.speed,math.hypot(gx-a.x,gy-a.y))
            if mode!='walk' and gap<100 and a.energy>a.max_energy/5+4:
                distance=min(16 if mode=='minimal' else 20,math.hypot(gx-a.x,gy-a.y))
            direction=wrap(math.atan2(gy-a.y,gx-a.x)-a.direction)
            turn=wrap(math.atan2(p.y-a.y,p.x-a.x)-a.direction)
        action=ActionRequest(agent_id=a.agent_id,move_distance=distance,move_direction=direction,turn_angle=turn,spawn_agent=False)
        env.agent_step(a.agent_id,distance,direction,turn,False)
        path_length+=math.dist(old,(a.x,a.y))
        if acquisition is None and math.dist((a.x,a.y),goal)<2:acquisition=env.time
        env.non_agent_step(.1)
        alive=a.agent_id in env.agents_dict
        if not alive and first_death is None:first_death=env.time
        held=(alive and math.dist((a.x,a.y),goal)<8 and (p.x-800)*side<0 and abs(p.x-800)<wall.width/2+45 and abs(p.y-600)<wall.height/2+10)
        holding.append(held)
        observed=p.observe(agents=[a] if alive else [],edges=list(env.edges))
        sensed.append(any(o['type']=='Agent' for o in observed))
        if tick%10==0:rows.append([round(env.time,1),round(a.x,2),round(a.y,2),round(a.energy,2),round(p.x,2),round(p.y,2),p.resting,held])
        if recorder:recorder.capture([(a.agent_id,action)],{a.agent_id:{'rule':'Acquire wall' if waypoints else 'Hold wall bait','detail':'Route around the inflated obstacle to its far face; conserve movement once positioned.'}},action_t=old_time)
    tail=holding[-200:]
    success=(a.agent_id in env.agents_dict and env.time>=seconds-.01 and sum(tail)/len(tail)>.9 and sum(sensed[-200:])/len(sensed[-200:])>.9)
    result=dict(**case,seconds=round(env.time,1),alive=a.agent_id in env.agents_dict,success=success,
                acquired_at=None if acquisition is None else round(acquisition,1),death_at=None if first_death is None else round(first_death,1),
                remaining_energy=round(a.energy,2),energy_spent=round(case['energy']-a.energy,2),distance_moved=round(path_length,2),
                held_fraction=sum(holding)/max(1,len(holding)),tail_hold_fraction=sum(tail)/max(1,len(tail)),
                tail_attention_fraction=sum(sensed[-200:])/max(1,len(sensed[-200:])),trace=rows)
    if recorder:
        recorder.save(record,reason='capture' if not result['alive'] else 'requested duration reached',overwrite=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/wall-acquisition.json');p.add_argument('--quick',action='store_true');args=p.parse_args()
    start=time.perf_counter();results=[]
    cases=[]
    for mode in ['walk','minimal','sprint']:
      for energy in [150,500]:
       for family in ['front','above','behind','diagonal']:
        for offset in [-20,20]:
         for gap in ([110] if args.quick else [80,140]):
          cases.append(dict(mode=mode,energy=energy,family=family,offset=offset,gap=gap,width=32,height=80,heading_delta=0))
    for index,case in enumerate(cases):
        result=trial(case);results.append(result)
        if (index+1)%24==0:print(f'{index+1}/{len(cases)} cases; {time.perf_counter()-start:.1f}s',flush=True)
    validation=[]
    for width,height in [(30,70),(35,100),(38,70),(32,50)]:
      for family in ['front','above','behind','diagonal']:
       for heading in [-.3,.3]:
        validation.append(trial(dict(mode='sprint',energy=150,family=family,offset=0,gap=110,width=width,height=height,heading_delta=heading)))
    output=dict(scope='Privileged geometry with unmodified engine physics; no food or new predators; success requires60s survival and >90% final20s containment AND attention.',train=results,validation=validation)
    args.output.write_text(json.dumps(output,indent=2,default=lambda v:v.item())+'\n')
    for mode in ['walk','minimal','sprint']:
        for energy in [150,500]:
            rows=[r for r in results if r['mode']==mode and r['energy']==energy]
            print(mode,energy,sum(r['success'] for r in rows),'/',len(rows))
    print('held-out geometry',sum(r['success'] for r in validation),'/',len(validation))

if __name__=='__main__':main()
