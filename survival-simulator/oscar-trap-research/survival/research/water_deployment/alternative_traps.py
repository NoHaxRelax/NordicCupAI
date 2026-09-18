"""Size-selective obstacle passages, distinct from thin-wall/water control.

Privileged geometry supplies the refuge and approach. Actions only walk toward
that fixed goal, then hold still. Native collision, sensing, energy and aging.
"""
from common import *
import argparse
import random
import numpy as np
from src.core import SimulationCore
from src.elements.agent import Agent
from src.elements.predator import Predator
from src.elements.obstacle import Obstacle
from src.utils.DTOs import ActionRequest
from predator_control import controlled_env
from recording import start_recording, finish_recording

def sites(env):
    out=[]
    for i,a in enumerate(env.obstacles):
        for j,b in enumerate(env.obstacles):
            if i==j:continue
            for axis in [0,1]:
                ax,ay,aw,ah=(a.x,a.y,a.width,a.height) if axis==0 else (a.y,a.x,a.height,a.width)
                bx,by,bw,bh=(b.x,b.y,b.width,b.height) if axis==0 else (b.y,b.x,b.height,b.width)
                gap=bx-(ax+aw);lo=max(ay,by);hi=min(ay+ah,by+bh)
                if not (11<=gap<=19 and hi-lo>=55):continue
                cross=ax+aw+gap/2
                for side in [1,-1]:
                    mouth=lo if side==1 else hi
                    def xy(u,v):return [u,v] if axis==0 else [v,u]
                    goal=xy(cross,mouth+side*30)
                    start=xy(cross,mouth-side*15)
                    pred=xy(cross,mouth-side*50)
                    if env._in_obstacle(goal,5,env.obstacles) or env._in_obstacle(start,5,env.obstacles) or env._in_obstacle(pred,10,env.obstacles):continue
                    # Check the founder route continuously at 2-unit spacing.
                    points=np.linspace(start,goal,31)
                    if any(env._in_obstacle(p,5,env.obstacles) for p in points):continue
                    fullspeed=all(env.biome_map[int(p[0]),int(p[1])].move_penalty==1 for p in points)
                    heading=math.atan2(goal[1]-start[1],goal[0]-start[0])
                    out.append(dict(obstacles=[i,j],axis=axis,gap=gap,length=hi-lo,goal=goal,
                        start=start,predator=pred,heading=heading,fullspeed_route=fullspeed,
                        mouth=xy(cross,mouth)))
    return out

def setup(case):
    if case.get('native'):
        env=SimulationCore(seed=case['seed']).env;s=case['site']
    else:
        env=controlled_env();env.rng=random.Random(case.get('seed',173));g=case['gap'];L=case.get('length',80)
        env.obstacles=[Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200),
            Obstacle(800-g/2-60,500,60,L),Obstacle(800+g/2,500,60,L)]
        env.edges=set()
        for o in env.obstacles:env.edges.update((min(a,b),max(a,b)) for a,b in o.edges)
        s=dict(goal=[800,530],start=[800,485],predator=[800,450],mouth=[800,500],heading=math.pi/2,gap=g,length=L)
    env.agents=[];env.agents_dict={};env.agent_observations={};env.predators=[];env._next_agent_id=1
    a=Agent(*s['start'],energy=150,rng=env.rng);a.agent_id=0;a.direction=s['heading']
    env.agents=[a];env.agents_dict={0:a}
    for i in range(case.get('predators',1)):
        tangent=np.array([math.cos(s['heading']),math.sin(s['heading'])])
        normal=np.array([-tangent[1],tangent[0]])
        pxy=np.array(s['predator'])-i*12*tangent+i*5*normal
        p=Predator(*pxy,energy=102,rng=env.rng);p.resting=False
        p.direction=s['heading']+case.get('heading_offset',0);env.predators.append(p)
    env.spawn_predator=lambda *a,**kw:None
    env._update_spatial_grid()
    return env,s

def run(case,record=None,*,reproduction=None):
    env,site=setup(case);a=env.agents[0]
    goal=np.array(site['goal']);estimated=np.array(site['start'],float);heading=site['heading']
    acquired=None;capture=None;first_escape=None;contained=attended=joint=active=activejoint=0
    elapsed_after=0;movement=0;maxerr=0;food=0;trace=[]
    recorder,replay_path=start_recording(env,case,policy='Size-selective gap refuge',
        demonstration=record,reproduction=reproduction,
        notes='Known gap geometry and initial pose; straight walking entry followed by stationary bait. Original engine and native energy. Arranged predator approach.')
    for tick in range(round(case.get('seconds',60)*10)):
        if a not in env.agents:break
        s=env.get_agent_state(0);t=env.time
        if case.get('release_s') is not None and t>=case['release_s']:
            goal=np.array(site['goal'])+200*np.array([math.cos(heading),math.sin(heading)])
        delta=goal-estimated
        angle=math.atan2(delta[1],delta[0]);penalty={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1)
        move=min(10,float(np.linalg.norm(delta))/penalty)
        if np.linalg.norm(delta)<.01:
            move=0
            if acquired is None:acquired=t
        action=ActionRequest(agent_id=0,move_distance=move,move_direction=wrap(angle-heading),turn_angle=0,spawn_agent=False)
        oldenergy=a.energy;oldage=a.age
        env.agent_step(0,move,action.move_direction,0,False);movement+=oldenergy-a.energy
        estimated+=move*penalty*np.array([math.cos(angle),math.sin(angle)])
        pre=a.energy
        selections=[];wasactive=[]
        for p in env.predators:
            obs=p.observe(agents=env._get_local_agents(p),edges=env._get_local_edges(p))
            selections.append(any(o['type']=='Agent' and o['id']==0 for o in obs))
            wasactive.append(not p.resting)
        env.non_agent_step(.1)
        drain=.1*env.biome_map[int(a.x),int(a.y)].energy_drain_rate+(.01*a.age if a.age>a.max_age else 0)
        if a.age>oldage:food+=max(0,a.energy-pre+drain)
        maxerr=max(maxerr,math.dist(estimated,[a.x,a.y]))
        # Containment means close to the refuge mouth, not merely not eaten.
        confined=all(math.dist([p.x,p.y],site['mouth'])<75 for p in env.predators)
        attention=all(selections)
        if acquired is not None:
            elapsed_after+=1;contained+=confined;attended+=attention;joint+=confined and attention
            active+=any(wasactive);activejoint+=any(wasactive) and confined and attention
            if not confined and first_escape is None:first_escape=env.time
        if a not in env.agents and a.energy>0:capture=env.time
        if tick%10==0:trace.append(dict(t=round(env.time,2),bait=[a.x,a.y,a.energy],
            predators=[[p.x,p.y,p.energy,p.resting] for p in env.predators],confined=confined,attention=attention))
        recorder.capture([(0,action)],{0:dict(rule='Walk through size-selective gap' if move else 'Hold in narrow refuge')},[s],t)
    r=dict(case=case,site=site,elapsed_s=env.time,acquired_s=acquired,capture_s=capture,
        alive=a in env.agents,energy_remaining=a.energy,movement_energy=movement,food_energy=food,
        first_escape_s=first_escape,after_acquisition_s=elapsed_after/10,
        containment_fraction=contained/max(1,elapsed_after),attention_fraction=attended/max(1,elapsed_after),
        joint_fraction=joint/max(1,elapsed_after),active_joint_fraction=activejoint/max(1,active),
        maximum_odometry_error=maxerr,trace=trace)
    r['recording']=finish_recording(recorder,replay_path,
        reason='requested horizon reached' if a in env.agents else 'bait captured or depleted')
    return r

def main(mode,start=1,stop=21,include_slow=False):
    rows=[]
    survey_name='alternative-gap-survey'+(f'-{start}-{stop-1}' if (start,stop)!=(1,21) else '')
    if mode=='controlled':
        cases=[dict(gap=g,length=L,heading_offset=h,seconds=60) for g in [12,15,18,22] for L in [60,90] for h in [-.3,.3]]
        cases += [dict(gap=15,length=90,heading_offset=h,seconds=60,predators=3) for h in [-.3,.3]]
    elif mode=='survey':
        for seed in range(start,stop):
            env=SimulationCore(seed=seed).env;s=sites(env)
            rows.append(dict(seed=seed,sites=s));print(seed,len(s),sum(x['fullspeed_route'] for x in s),flush=True)
            save(survey_name,rows)
        return
    else:
        maps=json.loads((OUT/(survey_name+'.json')).read_text())['data']
        cases=[]
        for m in maps:
            candidates=[s for s in m['sites'] if include_slow or s['fullspeed_route']]
            if candidates:
                s=max(candidates,key=lambda s:s['length'])
                for h in [-.3,.3]:cases.append(dict(native=True,seed=m['seed'],site=s,heading_offset=h,seconds=60))
    for case in cases:
        r=run(case);rows.append(r)
        print(case.get('seed'),r['site']['gap'],case['heading_offset'],r['elapsed_s'],r['capture_s'],r['joint_fraction'],flush=True)
        suffix=(f'-{start}-{stop-1}' if (start,stop)!=(1,21) else '')+('-all-terrain' if include_slow else '')
        save('alternative-gap-'+mode+suffix,rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['controlled','survey','native'])
    p.add_argument('--start',type=int,default=1);p.add_argument('--stop',type=int,default=21)
    p.add_argument('--include-slow',action='store_true');a=p.parse_args();main(a.mode,a.start,a.stop,a.include_slow)
