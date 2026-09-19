"""Record every native C++ policy/game tick for the existing sprite viewer."""
import argparse,gzip,hashlib,json,math,sys,time
from pathlib import Path
from fast_entrapment_game import atom,world,edges_from
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nightsim
import pygame
from src.core import SimulationCore as Background

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seed',type=int,default=1);p.add_argument('--seconds',type=float,default=3000)
p.add_argument('--configs',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.configs.read_text())[a.label]
for key in ('oracle_trees','trap_mode','hide_mode','refuge_mode','test_freeze','test_pred_life'):
    if cfg.get(key,0):p.error(f'{key} must be disabled for an honest avoidance run')
f=a.out
if f.exists():p.error('output already exists')
f.mkdir(parents=True);(f/'chunks').mkdir()
sim=nightsim.SimulationCore(seed=a.seed,predators=True);e=sim._engine
states=sim.step([])['observations'];e.policy_init(nightsim.seed_key(a.seed),cfg)
bg=Background(seed=a.seed);surface=bg.env.static_surface.copy();surface.blit(bg.env.shadow_surface,(0,0));surface.blit(bg.env.obstacle_surface,(0,0));pygame.image.save(surface,f/'background.png');del bg
atom(f/'static.json',dict(width=sim.env.width,height=sim.env.height,edges=edges_from([(o.x,o.y,o.width,o.height) for o in sim.env.obstacles])))
source=[ROOT/'nightsim/_nengine.cpp',ROOT/'nightsim/_npolicy.hpp',ROOT/'models/avoidance/native_corner.hpp',ROOT/'models/avoidance/native_escape_search.hpp']
atom(f/'manifest.json',dict(seed=a.seed,label=a.label,config=cfg,horizon=a.seconds,engine='C++ policy and engine',policy_inputs='Own public stats and cached observations; observed Orchard map',sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source}))
started=time.monotonic();chunk=[];events=[];history=[];tick=0;previous={};counts={};peak=0;seen=set();prior_counters={}
while True:
    now=sim.env.time;terminal=not states or now>=a.seconds
    raw=[] if terminal else e.policy_act()
    acts=[(aid,dict(agent_id=aid,move_distance=d,move_direction=h,turn_angle=t,spawn_agent=sp)) for aid,d,h,t,sp in raw]
    debug=e.corner_debug();steering={};poses={};roles={s['agent_id']:'gatherer' for s in states}
    for m in e.policy_minds():poses[m[0]]=dict(position=[m[2],m[3]],heading=m[4],uncertainty=None,group=m[1])
    for aid,phase,px,py,h,want,error,x,y,theta,cx,cy in debug['agents']:
        dx,dy=px-x,py-y;pos=[dx*math.cos(theta)+dy*math.sin(theta),-dx*math.sin(theta)+dy*math.cos(theta)]
        steering[aid]=dict(phase='align' if phase==1 else 'release',target_position=pos,target_heading=h-theta,desired_heading=want-theta,heading_error=error,corner=[cx,cy])
        roles[aid]='steering_predator'
        if aid not in previous:events.append(dict(time=now,kind='steering_started',agent=aid))
    for key in ('aligned','missed','aborted'):
        if debug[key]>prior_counters.get(key,0):events.append(dict(time=now,kind=key+'_exit' if key!='aborted' else 'steering_aborted'))
    previous=steering;prior_counters=debug.copy()
    ne=[]
    for kind,t,aid,age,energy in e.pop_events():
        counts[kind]=counts.get(kind,0)+1;ne.append(dict(kind=kind,time=t,agent=aid,age=age,energy=energy))
    seen.update(roles);peak=max(peak,len(states))
    ev=dict(agents=len(states),predators=len(sim.env.predators),fruit=len(sim.env.fruits),score=sim.env.score,near_bait=0,held30=0,bait_present_estimated=False,bait_energy=[])
    if tick%10==0:history.append(dict(time=now,**ev))
    policy=dict(phase='orchard_corner_'+a.label,roles=roles,steering=steering,estimated_agents=poses,guides={},site=None,bait=None,incoming=None,metrics={k:v for k,v in debug.items() if k!='agents'})
    chunk.append(dict(tick=tick,time=now,world=world(sim.env),policy=policy,input=states,actions=[d for _,d in acts],evaluation=ev,native_events=ne))
    if (tick+1)%100==0 or terminal:
        with gzip.open(f/'chunks'/f'{tick//100:05d}.json.gz','wt',compresslevel=1) as out:json.dump(chunk,out,separators=(',',':'))
        chunk=[]
        summary=dict(seed=a.seed,status='complete' if terminal else 'running',frames=tick+1,sim_time=now,runtime_seconds=time.monotonic()-started,score=sim.env.score,final_agents=len(states),peak_agents=peak,total_agents_seen=len(seen),predators=len(sim.env.predators),events=events,history=history,policy_metrics=policy['metrics'],native_evaluation=counts)
        atom(f/'summary.json',summary)
    if terminal:break
    states=sim.step(acts)['observations'];tick+=1
print(json.dumps({k:v for k,v in summary.items() if k not in ('events','history')},indent=2))
