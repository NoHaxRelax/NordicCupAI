"""Executable local-only experiments, original engine physics and cached DTOs.
Scenario construction is privileged; controllers never receive construction state.
"""
from __future__ import annotations
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import sys,math,json,random,hashlib,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'vendor/survival-simulator'),str(ROOT/'research'),str(ROOT/'debugger')]
from sprinter_decoys import fixture,add_agent,add_pred,observe,ripe_food,DecoyController
from importlib import import_module
CONTROLLER_MODULE=os.environ.get("DEDICATED_CONTROLLER","controller")
BaitController=import_module(CONTROLLER_MODULE).BaitController
from src.elements.tree import Tree
from src.elements.obstacle import Obstacle
from src.elements.biome import Swamp_biome,Desert_biome
from src.core import SimulationCore
from simple_policies import SimplePolicy
from recorder import ReplayRecorder
from recording import RunRecording,unique_id
OUT=ROOT/'results/dedicated_sprinter'
COMMIT='acfc31a4003a5f91bf11032a02cd98c178ddbd7e'

LOADED_HASHES={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
def hashes():
    return LOADED_HASHES

def build(c):
    seed=c.get('seed',0);e=fixture(seed)
    x,y=c.get('position',[800,600]);angle=c.get('rotation',0);gap=c.get('gap',60)
    a=add_agent(e,x,y,c.get('energy',500),c.get('capacity',500));a.direction=angle+math.pi
    a.speed=c.get('walk',10);a.sprint_speed=c.get('sprint',20);a.max_age=c.get('max_age',90)
    p=add_pred(e,x-gap*math.cos(angle),y-gap*math.sin(angle),c.get('pred_energy',102),angle+c.get('heading',0))
    def tr(px,py):return x+px*math.cos(angle)-py*math.sin(angle),y+px*math.sin(angle)+py*math.cos(angle)
    food=c.get('food','none')
    if food=='line':ripe_food(e,[tr(q,0) for q in (140,300,460,620)])
    if food=='ring':ripe_food(e,[tr(100*math.cos(k*math.pi/6),100*math.sin(k*math.pi/6)) for k in range(12)])
    if food=='patch':
        rr=random.Random(seed+9000)
        ripe_food(e,[tr(rr.uniform(-160,160),rr.uniform(-160,160)) for _ in range(24)])
    if food=='orchard':
        # Finite ordinary trees, original fruit birth, growth, rotting and tree death.
        for k in range(8):
            tx,ty=tr(100*math.cos(k*math.pi/4),100*math.sin(k*math.pi/4))
            tree=Tree(tx,ty);tree.grow(20);e.trees.append(tree)
        e._update_tree_grid()
        ripe_food(e,[tr(65*math.cos(k*math.pi/3),65*math.sin(k*math.pi/3)) for k in range(6)])
    if c.get('obstacle'):
        ox,oy=tr(130,-30);e.obstacles.append(Obstacle(ox,oy,40,100))
        e.edges=set(edge for o in e.obstacles for edge in o.edges)
    if c.get('terrain')=='split':e.biome_map[int(x+60):,:]=Swamp_biome()
    if c.get('terrain')=='desert':e.biome_map[:,:]=Desert_biome()
    workers=[]
    for k in range(c.get('workers',0)):
        wx,wy=tr(-350+40*k,150)
        workers.append(add_agent(e,wx,wy,150))
    e._update_spatial_grid()
    return e,a,p,workers


def run(c,policy='predictive',seconds=120,record=None,*,controller_module=None,reproduction=None,native=None):
    e,a,p,workers=build(c);aid=a.agent_id
    if policy.startswith('adaptive'):
        old=DecoyController(mode='adaptive',request=20,trigger=60,initial={aid:(0,0,0)})
        ctrl=None
    else:ctrl=import_module(controller_module or CONTROLLER_MODULE).BaitController(food=policy!='nofood',**c.get('controller',{}))
    states=observe(e)
    origin=(a.x,a.y,a.direction);pose_errors=[];retention_start=0.;continuous_until_loss=None
    rec=RunRecording(e,policy=policy,module=controller_module or CONTROLLER_MODULE,
        seed=c.get('seed',0),scenario='arranged-single-bait',parameters=c,horizon=seconds,
        label=record,reproduction=reproduction,native=native)
    m=dict(scenario=c,policy=policy,horizon=seconds,food_eaten=0,food_received=0.,nutrition=0.,movement_energy=0.,turn_energy=0.,passive_energy=0.,target_ticks=0,active_ticks=0,longest_active_loss_s=0.,longest_retained_s=0.,first_loss_s=None,first_cutoff_s=None,rest_cycles=0,worker_deaths={},min_gap=1e9,trace=[])
    loss=retained=0;occupation_start=None;longest_occupation=0.;acquired=None;rest_tp=rest_fp=rest_fn=0;rest=p.resting;lastenergy=a.energy;dead=False;death=None
    for tick in range(round(seconds*10)):
        inputs=states;action_t=e.time
        if ctrl:
            state=next(s for s in states if s['agent_id']==aid)
            action,why=ctrl.step(state,e.time)
            inferred=ctrl.last_decision.get('inferred_rest',False)
            actual=p.resting and p.energy<=100
            rest_tp+=inferred and actual;rest_fp+=inferred and not actual;rest_fn+=not inferred and actual
            actions=[(aid,action)];decisions={aid:why+' '+json.dumps(ctrl.last_decision)}
        else:actions,decisions=old(states,e.time)
        for wid in [w.agent_id for w in workers if w in e.agents]:
            if wid not in dict(actions):
                from src.utils.DTOs import ActionRequest
                actions.append((wid,ActionRequest(agent_id=wid,move_distance=0,move_direction=0,turn_angle=0,spawn_agent=False)))
        ea=a.energy
        for ident,action in actions:
            e.agent_step(ident,action.move_distance,action.move_direction,action.turn_angle,False)
            if ident==aid:
                tc=min(math.pi,abs(action.turn_angle))/(2*math.pi)
                m['turn_energy']+=tc;m['movement_energy']+=ea-a.energy-tc
        if ctrl:
            lx,ly=ctrl.track.xy
            ex=origin[0]+lx*math.cos(origin[2])-ly*math.sin(origin[2])
            ey=origin[1]+lx*math.sin(origin[2])+ly*math.cos(origin[2])
            pose_errors.append(math.dist((ex,ey),(a.x,a.y)))
        obs=p.observe(agents=e._get_local_agents(p),edges=e._get_local_edges(p))
        observed=[o for o in obs if o['type']=='Agent']
        target=min(observed,key=lambda o:o['distance'])['id'] if observed else None
        # A resting predator with >100 energy wakes and acts this tick.
        active=not p.resting or p.energy>100
        if active:
            m['active_ticks']+=1
            if target==aid:
                m['target_ticks']+=1;retained+=.1;loss=0
                if acquired is None:acquired=e.time
                if occupation_start is None:occupation_start=e.time
                m['longest_retained_s']=max(m['longest_retained_s'],retained)
            else:
                occupation_start=None
                retained=0;loss+=.1;m['longest_active_loss_s']=max(m['longest_active_loss_s'],loss)
                if m['first_loss_s'] is None:m['first_loss_s']=e.time
        if occupation_start is not None:longest_occupation=max(longest_occupation,e.time+.1-occupation_start)
        prevfruit=set(e.fruits);ea=a.energy
        passive=.1+(.01*(a.age+.1) if ea>.1 and a.age+.1>a.max_age else 0)
        m['passive_energy']+=passive
        e.non_agent_step(.1)
        eaten=[f for f in prevfruit if f not in e.fruits and f.age<=100 and math.dist((f.x,f.y),(a.x,a.y))<a.size+f.radius]
        m['food_eaten']+=len(eaten);m['nutrition']+=sum(f.energy for f in eaten)
        m['food_received']+=max(0.,a.energy-ea+passive)
        m['min_gap']=min(m['min_gap'],math.dist((a.x,a.y),(p.x,p.y)))
        if not rest and p.resting:m['rest_cycles']+=1
        rest=p.resting
        if m['first_cutoff_s'] is None and a.energy<a.max_energy/5:m['first_cutoff_s']=e.time
        for w in workers:
            if w not in e.agents and w.agent_id not in m['worker_deaths']:m['worker_deaths'][w.agent_id]=e.time
        states=[e.get_agent_state(o.agent_id) for o in e.agents]
        rec.capture(actions,decisions,inputs,action_t)
        if tick%10==0:
            m['trace'].append(dict(t=round(e.time,2),bait=[round(a.x,2),round(a.y,2),round(a.energy,2)],predator=[round(p.x,2),round(p.y,2),round(p.energy,2)],resting=p.resting,target=target,food=len(e.fruits),workers=sum(w in e.agents for w in workers)))
        if a not in e.agents:dead=True;death='capture' if a.energy>0 else 'starvation';break
    m.update(accounting_version="source-order-v2",acquisition_time=acquired,longest_occupation_s=longest_occupation,rest_inference=dict(true_positive=int(rest_tp),false_positive=int(rest_fp),false_negative=int(rest_fn)),duration=round(e.time,2),alive=not dead,death=death,remaining_energy=a.energy,score=e.score,workers_alive=sum(w in e.agents for w in workers),target_fraction=m['target_ticks']/max(1,m['active_ticks']),controller_stats=ctrl.stats if ctrl else {},max_pose_error=max(pose_errors,default=0.),mean_pose_error=sum(pose_errors)/max(1,len(pose_errors)),continuous_retention_s=e.time if m['first_loss_s'] is None else m['first_loss_s'],replacement_windows_per_minute=60/e.time if dead else None)
    for k,v in list(m.items()):
        if isinstance(v,float):m[k]=round(v,4)
    rec.finish(m,death or 'horizon reached')
    return m


def save(name,rows):
    OUT.mkdir(exist_ok=True,parents=True)
    tag=os.environ.get('DEDICATED_RUN_TAG','')
    name=name+('-'+tag if tag else '')
    directory=OUT/'summaries';directory.mkdir(exist_ok=True)
    path=directory/(unique_id(name)+'.json')
    path.write_text(json.dumps(dict(source_commit=COMMIT,controller_module=CONTROLLER_MODULE,hashes=hashes(),runs=rows),indent=2,allow_nan=False)+'\n')
    return path

def compact(r):return {k:r[k] for k in ('scenario','policy','duration','alive','target_fraction','food_eaten','movement_energy','remaining_energy','workers_alive','death')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',default='screen');args=ap.parse_args();rows=[]
    if args.phase=='screen':
        for margin in (20,24,28):
            for orbit in (0,3,10):
                c=dict(energy=500,controller=dict(margin=margin,orbit_weight=orbit))
                r=run(c,seconds=60);rows.append(r);print(json.dumps(compact(r)),flush=True)
    elif args.phase=='food':
        for food in ('none','line','ring','patch','orchard'):
            for policy in ('adaptive','nofood','predictive'):
                c=dict(energy=500,food=food)
                r=run(c,policy,120);rows.append(r);print(json.dumps(compact(r)),flush=True)
    elif args.phase=='traits':
        for walk,capacity in ((10,500),(10,300),(12,500),(12,300),(14,500),(16,500)):
            for energy in (75,150, min(400,capacity)):
                for food in ('none','orchard'):
                    c=dict(walk=walk,capacity=capacity,energy=energy,food=food)
                    r=run(c,seconds=140);rows.append(r);print(json.dumps(compact(r)),flush=True)
    elif args.phase=='heldout':
        rng=random.Random(178301)
        cases=[]
        for k in range(20):
            cases.append(dict(seed=101+k,position=[rng.uniform(350,1200),rng.uniform(300,900)],rotation=rng.uniform(-math.pi,math.pi),heading=rng.uniform(-math.pi,math.pi),gap=rng.uniform(48,90),pred_energy=rng.uniform(40,200),energy=rng.choice([150,250,400]),food='orchard',workers=2,max_age=rng.uniform(60,120),terrain='split' if k%5==0 else 'desert' if k%5==1 else 'forest',obstacle=k%4==0))
        for c in cases:
            for policy in ('adaptive','nofood','predictive'):
                r=run(c,policy,120);rows.append(r);print(json.dumps(compact(r)),flush=True)
    elif args.phase=='replays':
        for name,c,policy in [('orchard-founder',dict(energy=500,food='orchard'), 'predictive'),('orchard-specialist',dict(energy=150,walk=12,capacity=300,food='orchard'),'predictive')]:
            r=run(c,policy,140,name);rows.append(r);print(json.dumps(compact(r)),flush=True)
    save(args.phase,rows)
if __name__=='__main__':main()
