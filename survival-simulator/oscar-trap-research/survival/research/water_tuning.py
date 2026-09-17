"""Water bait tuning against unchanged upstream simulation mechanics.

Creature starts and map knowledge are explicitly arranged diagnostic fixtures.
The native-site trials retain genuine generated terrain, obstacles and food.
Default controllers know current predator geometry; `observed=True` replaces
that input with cached public observations and estimated pose, not hidden energy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
import sys
import time

from predator_control import ROOT, controlled_env, add_agent, add_predator, wrap
import numpy as np
from src.utils.DTOs import ActionRequest
from src.elements.fruit import Fruit
from src.elements.obstacle import Obstacle


@dataclass(frozen=True)
class Config:
    mode: str = "lead"
    lead: float = 55
    reserve: float = 12
    switch: float = 3
    max_move: float = 7
    lookahead: float = 3
    margin: float = 2
    track_y: float = 0
    escape: float = 27
    face: str = "back"
    turn_y: float = 920
    observed: bool = False
    rest_wait: bool = False
    hold_ticks: int = 3
    y_bias: float = 0
    prediction: str = "none"
    arc_cos: float = .7


@dataclass(frozen=True)
class Scenario:
    width: int = 60
    seconds: float = 60
    energy: float = 150
    heading: float = math.pi/2
    offset_x: float = 0
    offset_y: float = 0
    n_agents: int = 2
    max_age: float = 120
    food_spacing: float = 0
    seed: int = 173
    outside_distance: float = 0
    secondary_offset: float = 60
    shore_launch: bool = False


class Controller:
    def __init__(self, config, scenario, agents):
        self.c=config;self.s=scenario
        self.role=0
        self.direction=1
        self.switches=0
        self.last_switch=-999
        self.positions={a.agent_id:np.array([a.x,a.y],float) for a in agents}
        self.directions={a.agent_id:a.direction for a in agents}
        self.last_pred=None
        self.last_raw_pred=None
        self.estimate_age=0
        self.last_decisions={}
        self.acquisition_complete=0 if not scenario.outside_distance else None
        self.predator_entered=False
        self.acquisition_energy=None

    def acquisition_actions(self, env, pose, tick, states):
        """Cross one bait through the river, then hand pursuit to shore two.

        Both agents start on the predator's land side. Entry detection and all
        motion use the same cached-observation pose estimate as containment.
        """
        px,py,_=pose
        c,s=self.c,self.s
        edge=800+s.width/2
        if px<edge-.5:self.predator_entered=True
        by_id={x['agent_id']:x for x in states}
        if 0 not in by_id or 1 not in by_id:return []
        left=800-s.width/2-c.margin
        a0=self.positions[0];a1=self.positions[1]
        if a0[0]<=left-15 and abs(a1[1]-600)<=25 and 800-s.width/2<px<edge:
            self.acquisition_complete=tick/10
            self.acquisition_energy={aid:by_id[aid]['energy'] for aid in [0,1]}
            self.role=1;self.last_switch=tick
            return None
        actions=[];self.last_decisions={}
        penalties={'river':.3,'swamp':.5,'desert':.8}
        for aid in [0,1]:
            apos=self.positions[aid];facing=self.directions[aid]
            if aid==0:
                if s.shore_launch and apos[0]>edge+.001:
                    move=min(20,apos[0]-edge)
                    rule='Step onto dry shoreline launch point'
                elif apos[0]>left:
                    move=min(20,(apos[0]-left)/penalties.get(by_id[aid]['biome'],1))
                    rule='Cross river to draw predator into water'
                else:
                    move=20
                    rule='Retreat to transfer pursuit to the other bank'
                absolute=math.pi
            else:
                dy=600-apos[1]
                move=min(10,abs(dy)) if self.predator_entered else 0
                absolute=math.pi/2 if dy>0 else -math.pi/2
                rule='Approach second shore after predator enters water' if self.predator_entered else 'Wait clear of initial chase'
            to_pred=math.atan2(py-apos[1],px-apos[0])
            turn=wrap(to_pred-facing)
            if abs(turn)<.05:turn=0
            actions.append((aid,ActionRequest(agent_id=aid,move_distance=move,move_direction=wrap(absolute-facing),turn_angle=turn,spawn_agent=False)))
            self.last_decisions[aid]=dict(rule=rule,detail='Acquisition uses mapped straight-river geometry and cached public predator observations.')
        return actions

    def estimate(self, states):
        estimates=[]
        for s in states:
            aid=s['agent_id'];o=[o for o in s['observations'] if o['type']=='Predator']
            if not o:continue
            obs=min(o,key=lambda x:x['distance'])
            theta=self.directions[aid]+obs['angle']
            pos=self.positions[aid]+obs['distance']*np.array([math.cos(theta),math.sin(theta)])
            heading=wrap(theta+math.pi-obs['rel_dir'])
            estimates.append((pos,heading))
        if estimates:
            pos=np.mean([x[0] for x in estimates],axis=0)
            angle=math.atan2(sum(math.sin(x[1]) for x in estimates),sum(math.cos(x[1]) for x in estimates))
            raw=np.array([pos[0],pos[1],angle])
            predicted=raw.copy()
            if self.last_raw_pred is not None and self.c.prediction!='none':
                velocity=raw[:2]-self.last_raw_pred[:2]
                if self.c.prediction=='velocity':predicted[:2]+=velocity
                else:
                    # Model the deterministic chase turn from observed pose
                    # and our own tracked bait positions. Speed/rest is inferred
                    # from consecutive observed positions, never read from p.
                    speed=float(np.linalg.norm(velocity))
                    possible=[]
                    for aid,apos in self.positions.items():
                        rel=apos-raw[:2];d=float(np.linalg.norm(rel));q=wrap(math.atan2(rel[1],rel[0])-angle)
                        if d<=60 or (d<=250 and abs(q)<=math.pi/6):possible.append((d,q))
                    if possible and speed>.01:
                        d,q=min(possible)
                        turn=max(-.3,min(.3,q*.5)) if abs(q)>.05 else q
                        predicted[:2]+=speed*np.array([math.cos(angle+turn),math.sin(angle+turn)])
                        predicted[2]=wrap(angle+(turn if abs(q)>.05 else 0))
            self.last_raw_pred=raw
            self.last_pred=predicted;self.estimate_age=0
        else:
            self.estimate_age+=1
        return self.last_pred

    def actions(self, env, p, tick):
        c,s=self.c,self.s
        states=[env.get_agent_state(a.agent_id) for a in env.agents]
        observed=self.estimate(states)
        if c.observed:
            if observed is None:
                return [(a.agent_id,ActionRequest(agent_id=a.agent_id,move_distance=0,move_direction=0,turn_angle=.2,spawn_agent=False)) for a in env.agents]
            px,py,pheading=observed
        else:px,py,pheading=p.x,p.y,p.direction
        if self.acquisition_complete is None:
            acquiring=self.acquisition_actions(env,(px,py,pheading),tick,states)
            if acquiring is not None:return acquiring
        if py>c.turn_y:self.direction=-1
        elif py<1200-c.turn_y:self.direction=1
        pred_x=px+c.lookahead*4.5*math.cos(pheading)
        desired=self.role
        if c.mode=='arc':
            side=-1 if self.role==0 else 1
            if side*math.cos(pheading)>c.arc_cos:desired=1-self.role
        else:
            if pred_x>800+c.switch:desired=0
            elif pred_x<800-c.switch:desired=1
        if desired!=self.role and tick-self.last_switch>=c.hold_ticks:
            self.role=desired;self.switches+=1;self.last_switch=tick
        B=s.width/2+c.margin
        goals={}
        if c.mode=="lead":
            for a in env.agents:
                side=-1 if a.agent_id%2==0 else 1
                goals[a.agent_id]=np.array([800+side*B,py+self.direction*(c.lead+(0 if a.agent_id%2==self.role else c.reserve))])
        elif c.mode in ["cross","cross_side","arc"]:
            # Keep chosen bait at its shore, put the rival just farther away
            # from the predator. The formula uses the actual other-bait
            # distance, rather than a fixed large retreat on every switch.
            role_side=-1 if self.role==0 else 1
            ry=600+c.track_y*(py-600)
            if c.y_bias:
                ry=py+max(-c.y_bias,min(c.y_bias,600-py))
            active=np.array([800+role_side*B,ry])
            d=math.hypot(active[0]-px,active[1]-py)+c.reserve
            other_side=-role_side
            if c.mode in ['cross','arc']:
                retreat_x=px+other_side*math.sqrt(max(0,d*d-(ry-py)**2))
                reserve=np.array([800+other_side*max(B,other_side*(retreat_x-800)),ry])
            else:
                bx=800+other_side*B
                dy=math.sqrt(max(0,d*d-(bx-px)**2))
                reserve=np.array([bx,py+self.direction*dy])
            for a in env.agents:goals[a.agent_id]=active if a.agent_id%2==self.role else reserve
        elif c.mode=="single_bank":
            for a in env.agents:goals[a.agent_id]=np.array([800+B,py+self.direction*c.lead])
        else:raise ValueError(c.mode)
        actions=[]
        self.last_decisions={}
        for a in env.agents:
            aid=a.agent_id
            pos=self.positions[aid] if c.observed else np.array([a.x,a.y])
            facing=self.directions[aid] if c.observed else a.direction
            delta=goals[aid]-pos
            requested=min(c.max_move,float(np.linalg.norm(delta)))
            rule="Lead predator along bank" if c.mode=="lead" else "Shift nearest bait"
            to_pred=np.array([px,py])-pos
            distance=float(np.linalg.norm(to_pred))
            # Failed-switch escape is allowed to continue beyond the nominal
            # goal. It uses actual legal sprinting and low-energy restrictions.
            if distance<c.escape:
                away=-to_pred/max(.001,distance)
                if c.mode in ["lead","cross_side","single_bank"]:
                    # Prefer along-bank escape while retaining dry ground.
                    away=np.array([0,self.direction],float)
                    if abs(to_pred[0])<10:away=-to_pred/max(.001,distance)
                requested=min(a.sprint_speed,max(c.max_move,c.escape-distance+4.6))
                delta=away;rule="Escape failed handoff"
            if c.rest_wait and self.last_pred is not None:
                # Oracle rest is intentionally absent. Settling at the fixed
                # requested lead already pauses bait movement during sleep.
                pass
            move_angle=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-8 else facing
            if c.face=="back":desired_face=math.atan2(to_pred[1],to_pred[0])
            elif c.face=="away":desired_face=math.atan2(-to_pred[1],-to_pred[0])
            elif c.face=="relay":
                desired_face=math.atan2(-to_pred[1],-to_pred[0]) if aid%2==self.role else math.atan2(to_pred[1],to_pred[0])
            else:desired_face=facing
            turn=wrap(desired_face-facing)
            # Avoid paying for microscopic turns when already seeing the target.
            if abs(turn)<.05:turn=0
            action=ActionRequest(agent_id=aid,move_distance=requested,move_direction=wrap(move_angle-facing),turn_angle=turn,spawn_agent=False)
            actions.append((aid,action))
            self.last_decisions[aid]=dict(rule=rule,detail=f"Family {c.mode}; target role {self.role}; estimated predator distance {distance:.1f}; move request {requested:.2f}.")
        return actions

    def integrate(self, actions, states):
        """Dead reckoning only, with biome modifier from public own state."""
        state={s['agent_id']:s for s in states}
        penalties={'river':.3,'swamp':.5,'desert':.8}
        for aid,action in actions:
            s=state[aid]
            move=min(action.move_distance,s['sprint_speed'])
            if s['energy']<s['max_energy']/5:move=min(move,s['speed'])
            move*=penalties.get(s['biome'],1)
            heading=self.directions[aid]+action.move_direction
            self.positions[aid]+=move*np.array([math.cos(heading),math.sin(heading)])
            self.directions[aid]+=action.turn_angle


def setup(config,scenario):
    env=controlled_env(scenario.width)
    env.rng=random.Random(scenario.seed)
    # Match the actual engine's physical 30-unit boundaries. The older
    # mechanics helper contains only boundary edges and otherwise permits
    # center clamping at x/y=size; that is inadequate for long lure trials.
    env.obstacles=[Obstacle(0,0,1600,30),Obstacle(0,1170,1600,30),
                   Obstacle(0,0,30,1200),Obstacle(1570,0,30,1200)]
    env.edges=set()
    for o in env.obstacles:
        env.edges.update((min(a,b),max(a,b)) for a,b in o.edges)
    env._update_spatial_grid()
    if scenario.width%2:
        raise ValueError('Use even river widths to avoid half-pixel fixture ambiguity')
    px=800+scenario.offset_x if not scenario.outside_distance else 800+scenario.width/2+scenario.outside_distance
    heading=scenario.heading if not scenario.outside_distance else math.pi+scenario.heading
    p=add_predator(env,x=px,y=600+scenario.offset_y,heading=heading)
    B=scenario.width/2+config.margin
    agents=[]
    for i in range(scenario.n_agents):
        side=-1 if i==0 else 1
        if config.mode=='single_bank':side=1
        y=600+(config.lead if config.mode in ['lead','single_bank'] else 0)
        ax=800+side*B
        if scenario.outside_distance:
            ax=800+scenario.width/2+(10 if i==0 else config.margin)
            y=600+(0 if i==0 else scenario.secondary_offset)
        a=add_agent(env,ax,y,energy=scenario.energy)
        a.max_age=scenario.max_age
        a.direction=math.atan2(p.y-a.y,p.x-a.x) if config.face in ['back','relay'] else math.pi/2
        if scenario.outside_distance:a.direction=math.atan2(p.y-a.y,p.x-a.x)
        agents.append(a)
    if scenario.food_spacing:
        # Explicit prepared orchard fixture, never claimed as naturally present.
        for x in [800-B,800+B]:
            for y in np.arange(100,1101,scenario.food_spacing):
                f=Fruit(float(x),float(y),radius=9);f.energy=60;f.age=40
                f.fruit_id=env._next_fruit_id;env._next_fruit_id+=1
                env.fruits.append(f);env.fruits_dict[f.fruit_id]=f
        env._update_fruit_grid()
    for a in agents:
        env.agent_observations[a.agent_id]=a.observe(agents=agents,predators=[p],edges=list(env.edges))
    return env,p,Controller(config,scenario,agents)


def run(config,scenario,*,record=None,stop_on_capture=True,trace=False):
    env,p,controller=setup(config,scenario)
    recorder=None
    if record:
        sys.path.insert(0,str(ROOT/'debugger'))
        from recorder import ReplayRecorder
        recorder=ReplayRecorder(env,title=f'Water bait tuning · {config.mode} · width {scenario.width}',policy=f'water {config.mode}',
            seed=scenario.seed,every=4,scenario='controlled',native_render=True,notes='Synthetic straight river and arranged starts; native engine physics. '+
            ('Actions use cached public observations plus known fixture coordinates.' if config.observed else 'Controller uses privileged current predator geometry; diagnostic upper bound.'))
        recorder.capture(force=True)
    metrics=dict(water=0,attention=0,contained_attention=0,active=0,active_water=0,active_attention=0,active_contained_attention=0,correct_target=0)
    first_exit=first_capture=first_starvation=None
    first_entry=0 if 800-scenario.width/2<=p.x<800+scenario.width/2 else None
    after_acquisition=after_acquisition_water=after_acquisition_attention=0
    two_bait_ticks=two_bait_water=two_bait_attention=0
    first_loss=None
    min_gap=math.inf
    requests=energy_cost=0
    rows=[]
    extrema=[p.x,p.x,p.y,p.y]
    actual_target_switches=0;last_target=None;max_observation_age=0
    for tick in range(round(scenario.seconds*10)):
        old=list(env.agents)
        if not old:break
        inputs=[env.get_agent_state(a.agent_id) for a in old]
        old_time=env.time
        actions=controller.actions(env,p,tick)
        before_energy=sum(a.energy for a in old)
        controller.integrate(actions,inputs)
        for aid,a in actions:
            env.agent_step(aid,a.move_distance,a.move_direction,a.turn_angle,False)
            requests+=a.move_distance
        energy_cost+=before_energy-sum(a.energy for a in old)
        seen=p.observe(agents=list(env.agents),edges=list(env.edges))
        detectable=[o for o in seen if o['type']=='Agent']
        attention=bool(detectable)
        selected=min(detectable,key=lambda o:o['distance'])['id'] if detectable else None
        active=not p.resting
        if active and selected is not None:
            actual_target_switches+=last_target is not None and last_target!=selected
            last_target=selected
        env.non_agent_step(.1)
        water=800-scenario.width/2<=p.x<800+scenario.width/2
        if water and first_entry is None:first_entry=env.time
        if controller.acquisition_complete is not None:
            after_acquisition+=1;after_acquisition_water+=water;after_acquisition_attention+=attention
        metrics['water']+=water;metrics['attention']+=attention;metrics['contained_attention']+=water and attention
        metrics['active']+=active;metrics['active_water']+=active and water
        metrics['active_attention']+=active and attention
        metrics['active_contained_attention']+=active and attention and water
        metrics['correct_target']+=active and selected==controller.role
        extrema=[min(extrema[0],p.x),max(extrema[1],p.x),min(extrema[2],p.y),max(extrema[3],p.y)]
        max_observation_age=max(max_observation_age,controller.estimate_age)
        if not water and first_exit is None and first_entry is not None:first_exit=env.time
        min_gap=min(min_gap,*[math.hypot(a.x-p.x,a.y-p.y) for a in old])
        lost=[a for a in old if a not in env.agents]
        for a in lost:
            if a.energy<=0:
                if first_starvation is None:first_starvation=env.time
            elif first_capture is None:first_capture=env.time
        if lost and first_loss is None:first_loss=env.time
        if controller.acquisition_complete is not None and len(env.agents)>=2:
            two_bait_ticks+=1;two_bait_water+=water;two_bait_attention+=attention
        if trace and (tick<40 or tick%10==0):
            rows.append(dict(t=round(env.time,2),p=[round(p.x,3),round(p.y,3),round(p.direction,3),round(p.energy,3),bool(p.resting)],
                             a=[[a.agent_id,round(a.x,3),round(a.y,3),round(a.energy,3)] for a in env.agents],role=controller.role,
                             observed_age=controller.estimate_age,water=bool(water),attention=attention))
        if recorder:recorder.capture(actions,controller.last_decisions,inputs,old_time)
        if first_capture is not None and stop_on_capture:break
    elapsed=int(round(env.time*10))
    result=dict(config=asdict(config),scenario=asdict(scenario),elapsed_s=elapsed/10,alive=len(env.agents),
                first_exit_s=first_exit,first_capture_s=first_capture,first_starvation_s=first_starvation,
                water_fraction=metrics['water']/elapsed,attention_fraction=metrics['attention']/elapsed,
                contained_attention_fraction=metrics['contained_attention']/elapsed,
                active_water_fraction=metrics['active_water']/max(1,metrics['active']),
                active_attention_fraction=metrics['active_attention']/max(1,metrics['active']),
                active_contained_attention_fraction=metrics['active_contained_attention']/max(1,metrics['active']),
                intended_target_fraction=metrics['correct_target']/max(1,metrics['active']),
                minimum_gap=min_gap,energy_remaining=[a.energy for a in env.agents],movement_turn_energy=energy_cost,
                requested_distance=requests,switches=controller.switches,predator_target_switches=actual_target_switches,
                predator_bounds=extrema,maximum_observation_age_ticks=max_observation_age)
    result.update(first_entry_s=first_entry,acquisition_complete_s=controller.acquisition_complete,
                  after_acquisition_s=after_acquisition/10,
                  after_acquisition_water_fraction=after_acquisition_water/max(1,after_acquisition),
                  after_acquisition_attention_fraction=after_acquisition_attention/max(1,after_acquisition),
                  acquisition_energy=controller.acquisition_energy,first_loss_s=first_loss,
                  two_bait_containment_s=two_bait_ticks/10,
                  two_bait_water_fraction=two_bait_water/max(1,two_bait_ticks),
                  two_bait_attention_fraction=two_bait_attention/max(1,two_bait_ticks))
    result['physical_boundaries']=True
    if trace:result['trace']=rows
    if recorder:result['recording']=dict(path=str(record),summary=recorder.save(record,overwrite=True))
    return result


def quality(result):
    # Seconds of attention while inside water, strongly penalize capture.
    return result['elapsed_s']*result['contained_attention_fraction']-(30 if result['first_capture_s'] else 0)


def save(name,data):
    path=ROOT/'results'/f'water-tuning-{name}.json'
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
    print(path,flush=True)


def initial_search():
    configs=[]
    for mode in ['cross','cross_side']:
        for reserve in [1,4,10]:
            for look in [0,3,6]:
                for track in [0,1]:
                    configs.append(Config(mode=mode,reserve=reserve,lookahead=look,track_y=track,max_move=10,face='fixed',escape=26))
    for lead in [30,50,70,100]:
        for reserve in [6,15,30]:
            for face in ['back','relay']:
                configs.append(Config(mode='lead',lead=lead,reserve=reserve,lookahead=0,max_move=7,face=face,escape=26))
    configs += [Config(mode='single_bank',lead=lead,max_move=speed,face=face)
                for lead in [35,60,100] for speed in [4.6,7] for face in ['back','away']]
    rows=[]
    start=time.time()
    for i,c in enumerate(configs):
        scenarios=[Scenario(width=w,seconds=35,heading=(0 if c.mode.startswith('cross') else math.pi/2),n_agents=1 if c.mode=='single_bank' else 2) for w in [40,60,80]]
        rs=[run(c,s) for s in scenarios]
        rows.append(dict(config=asdict(c),mean_quality=sum(map(quality,rs))/len(rs),runs=rs))
        if i%10==0:print(f'{i+1}/{len(configs)} configs {time.time()-start:.1f}s',flush=True)
    rows.sort(key=lambda x:x['mean_quality'],reverse=True)
    save('search',rows)
    for row in rows[:10]:print(row['mean_quality'],row['config'],flush=True)


def tuning_search():
    rng=random.Random(719)
    candidates=[]
    for hold in [3,6,9,12,15]:
        for track in [0,.5,1]:
            for look in [0,2,3,5]:
                for reserve in [1,3,6]:
                    candidates.append(Config(mode='cross',reserve=reserve,lookahead=look,track_y=track,hold_ticks=hold,max_move=10,face='fixed',escape=23))
    for _ in range(100):
        candidates.append(Config(mode='cross',reserve=rng.choice([.2,1,2,4,8]),switch=rng.choice([0,2,4,7]),
            max_move=rng.choice([4.6,6,8,10]),lookahead=rng.choice([0,2,3,4,6]),
            track_y=1,y_bias=rng.choice([8,15,25]),hold_ticks=rng.choice([4,8,11,14]),face='fixed',escape=rng.choice([20,24,28])))
    rows=[];start=time.time()
    train=[Scenario(width=60,seconds=40,heading=0),Scenario(width=80,seconds=40,heading=0),
           Scenario(width=60,seconds=40,heading=.3,offset_x=3,offset_y=8)]
    for i,c in enumerate(candidates):
        runs=[run(c,s) for s in train]
        # Reward keeping both baits alive economically, to avoid fitting
        # starvation exactly at a short test's stopping point.
        q=sum(quality(x)+.025*sum(x['energy_remaining']) for x in runs)/len(runs)
        rows.append(dict(config=asdict(c),mean_quality=q,runs=runs))
        if i%30==0:print(f'{i+1}/{len(candidates)} {time.time()-start:.1f}s',flush=True)
    rows.sort(key=lambda x:x['mean_quality'],reverse=True)
    save('tune',rows)
    for row in rows[:10]:print(row['mean_quality'],row['config'],flush=True)


def observed_search():
    rng=random.Random(9119)
    candidates=[Config(mode='cross',reserve=12,lookahead=0,track_y=0,max_move=10,face='fixed',escape=22,observed=True,prediction='velocity')]
    for _ in range(320):
        candidates.append(Config(mode=rng.choice(['cross','arc']),reserve=rng.choice([6,8,10,12,14,18]),
            lookahead=rng.choice([0,1,2,3]),switch=rng.choice([0,3,6,9]),track_y=rng.choice([0,.3,.7,1]),
            hold_ticks=rng.choice([3,4,6,9]),max_move=rng.choice([4.6,7,9,10]),face='fixed',escape=rng.choice([20,22,25,28]),
            observed=True,prediction=rng.choice(['velocity','model']),arc_cos=rng.choice([.6,.8,.9,.97])))
    rows=[];start=time.time()
    training=[Scenario(width=50,seconds=60,heading=0),Scenario(width=60,seconds=60,heading=0),Scenario(width=50,seconds=60,heading=.2,offset_y=5)]
    for i,c in enumerate(candidates):
        runs=[run(c,s) for s in training]
        scores=[quality(x)+.025*sum(x['energy_remaining']) for x in runs]
        rows.append(dict(config=asdict(c),mean_quality=sum(scores)/len(scores),runs=runs))
        if i%40==0:print(f'{i+1}/{len(candidates)} {time.time()-start:.1f}s',flush=True)
    rows.sort(key=lambda x:x['mean_quality'],reverse=True);save('observed-tune',rows)
    for row in rows[:10]:print(row['mean_quality'],row['config'],flush=True)


def held_out():
    trained=json.loads((ROOT/'results'/'water-tuning-observed-tune.json').read_text())
    candidates={
        'observed_robust':Config(**trained[0]['config']),
        'observed_long_narrow':Config(mode='cross',reserve=12,lookahead=0,track_y=0,max_move=10,face='fixed',escape=22,observed=True,prediction='velocity'),
        'privileged_arc':Config(mode='arc',arc_cos=.9,track_y=0,reserve=8,face='fixed',max_move=10,escape=24,hold_ticks=3),
    }
    conditions=[(-.35,-3,-12),(.35,3,12),(math.pi/2,0,8),(-math.pi/2,0,-8)]
    rows=[]
    for name,c in candidates.items():
        for width in [30,40,44,48,54,56,64,72,90,100,120,160,200]:
            for heading,dx,dy in conditions:
                s=Scenario(width=width,seconds=60,heading=heading,offset_x=dx,offset_y=dy,max_age=90,seed=983)
                rows.append(dict(candidate=name,**run(c,s)))
        print(name,'complete',flush=True)
    save('held-out',dict(selection='Configs frozen after initial/tuning training; no selection using these rows.',runs=rows))
    extras=[]
    for name,c in candidates.items():
        for energy in [75,150,300]:
            for width in [40,50,60,80]:
                extras.append(dict(candidate=name,**run(c,Scenario(width=width,energy=energy,seconds=120,heading=0))))
    save('energy-lifetime',extras)
    c=candidates['observed_long_narrow']
    success=run(c,Scenario(width=50,heading=0,seconds=64),trace=True,
        record=ROOT/'results'/'water-tuning-observed-success.replay.json')
    failure=run(c,Scenario(width=100,heading=.35,offset_y=12,seconds=15),trace=True,
        record=ROOT/'results'/'water-tuning-observed-failure.replay.json')
    save('recorded-cases',dict(success=success,failure=failure))


def resource_probes():
    c=Config(**json.loads((ROOT/'results'/'water-tuning-observed-tune.json').read_text())[0]['config'])
    fruit=[];late=[];speed=[]
    for spacing in [0,100,50,25]:
        for width in [44,50,56,60]:
            for heading,dy in [(0,0),(.35,12),(-.35,-12)]:
                fruit.append(run(c,Scenario(width=width,seconds=120,heading=heading,offset_y=dy,food_spacing=spacing,max_age=90)))
    save('nearby-fruit',fruit)
    for width in [44,50,56,60]:
        for heading,dy in [(0,0),(.35,12),(-.35,-12)]:
            late.append(run(c,Scenario(width=width,seconds=120,heading=heading,offset_y=dy,food_spacing=25,max_age=120)))
    save('fruit-late-age',late)
    for cap in [4.6,6,7,8,9,10]:
        for width in [44,50,56,60]:
            speed.append(run(replace(c,max_move=cap),Scenario(width=width,heading=0,seconds=60)))
    save('speed-ablation',speed)
    fed=run(c,Scenario(width=50,heading=.35,offset_y=12,seconds=120,food_spacing=25,max_age=120),trace=True,
            record=ROOT/'results'/'water-tuning-fed-success.replay.json')
    save('fed-recorded-case',fed)


def geometry_survey(seeds=range(1,41)):
    """Generated source maps, not random straight-river fixtures.

    Find a conservative120-unit straight corridor with shared40-unit water
    interior, <=64-unit total bank envelope, and full-speed banks. This is an
    oracle availability survey, not a policy observation or acquisition test.
    """
    from src.elements.biome import Map_generator
    results=[]
    for seed in seeds:
        biome=Map_generator(1600,1200,random.Random(seed),num_biomes=10,num_rivers=1).generate()
        river=np.fromiter((b.type=='river' for b in biome.flat),dtype=bool,count=biome.size).reshape(biome.shape)
        sites=[]
        for axis in [0,1]:
            mask=river if axis==0 else river.T
            terrain=biome if axis==0 else biome.T
            for along in range(100,mask.shape[1]-100,20):
                line=mask[:,along]
                changes=np.diff(np.r_[False,line,False].astype(int))
                for left,right in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                    if not 40<=right-left<=64:continue
                    center=(left+right)//2
                    bounds=[]
                    for row in range(along-60,along+61,15):
                        if not mask[center,row]:break
                        lo=center
                        while lo>0 and mask[lo-1,row]:lo-=1
                        hi=center+1
                        while hi<mask.shape[0] and mask[hi,row]:hi+=1
                        bounds.append((lo,hi))
                    if len(bounds)!=9:continue
                    lo=min(b[0] for b in bounds);hi=max(b[1] for b in bounds)
                    interior=min(b[1] for b in bounds)-max(b[0] for b in bounds)
                    if hi-lo>64 or interior<40 or lo<50 or hi>mask.shape[0]-50:continue
                    bank_types=set()
                    okay=True
                    for row in range(along-60,along+61,15):
                        for x in [lo-2,hi+2]:
                            b=terrain[x,row];bank_types.add(b.type)
                            if b.move_penalty!=1 or b.type=='river':okay=False
                    if okay:
                        sites.append(dict(axis='vertical' if axis==0 else 'horizontal',across_center=(lo+hi)/2,
                            along_center=along,shore_envelope_width=hi-lo,common_water_width=interior,
                            banks=sorted(bank_types),across_shores=[lo-2,hi+2]))
        unique={}
        for site in sites:
            key=(site['axis'],round(site['across_center']/100),round(site['along_center']/100))
            unique.setdefault(key,site)
        results.append(dict(seed=seed,river_area_fraction=float(river.mean()),candidate_samples=len(sites),
                            approximate_distinct_sites=list(unique.values())))
        print(seed,len(sites),len(unique),flush=True)
    save('generated-geometry',dict(criteria='Oracle survey; 120-long corridor; common water width>=40; bank envelope<=64; both banks full-speed; physical random obstacles not yet checked.',maps=results))


def acquisition_trials():
    """Reproduce the frozen dry-shore launch acquisition validation."""
    c=Config(**json.loads((ROOT/'results'/'water-tuning-observed-tune.json').read_text())[0]['config'])
    rows=[]
    for width in [46,54,58]:
        for distance,heading,dy in [(37,.15,8),(53,-.15,-8),(67,.3,14),(83,-.3,-14)]:
            s=Scenario(width=width,seconds=60,energy=150,heading=heading,offset_y=dy,
                       max_age=90,seed=2197,outside_distance=distance,secondary_offset=75,shore_launch=True)
            rows.append(run(c,s,stop_on_capture=True))
    save('acquisition-final-held-out',rows)


def native_site_trials():
    """Use genuine generated biomes/obstacles/trees/fruits at surveyed sites.

    Creature starts and shared local coordinate frame are arranged. Native
    food/tree spawning continues; only additional predator spawns are disabled.
    """
    from src.core import SimulationCore
    survey=json.loads((ROOT/'results'/'water-tuning-generated-geometry.json').read_text())
    c=Config(**json.loads((ROOT/'results'/'water-tuning-observed-tune.json').read_text())[0]['config'])
    results=[]
    for m in survey['maps']:
        for site in m['approximate_distinct_sites']:
            env=SimulationCore(seed=m['seed']).env
            rotation=0 if site['axis']=='vertical' else math.pi/2
            cx=site['across_center'] if not rotation else site['along_center']
            cy=site['along_center'] if not rotation else site['across_center']
            def world(x,y):
                dx,dy=x-800,y-600
                return cx+dx*math.cos(rotation)-dy*math.sin(rotation),cy+dx*math.sin(rotation)+dy*math.cos(rotation)
            width=site['shore_envelope_width'];B=width/2+c.margin
            positions=[world(800-B,600),world(800+B,600)]
            blocked=[env._in_obstacle(pos,5,env.obstacles) for pos in positions]
            pblocked=env._in_obstacle((cx,cy),10,env.obstacles)
            initial_tree_counts=[sum(math.hypot(t.x-pos[0],t.y-pos[1])<100 for t in env.trees) for pos in positions]
            initial_fruit_counts=[sum(math.hypot(f.x-pos[0],f.y-pos[1])<100 for f in env.fruits) for pos in positions]
            base=dict(seed=m['seed'],site=site,blocked_agent_starts=blocked,blocked_predator_start=pblocked,
                      initial_trees_within100=initial_tree_counts,initial_fruits_within100=initial_fruit_counts)
            if any(blocked) or pblocked:
                results.append(dict(**base,skipped='Creature start intersects generated obstacle'));print('blocked',m['seed'],site,flush=True);continue
            env.agents=[];env.agents_dict={};env.agent_observations={};env.predators=[];env._next_agent_id=0
            agents=[add_agent(env,*xy,energy=150) for xy in positions]
            for a in agents:a.direction=rotation+math.pi/2;a.max_age=90
            p=add_predator(env,cx,cy,heading=rotation)
            env.spawn_predator=lambda *args,**kwargs:None
            env._update_spatial_grid()
            for a in agents:
                env.agent_observations[a.agent_id]=a.observe(agents=env._get_local_agents(a),predators=env._get_local_predators(a),edges=env._get_local_edges(a))
            scenario=Scenario(width=width,seconds=120,max_age=90,seed=m['seed'])
            controller=Controller(c,scenario,agents)
            for i,a in enumerate(agents):
                controller.positions[a.agent_id]=np.array([800+(-B if i==0 else B),600.])
                controller.directions[a.agent_id]=math.pi/2
            water_ticks=attention_ticks=0;first_loss=None;first_capture=None;first_starvation=None
            food_energy=0;trace=[]
            for tick in range(1200):
                before=list(env.agents)
                if not before:break
                states=[env.get_agent_state(a.agent_id) for a in before]
                actions=controller.actions(env,p,tick);controller.integrate(actions,states)
                for aid,action in actions:env.agent_step(aid,action.move_distance,action.move_direction,action.turn_angle,False)
                expected=0
                for a in before:
                    biome=env.biome_map[int(a.x),int(a.y)]
                    remaining=a.energy-.1*biome.energy_drain_rate
                    if remaining>0 and a.age+.1>a.max_age:remaining-=.01*(a.age+.1)
                    expected+=remaining
                seen=p.observe(agents=env._get_local_agents(p),edges=env._get_local_edges(p))
                attention=any(o['type']=='Agent' for o in seen)
                env.non_agent_step(.1)
                gain=sum(a.energy for a in before)-expected
                if gain>1:food_energy+=gain
                water=env.biome_map[int(p.x),int(p.y)].type=='river'
                water_ticks+=water;attention_ticks+=attention
                lost=[a for a in before if a not in env.agents]
                if lost:
                    first_loss=env.time
                    if any(a.energy>0 for a in lost):first_capture=env.time
                    else:first_starvation=env.time
                    break
                if tick%10==0:trace.append(dict(t=env.time,p=[p.x,p.y],energy=[a.energy for a in env.agents],water=water))
            n=int(round(env.time*10))
            row=dict(**base,elapsed_s=env.time,alive=len(env.agents),first_loss_s=first_loss,first_capture_s=first_capture,
                     first_starvation_s=first_starvation,water_fraction=water_ticks/max(1,n),attention_fraction=attention_ticks/max(1,n),
                     fruit_energy_collected=food_energy,energy_remaining=[a.energy for a in env.agents],trace=trace,
                     qualification='Native generated map and food dynamics; arranged creature starts and mapped local frame; additional predator spawns disabled.')
            results.append(row);print(m['seed'],site['axis'],site['along_center'],env.time,row['water_fraction'],food_energy,flush=True)
    save('native-sites',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--search',action='store_true')
    parser.add_argument('--tune',action='store_true')
    parser.add_argument('--observe-tune',action='store_true')
    parser.add_argument('--held-out',action='store_true')
    parser.add_argument('--resources',action='store_true')
    parser.add_argument('--geometry',action='store_true')
    parser.add_argument('--acquisition',action='store_true')
    parser.add_argument('--native-sites',action='store_true')
    args=parser.parse_args()
    if args.search:initial_search()
    elif args.tune:tuning_search()
    elif args.observe_tune:observed_search()
    elif args.held_out:held_out()
    elif args.resources:resource_probes()
    elif args.geometry:geometry_survey()
    elif args.acquisition:acquisition_trials()
    elif args.native_sites:native_site_trials()
    else:print(json.dumps(run(Config(),Scenario(),trace=True),indent=2))
